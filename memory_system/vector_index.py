"""
HNSW vector index wrapper — Section 4.2 of the architecture.

Backend priority: usearch → hnswlib (both implement HNSW).
Persistence: saves the index binary + an id_map JSON sidecar.
One index instance per embedding model version (tracked via the file path).
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Callable, Optional

# Detect available backend
try:
    from usearch.index import Index as _UsearchIndex
    _BACKEND = "usearch"
except ImportError:
    _UsearchIndex = None
    try:
        import hnswlib as _hnswlib
        _BACKEND = "hnswlib"
    except ImportError:
        _hnswlib = None
        _BACKEND = None


def _require_backend() -> None:
    if _BACKEND is None:
        raise ImportError(
            "No vector search backend found. Install one of:\n"
            "  pip install usearch\n"
            "  pip install hnswlib"
        )


class VectorIndex:
    """
    HNSW vector index with fragment-ID ↔ internal-ID mapping,
    optional post-retrieval filter (over-fetches 3× to maintain recall),
    and disk persistence (index binary + JSON id_map sidecar).
    """

    def __init__(
        self,
        dim: int = 1024,
        metric: str = "cosine",
        max_elements: int = 500_000,
        persist_path: Optional[str | Path] = None,
    ):
        _require_backend()
        self.dim = dim
        self.metric = metric
        self.max_elements = max_elements
        self.persist_path = Path(persist_path) if persist_path else None

        self._backend = _BACKEND
        self._index = None               # initialised lazily
        self._id_map: dict[int, str] = {}   # int_id → fragment_id
        self._rev_map: dict[str, int] = {}  # fragment_id → int_id
        self._next_id: int = 0
        self._initialized = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def init_fresh(self) -> None:
        if self._backend == "usearch":
            self._index = _UsearchIndex(
                ndim=self.dim,
                metric="cos",
                dtype="f32",
                connectivity=32,        # M in hnswlib terms
                expansion_add=200,      # ef_construction
                expansion_search=128,   # ef at query time
            )
        else:
            self._index = _hnswlib.Index(space=self.metric, dim=self.dim)
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=200,
                M=32,
            )
            self._index.set_ef(128)
        self._initialized = True

    def load(self) -> bool:
        """Load from disk if files exist. Returns True on success."""
        if not self.persist_path:
            return False
        index_file = self.persist_path
        map_file = self.persist_path.with_suffix(".json")
        if not index_file.exists() or not map_file.exists():
            return False

        if self._backend == "usearch":
            self._index = _UsearchIndex(
                ndim=self.dim, metric="cos", dtype="f32",
                connectivity=32, expansion_add=200, expansion_search=128,
            )
            self._index.load(str(index_file))
        else:
            self._index = _hnswlib.Index(space=self.metric, dim=self.dim)
            self._index.load_index(str(index_file), max_elements=self.max_elements)
            self._index.set_ef(128)

        with map_file.open() as f:
            data = json.load(f)
        self._id_map  = {int(k): v for k, v in data["id_map"].items()}
        self._rev_map = {v: int(k) for k, v in self._id_map.items()}
        self._next_id = data["next_id"]
        self._initialized = True
        return True

    def save(self) -> None:
        if not self.persist_path or not self._initialized:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        if self._backend == "usearch":
            self._index.save(str(self.persist_path))
        else:
            self._index.save_index(str(self.persist_path))
        map_file = self.persist_path.with_suffix(".json")
        with map_file.open("w") as f:
            json.dump({"id_map": self._id_map, "next_id": self._next_id}, f)

    def _ensure_init(self) -> None:
        if not self._initialized:
            if not self.load():
                self.init_fresh()

    # ── Mutations ───────────────────────────────────────────────────────────

    def add(self, fragment_id: str, embedding: list[float]) -> None:
        self._ensure_init()
        # Remove old entry for this fragment if it exists
        if fragment_id in self._rev_map:
            self.remove(fragment_id)

        hid = self._next_id
        self._next_id += 1

        if self._backend == "usearch":
            self._index.add(hid, np.array(embedding, dtype=np.float32))
        else:
            self._index.add_items([embedding], [hid])

        self._id_map[hid]       = fragment_id
        self._rev_map[fragment_id] = hid

    def remove(self, fragment_id: str) -> None:
        self._ensure_init()
        hid = self._rev_map.pop(fragment_id, None)
        if hid is None:
            return
        del self._id_map[hid]
        try:
            if self._backend == "usearch":
                self._index.remove(hid)
            else:
                self._index.mark_deleted(hid)
        except Exception:
            pass  # index may not support soft-delete for this entry

    def get_vector(self, fragment_id: str) -> Optional[list[float]]:
        """Return the stored embedding for a fragment, or None if absent.

        Lets maintenance consumers (audit clustering, eviction) reuse the vector
        already in the index instead of re-embedding the content from scratch —
        which, across all facts on every audit, is the path that scales worst.
        """
        self._ensure_init()
        hid = self._rev_map.get(fragment_id)
        if hid is None:
            return None
        try:
            if self._backend == "usearch":
                vec = self._index.get(hid)
                if vec is None:
                    return None
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                return arr.tolist()
            else:
                items = self._index.get_items([hid])
                if not items:
                    return None
                return [float(x) for x in items[0]]
        except Exception:
            return None

    # ── Query ───────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        k: int = 20,
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> list[tuple[str, float]]:
        """
        Returns up to k (fragment_id, similarity) pairs, highest first.
        Over-fetches 3× when filter_fn is provided to maintain recall.
        """
        self._ensure_init()
        if self._next_id == 0:
            return []

        fetch_k = min(k * 3 if filter_fn else k, max(1, self._next_id))

        if self._backend == "usearch":
            matches = self._index.search(np.array(embedding, dtype=np.float32), fetch_k)
            pairs = list(zip(matches.keys.tolist(), matches.distances.tolist()))
            # usearch cosine distance is in [0,2]; convert to similarity [0,1]
            pairs = [(int(hid), max(0.0, 1.0 - float(dist) / 2.0))
                     for hid, dist in pairs]
        else:
            ids, distances = self._index.knn_query([embedding], k=fetch_k)
            # hnswlib cosine distance is 1 - similarity
            pairs = [(int(hid), 1.0 - float(dist))
                     for hid, dist in zip(ids[0], distances[0])]

        results: list[tuple[str, float]] = []
        for hid, similarity in pairs:
            frag_id = self._id_map.get(hid)
            if frag_id is None:
                continue
            if filter_fn and not filter_fn(frag_id):
                continue
            results.append((frag_id, similarity))
            if len(results) >= k:
                break

        return results

    # ── Stats ───────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._rev_map)

    @property
    def backend(self) -> str:
        return self._backend or "none"
