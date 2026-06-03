"""
=============================================================================
  MEMORY SYSTEM — END-TO-END SMOKE TEST SUITE
=============================================================================

  Run from the project root:
      python -m memory_system.test_smoke

  Or directly:
      python memory_system/test_smoke.py

  Requirements: NONE beyond what the memory system itself needs.
    - No API keys  (uses RandomEmbedder + mocked LLM)
    - No daemon    (tests components directly)
    - No network   (fully offline)

  The test creates a temporary SQLite DB and HNSW index in memory / tempdir,
  exercises every layer of the system, and prints a diagnostic report at the
  end.  If every test passes, the system is correctly wired.

  Exit code:  0 = all pass,  1 = failures detected
=============================================================================
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import shutil
import atexit

# ── Make sure we can import the package ──────────────────────────────────────
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Temp directory manager for Windows ──────────────────────────────────────
# On Windows, SQLite WAL-mode creates .db-wal and .db-shm sidecar files that
# may still be locked when TemporaryDirectory.__exit__ tries to delete them.
# We use a single persistent scratch dir and clean it up at process exit.

_SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="mem_smoke_"))

def _cleanup_scratch():
    try:
        shutil.rmtree(_SCRATCH_DIR, ignore_errors=True)
    except Exception:
        pass

atexit.register(_cleanup_scratch)

_scratch_counter = 0
def scratch_dir() -> Path:
    """Return a unique sub-directory inside the scratch area."""
    global _scratch_counter
    _scratch_counter += 1
    d = _SCRATCH_DIR / f"test_{_scratch_counter}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Test infrastructure ─────────────────────────────────────────────────────

class TestResult:
    def __init__(self, layer: str, name: str):
        self.layer = layer
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.details: list[str] = []
        self._start = time.perf_counter()
        self.elapsed_ms: float = 0

    def ok(self, detail: str = ""):
        self.passed = True
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if detail:
            self.details.append(detail)

    def fail(self, error: str):
        self.passed = False
        self.error = error
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    def note(self, detail: str):
        self.details.append(detail)


results: list[TestResult] = []


def _make_test(layer: str, name: str):
    """Decorator to register a test function."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            r = TestResult(layer, name)
            try:
                fn(r, *args, **kwargs)
                if not r.passed and r.error is None:
                    r.ok()
            except Exception as e:
                r.fail(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            results.append(r)
        wrapper._test = True
        wrapper._layer = layer
        wrapper._name = name
        return wrapper
    return decorator


# =============================================================================
#  LAYER 1: Configuration
# =============================================================================

@_make_test("L1-Config", "load_config returns defaults without YAML file")
def test_config_defaults(r: TestResult):
    from memory_system.config import load_config
    cfg = load_config(path="/nonexistent/path/memory.yaml")
    assert cfg.compression.distillation_model == "claude-haiku-4-5-20251001", \
        f"Unexpected model: {cfg.compression.distillation_model}"
    assert cfg.retrieval.default_token_budget == 4000
    assert cfg.eviction.recency_decay_lambda == 0.003
    r.note(f"Compression model: {cfg.compression.distillation_model}")
    r.note(f"Token budget: {cfg.retrieval.default_token_budget}")
    r.ok("All default config values correct")


@_make_test("L1-Config", "MemoryConfig dataclass hierarchy is complete")
def test_config_structure(r: TestResult):
    from memory_system.config import MemoryConfig
    cfg = MemoryConfig()
    required_sections = [
        "daemon", "storage", "embedding", "retrieval",
        "compression", "eviction", "cross_project", "self_improvement",
    ]
    for section in required_sections:
        assert hasattr(cfg, section), f"Missing config section: {section}"
    r.ok(f"All {len(required_sections)} config sections present")


# =============================================================================
#  LAYER 2: Database (SQLite schema + CRUD)
# =============================================================================

@_make_test("L2-Database", "Schema creation and fragment CRUD")
def test_db_schema_and_crud(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.ids import new_id

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.connect()

        now = datetime.now(tz=timezone.utc).isoformat()
        frag = MemoryFragment(
            id=new_id(),
            project_id="test-project",
            scope="project",
            category="fact",
            content="Docker builds require a .dockerignore file.",
            token_count=12,
            confidence=0.9,
            source_type="distillation",
            created_at=now,
            last_accessed=now,
            embedding_model="random",
            embedding_dim=128,
        )
        db.upsert_fragment(frag)
        r.note(f"Inserted fragment {frag.id[:12]}...")

        # Read back
        loaded = db.get_fragment(frag.id)
        assert loaded is not None, "Fragment not found after insert"
        assert loaded.content == frag.content, "Content mismatch"
        assert loaded.confidence == 0.9, f"Confidence mismatch: {loaded.confidence}"
        r.note("Read-back content matches")

        # List
        frags = db.list_fragments("test-project")
        assert len(frags) == 1, f"Expected 1 fragment, got {len(frags)}"

        # Touch (access count)
        db.touch_fragment(frag.id, now)
        updated = db.get_fragment(frag.id)
        assert updated.access_count == 1, f"Access count should be 1, got {updated.access_count}"

        # Deprecate
        db.mark_deprecated(frag.id, deprecated_by=None)
        deprecated = db.get_fragment(frag.id)
        assert deprecated.is_deprecated, "Fragment should be deprecated"

        # List excludes deprecated by default
        active = db.list_fragments("test-project")
        assert len(active) == 0, "Deprecated fragment should be excluded from list"

        db.close()
        r.ok("All CRUD operations verified")


@_make_test("L2-Database", "BM25 full-text search")
def test_bm25_search(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.ids import new_id

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_fts.db")
        db.connect()

        now = datetime.now(tz=timezone.utc).isoformat()
        contents = [
            "Docker builds fail when .dockerignore is missing",
            "Python virtual environments should use venv not virtualenv",
            "The database migration script runs on deploy",
        ]
        for c in contents:
            frag = MemoryFragment(
                id=new_id(), project_id="p1", scope="project",
                category="fact", content=c, token_count=len(c) // 4,
                created_at=now, last_accessed=now,
                embedding_model="random", embedding_dim=128,
            )
            db.upsert_fragment(frag)

        hits = db.bm25_search("docker", "p1", limit=5)
        assert len(hits) >= 1, f"Expected at least 1 BM25 hit for 'docker', got {len(hits)}"
        r.note(f"BM25 'docker' returned {len(hits)} hit(s)")

        hits2 = db.bm25_search("python venv", "p1", limit=5)
        assert len(hits2) >= 1, f"Expected at least 1 BM25 hit for 'python venv', got {len(hits2)}"
        r.note(f"BM25 'python venv' returned {len(hits2)} hit(s)")

        db.close()
        r.ok("BM25 full-text search functional")


@_make_test("L2-Database", "Entity + Triple CRUD and graph traversal")
def test_entity_graph(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import Entity, Triple
    from memory_system.ids import new_id

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_graph.db")
        db.connect()

        now = datetime.now(tz=timezone.utc).isoformat()
        e1 = Entity(id=new_id(), project_id="p1", entity_type="file",
                     qualified_name="src/main.py", last_seen=now)
        e2 = Entity(id=new_id(), project_id="p1", entity_type="function",
                     qualified_name="src/main.py::main", last_seen=now)
        e3 = Entity(id=new_id(), project_id="p1", entity_type="class",
                     qualified_name="src/services/UserService.ts", last_seen=now)

        db.upsert_entity(e1)
        db.upsert_entity(e2)
        db.upsert_entity(e3)
        r.note(f"Created 3 entities")

        t1 = Triple(id=new_id(), subject_id=e2.id, predicate="depends_on",
                     object_id=e1.id, project_id="p1", created_at=now)
        t2 = Triple(id=new_id(), subject_id=e3.id, predicate="calls",
                     object_id=e2.id, project_id="p1", created_at=now)
        db.upsert_triple(t1)
        db.upsert_triple(t2)
        r.note(f"Created 2 triples")

        # Graph traversal
        neighbors = db.get_neighbors(e1.id, max_hops=2)
        assert len(neighbors) >= 1, f"Expected at least 1 neighbor, got {len(neighbors)}"
        r.note(f"2-hop neighbors from e1: {len(neighbors)} found")

        # Fuzzy entity match
        matched = db.fuzzy_match_entities("p1", "main")
        assert len(matched) >= 1, f"Expected at least 1 entity matching 'main'"

        db.close()
        r.ok("Entity/Triple CRUD and graph traversal working")


# =============================================================================
#  LAYER 3: Embedder + Vector Index
# =============================================================================

@_make_test("L3-Vector", "RandomEmbedder produces deterministic unit vectors")
def test_random_embedder(r: TestResult):
    from memory_system.embedder import RandomEmbedder

    emb = RandomEmbedder(dim=128)
    v1 = emb.embed("hello world")
    v2 = emb.embed("hello world")
    v3 = emb.embed("something different")

    assert v1 == v2, "Same text should produce identical embeddings"
    assert v1 != v3, "Different text should produce different embeddings"

    # Verify unit norm
    norm = math.sqrt(sum(x*x for x in v1))
    assert abs(norm - 1.0) < 1e-6, f"Embedding norm should be 1.0, got {norm}"

    r.note(f"dim={emb.dim}, model_name='{emb.model_name}'")
    r.ok("Deterministic unit vectors confirmed")


@_make_test("L3-Vector", "CachedEmbedder caches and delegates correctly")
def test_cached_embedder(r: TestResult):
    from memory_system.embedder import RandomEmbedder, CachedEmbedder

    inner = RandomEmbedder(dim=64)
    cached = CachedEmbedder(inner, max_entries=5)

    v1 = cached.embed("test text")
    v2 = cached.embed("test text")  # should be cache hit
    assert v1 == v2, "Cache should return same vector"

    batch = cached.embed_batch(["a", "b", "test text", "c"])
    assert len(batch) == 4, f"Batch should return 4 vectors, got {len(batch)}"
    assert batch[2] == v1, "Cached value should match for 'test text' in batch"

    r.note(f"Cache model_name='{cached.model_name}', dim={cached.dim}")
    r.ok("CachedEmbedder delegation and caching verified")


@_make_test("L3-Vector", "VectorIndex add/query/remove/persist")
def test_vector_index(r: TestResult):
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder

    emb = RandomEmbedder(dim=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = Path(tmpdir) / "test.hnsw"
        idx = VectorIndex(dim=128, persist_path=str(idx_path))
        idx.init_fresh()

        # Add vectors
        ids = [f"frag_{i}" for i in range(10)]
        for fid in ids:
            vec = emb.embed(f"content for {fid}")
            idx.add(fid, vec)

        assert idx.size == 10, f"Index size should be 10, got {idx.size}"
        r.note(f"Backend: {idx.backend}, size: {idx.size}")

        # Query
        q = emb.embed("content for frag_3")
        results = idx.query(q, k=3)
        assert len(results) >= 1, "Query should return at least 1 result"
        assert results[0][0] == "frag_3", f"Top result should be frag_3, got {results[0][0]}"
        r.note(f"Top result for 'frag_3' query: {results[0][0]} (sim={results[0][1]:.4f})")

        # Query with filter
        filtered = idx.query(q, k=3, filter_fn=lambda fid: fid != "frag_3")
        if filtered:
            assert filtered[0][0] != "frag_3", "Filter should exclude frag_3"

        # Remove
        idx.remove("frag_3")
        assert idx.size == 9, f"After remove, size should be 9, got {idx.size}"

        # Persist and reload
        idx.save()
        assert idx_path.exists(), f"Index file should exist at {idx_path}"

        idx2 = VectorIndex(dim=128, persist_path=str(idx_path))
        loaded = idx2.load()
        assert loaded, "Load should return True"
        assert idx2.size == 9, f"Reloaded index size should be 9, got {idx2.size}"
        r.note("Persistence: save + reload confirmed")

        r.ok("VectorIndex fully functional")


# =============================================================================
#  LAYER 4: CRS Scoring
# =============================================================================

@_make_test("L4-Scoring", "CRS computation and tier classification")
def test_crs_and_tiers(r: TestResult):
    from memory_system.scoring import composite_relevance_score, tier
    from memory_system.models import MemoryFragment
    from memory_system.embedder import RandomEmbedder

    emb = RandomEmbedder(dim=128)
    now = datetime.now(tz=timezone.utc).isoformat()

    frag = MemoryFragment(
        id="test-crs", project_id="p1", scope="project",
        category="fact", content="test content", token_count=5,
        confidence=0.9, created_at=now, last_accessed=now,
        access_count=10, graph_centrality=0.5,
        embedding_model="random", embedding_dim=128,
        embedding=emb.embed("test content"),
    )

    # With query embedding (semantic signal active)
    q_emb = emb.embed("test content")
    crs_with_query = composite_relevance_score(frag, q_emb)
    assert 0.0 <= crs_with_query <= 1.0, f"CRS out of range: {crs_with_query}"
    r.note(f"CRS (with matching query): {crs_with_query:.4f}")

    # Without query embedding (eviction mode)
    crs_no_query = composite_relevance_score(frag)
    assert 0.0 <= crs_no_query <= 1.0, f"CRS out of range: {crs_no_query}"
    r.note(f"CRS (eviction mode): {crs_no_query:.4f}")

    # Pinned fragment should always be 1.0
    frag.is_pinned = True
    crs_pinned = composite_relevance_score(frag, q_emb)
    assert crs_pinned == 1.0, f"Pinned CRS should be 1.0, got {crs_pinned}"
    r.note(f"CRS (pinned): {crs_pinned}")

    # Tier classification
    assert tier(0.70) == "hot"
    assert tier(0.40) == "warm"
    assert tier(0.10) == "cold"
    r.note("Tier classification: hot/warm/cold boundaries correct")

    r.ok("CRS scoring and tier classification verified")


# =============================================================================
#  LAYER 5: Retrieval (Fused 3-signal RRF)
# =============================================================================

@_make_test("L5-Retrieval", "Fused retrieval returns token-budget-packed fragments")
def test_fused_retrieval(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.retrieval import fused_retrieval
    from memory_system.config import RetrievalConfig
    from memory_system.ids import new_id

    emb = RandomEmbedder(dim=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_ret.db")
        db.connect()
        idx = VectorIndex(dim=128, persist_path=str(Path(tmpdir) / "test.hnsw"))
        idx.init_fresh()

        now = datetime.now(tz=timezone.utc).isoformat()
        contents = [
            "Docker builds need a .dockerignore file to exclude node_modules",
            "The CI pipeline runs pytest with coverage on every push",
            "Use multi-stage Docker builds for smaller production images",
            "Python 3.12 is the minimum supported version",
            "The payment service depends on Stripe SDK version 8.x",
        ]

        for content in contents:
            fid = new_id()
            vec = emb.embed(content)
            frag = MemoryFragment(
                id=fid, project_id="p1", scope="project",
                category="fact", content=content, token_count=len(content) // 4,
                confidence=0.8, created_at=now, last_accessed=now,
                embedding_model="random", embedding_dim=128, embedding=vec,
            )
            db.upsert_fragment(frag)
            idx.add(fid, vec)

        cfg = RetrievalConfig(default_token_budget=200, max_fragments=5, min_crs=0.0)
        q_emb = emb.embed("docker build problems")

        result = fused_retrieval(
            db=db, vector_index=idx,
            query_embedding=q_emb, query_text="docker build problems",
            project_id="p1", token_budget=200, cfg=cfg,
        )

        assert len(result.fragments) >= 1, f"Expected at least 1 fragment, got {len(result.fragments)}"
        assert result.token_budget_used <= 200, f"Token budget exceeded: {result.token_budget_used}"
        r.note(f"Retrieved {len(result.fragments)} fragments, {result.token_budget_used} tokens used")

        # Verify CRS is set on each fragment
        for frag in result.fragments:
            assert frag.crs > 0, f"Fragment {frag.id[:8]} has CRS=0 (should be positive)"

        db.close()
        r.ok("Fused retrieval returns ranked, budget-packed results")


@_make_test("L5-Retrieval", "CRS semantic signal survives the DB load path (round-trip)")
def test_crs_semantic_roundtrip(r: TestResult):
    # Regression guard for the dead-semantic bug: fragments loaded from the DB
    # carry NO embedding (vectors live in the HNSW index, not a column), so the
    # 0.30 semantic CRS weight must be fed by the vector-lane similarity — not
    # fragment.embedding. Insert → close → reopen from disk → a query matching
    # one fragment must clearly out-score an unrelated one. Before the fix both
    # semantic components collapsed to the 0.5 neutral fallback and tied.
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.retrieval import fused_retrieval
    from memory_system.config import RetrievalConfig
    from memory_system.ids import new_id

    emb = RandomEmbedder(dim=128)
    # scratch_dir() (not TemporaryDirectory) — WAL leaves .db-wal/.db-shm
    # sidecars that lock the file on Windows; scratch_dir isn't deleted mid-suite.
    tmpdir = scratch_dir()
    db_path = tmpdir / "rt_sem.db"
    idx_path = tmpdir / "rt_sem.hnsw"
    now = datetime.now(tz=timezone.utc).isoformat()

    # Two fragments with IDENTICAL non-semantic signals (recency, frequency,
    # confidence, importance) so the ONLY thing that can separate their CRS
    # is the semantic component. RandomEmbedder hashes text → identical text
    # gives cosine ≈ 1.0, unrelated text ≈ 0.
    target_text   = "alpha beta gamma delta the load bearing fact"
    distract_text = "completely unrelated zeta eta theta iota kappa"
    target_id, distract_id = new_id(), new_id()

    db = Database(db_path); db.connect()
    idx = VectorIndex(dim=128, persist_path=str(idx_path)); idx.init_fresh()
    for fid, text in [(target_id, target_text), (distract_id, distract_text)]:
        vec = emb.embed(text)
        frag = MemoryFragment(
            id=fid, project_id="p1", scope="project", category="fact",
            content=text, token_count=8, confidence=0.8,
            created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, embedding=vec,
        )
        db.upsert_fragment(frag)
        idx.add(fid, vec)
    idx.save()
    db.close()

    # ── Round-trip: reopen from disk; loaded fragments have no embedding ──
    db2 = Database(db_path); db2.connect()
    reloaded = db2.get_fragment(target_id)
    assert not reloaded.embedding, (
        "Precondition broken: DB load path now carries an embedding — this "
        "test no longer exercises the seam it was written to guard.")
    idx2 = VectorIndex(dim=128, persist_path=str(idx_path))
    assert idx2.load(), "Vector index should reload from disk"

    cfg = RetrievalConfig(default_token_budget=500, max_fragments=5, min_crs=0.0)
    q_emb = emb.embed(target_text)   # exact match → cosine ≈ 1.0 for target
    result = fused_retrieval(
        db=db2, vector_index=idx2,
        query_embedding=q_emb, query_text=target_text,
        project_id="p1", token_budget=500, cfg=cfg,
    )
    by_id = {f.id: f for f in result.fragments}
    assert target_id in by_id and distract_id in by_id, \
        f"Both fragments should be retrieved, got {list(by_id.keys())}"
    t_crs = float(by_id[target_id].crs)
    d_crs = float(by_id[distract_id].crs)
    r.note(f"target CRS={t_crs:.3f} vs distractor CRS={d_crs:.3f} (Δ={t_crs - d_crs:.3f})")
    # 0.30 semantic weight on ~1.0 vs ~0.0 cosine should open a clear margin.
    # Without the fix the gap would be ~0 (both semantic = 0.5).
    assert t_crs - d_crs > 0.05, (
        f"Semantic signal is inert on the load path: target {t_crs:.3f} did "
        f"not clearly out-score distractor {d_crs:.3f} (Δ={t_crs - d_crs:.3f})")
    db2.close()
    r.note("Embedding-free reloaded fragment still scored real semantics")
    r.ok("CRS semantic signal is live after DB reload (round-trip)")


@_make_test("L5-Retrieval", "Cross-project semantic gate filters irrelevant globals (round-trip)")
def test_semantic_gate_roundtrip(r: TestResult):
    # Regression guard: the SemanticGate stops an abstract, transferable
    # __global__ fragment from leaking into a project unless it's relevant to
    # the query. That relevance check reads frag.embedding — which is None after
    # the DB load path — so it was silently skipped, letting ANY transferable
    # global through. The fix feeds the gate the vector-lane similarity. Here a
    # relevant global passes and an irrelevant one (identical except content) is
    # gated out; nothing else (CRS floor, budget) can explain its absence.
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.retrieval import fused_retrieval
    from memory_system.config import RetrievalConfig, CrossProjectConfig
    from memory_system.ids import new_id

    emb = RandomEmbedder(dim=128)
    # scratch_dir() (not TemporaryDirectory) — WAL sidecars lock the file on
    # Windows; scratch_dir isn't deleted mid-suite.
    tmpdir = scratch_dir()
    db_path = tmpdir / "gate.db"
    idx_path = tmpdir / "gate.hnsw"
    now = datetime.now(tz=timezone.utc).isoformat()

    relevant_text   = "always use dependency injection for testable services"
    irrelevant_text = "prefer tabs over spaces when editing makefiles"
    rel_id, irr_id = new_id(), new_id()

    db = Database(db_path); db.connect()
    idx = VectorIndex(dim=128, persist_path=str(idx_path)); idx.init_fresh()
    # Both __global__ and abstract (>=0.6); 'preference' is a DB-valid category
    # we mark transferable below — so the ONLY discriminator left is the
    # similarity gate (threshold 0.70).
    for fid, text in [(rel_id, relevant_text), (irr_id, irrelevant_text)]:
        vec = emb.embed(text)
        frag = MemoryFragment(
            id=fid, project_id="__global__", scope="global",
            category="preference", content=text, token_count=8,
            abstraction_lvl=0.8, confidence=0.8,
            created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, embedding=vec,
        )
        db.upsert_fragment(frag)
        idx.add(fid, vec)
    idx.save()
    db.close()

    # ── Round-trip: reopen; loaded global fragments carry no embedding ──
    db2 = Database(db_path); db2.connect()
    assert not db2.get_fragment(irr_id).embedding, \
        "Precondition broken: DB load path now carries an embedding."
    idx2 = VectorIndex(dim=128, persist_path=str(idx_path))
    assert idx2.load(), "Vector index should reload from disk"

    # Generous budget + zero CRS floor so the gate is the only filter.
    cfg = RetrievalConfig(default_token_budget=2000, max_fragments=50, min_crs=0.0)
    xcfg = CrossProjectConfig(
        abstraction_threshold=0.6, similarity_threshold=0.70,
        transferable_categories=["preference"],
    )
    q_emb = emb.embed(relevant_text)   # exact match to the relevant global
    result = fused_retrieval(
        db=db2, vector_index=idx2, query_embedding=q_emb,
        query_text=relevant_text, project_id="p1",
        token_budget=2000, cfg=cfg, include_global=True,
        cross_project_cfg=xcfg,
    )
    ids = {f.id for f in result.fragments}
    assert rel_id in ids, "Relevant global memory should pass the gate"
    assert irr_id not in ids, (
        "Irrelevant global memory leaked past the semantic gate — its "
        "similarity check is inert on the DB load path.")
    db2.close()
    r.note(f"relevant global passed; irrelevant global gated out ({len(ids)} returned)")
    r.ok("Cross-project semantic gate is live after DB reload (round-trip)")


@_make_test("L1-Schema", "Shared DB connection is safe under concurrent threads")
def test_db_concurrency(r: TestResult):
    # Regression guard for the daemon's single-connection race: the asyncio
    # executor pool AND the dashboard's HTTP threads all drive ONE sqlite
    # connection. Concurrent BEGIN/COMMIT + reads on it raise
    # "cannot start a transaction within a transaction" / "recursive use of
    # cursors". The Database lock must serialize all access. This test hammers
    # the connection from many threads at once and asserts zero errors.
    import threading
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.ids import new_id

    now = datetime.now(tz=timezone.utc).isoformat()

    def _mk_frag(fid: str, content: str) -> MemoryFragment:
        return MemoryFragment(
            id=fid, project_id="p1", scope="project", category="fact",
            content=content, token_count=4, confidence=0.8,
            created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128,
        )

    db = Database(scratch_dir() / "concurrency.db")
    db.connect()
    # Seed a few rows so readers always have something to scan mid-write.
    seed_ids = [new_id() for _ in range(5)]
    for sid in seed_ids:
        db.upsert_fragment(_mk_frag(sid, f"seed {sid}"))

    N_WRITERS, N_READERS, ITERS = 4, 4, 60
    errors: list[str] = []
    errors_lock = threading.Lock()
    start = threading.Barrier(N_WRITERS + N_READERS)

    def _record(e: Exception):
        with errors_lock:
            errors.append(f"{type(e).__name__}: {e}")

    def _writer(wid: int):
        start.wait()  # release all threads at once for maximum contention
        try:
            for i in range(ITERS):
                fid = f"w{wid}-{i}-{new_id()}"
                db.upsert_fragment(_mk_frag(fid, f"writer {wid} item {i}"))
                # also exercise a bare write path (no explicit transaction)
                db.execute(
                    "UPDATE memory_fragments SET access_count=access_count+1 WHERE id=?",
                    (fid,),
                )
        except Exception as e:  # noqa: BLE001 — we want to surface ANY raise
            _record(e)

    def _reader():
        start.wait()
        try:
            for _ in range(ITERS):
                db.fetchall("SELECT * FROM memory_fragments ORDER BY created_at DESC")
                db.list_fragment_ids(["p1"])
                db.get_fragment(seed_ids[0])
        except Exception as e:  # noqa: BLE001
            _record(e)

    threads = [threading.Thread(target=_writer, args=(w,)) for w in range(N_WRITERS)]
    threads += [threading.Thread(target=_reader) for _ in range(N_READERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, (
        f"Concurrent DB access raised {len(errors)} error(s); the shared "
        f"connection is not serialized. First: {errors[0]}")

    expected = len(seed_ids) + N_WRITERS * ITERS
    row = db.fetchone("SELECT COUNT(*) AS n FROM memory_fragments")
    db.close()
    assert row["n"] == expected, f"Expected {expected} rows, got {row['n']}"
    r.note(f"{N_WRITERS} writers + {N_READERS} readers × {ITERS} iters, 0 errors")
    r.note(f"Final row count {row['n']} == expected {expected}")
    r.ok("Shared connection is thread-safe under concurrent load")


@_make_test("L5-Retrieval", "Prompt assembly produces L0/L1/L2/L3 structure")
def test_prompt_assembly(r: TestResult):
    from memory_system.retrieval import assemble_prompt
    from memory_system.models import MemoryFragment, RetrievalResult

    now = datetime.now(tz=timezone.utc).isoformat()
    frags = [
        MemoryFragment(
            id="f1", project_id="p1", scope="project", category="fact",
            content="Docker uses multi-stage builds", token_count=8,
            confidence=0.9, created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, crs=0.85,
        ),
    ]
    result = RetrievalResult(
        fragments=frags,
        project_summary="A Python web service with Docker deployment.",
        global_profile_hash=None,
        token_budget_used=8,
    )

    prompt = assemble_prompt(
        system_prompt="You are a helpful assistant.",
        global_profile="User prefers concise answers.",
        result=result,
        user_query="How do I fix the Docker build?",
        file_contents="# Dockerfile\nFROM python:3.12",
    )

    assert "<global_profile>" in prompt, "Missing L0 global_profile tag"
    assert "<project_memory>" in prompt, "Missing L1 project_memory tag"
    assert "<recalled_memories>" in prompt, "Missing L2 recalled_memories tag"
    assert "<active_files>" in prompt, "Missing L3 active_files tag"
    assert "<user_message>" in prompt, "Missing L3 user_message tag"
    assert "Docker uses multi-stage builds" in prompt, "Fragment content missing"

    r.note(f"Prompt length: {len(prompt)} chars")
    r.note("All L0/L1/L2/L3 sections present")
    r.ok("Prompt assembly structure verified")


# =============================================================================
#  LAYER 6: Compression Pipeline (with mocked LLM)
# =============================================================================

@_make_test("L6-Pipeline", "4-stage pipeline ingests transcript and produces fragments")
def test_pipeline_with_mock_llm(r: TestResult):
    from memory_system.schema import Database
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.config import CompressionConfig
    from memory_system.pipeline import ConsolidationPipeline, TranscriptMessage
    from memory_system.ids import new_id
    import memory_system.llm as llm_module

    # ── Mock the LLM layer ──────────────────────────────────────────────────
    # This replaces call_model_json with a deterministic fake that returns
    # structured data matching what the pipeline expects.
    _original_call_model_json = llm_module.call_model_json
    _original_call_model = llm_module.call_model

    call_count = {"distill": 0, "entity": 0}

    def mock_call_model_json(prompt, **kwargs):
        if "Extract structured information" in prompt:
            call_count["distill"] += 1
            return {
                "intent": "Fix the Docker build failure",
                "outcome": "Created .dockerignore file",
                "key_decisions": ["Exclude node_modules", "Exclude .git"],
                "tools_used": ["read_file", "write_file"],
                "files_modified": [".dockerignore"],
                "errors_encountered": [],
                "summary": "Fixed Docker build by creating .dockerignore to exclude large directories.",
                "confidence": 0.85,
            }
        elif "Extract all code entities" in prompt:
            call_count["entity"] += 1
            return {
                "entities": [
                    {"type": "file", "name": ".dockerignore"},
                    {"type": "file", "name": "Dockerfile"},
                    {"type": "config", "name": "docker-compose.yml"},
                ],
                "relations": [
                    {"subject": "Dockerfile", "predicate": "depends_on", "object": ".dockerignore"},
                ],
            }
        elif "These" in prompt and "episodes describe similar" in prompt:
            return {
                "fact": "Docker builds should always include a .dockerignore file.",
                "confidence": 0.9,
            }
        else:
            return {"summary": "mock response", "confidence": 0.7}

    def mock_call_model(prompt, **kwargs):
        return json.dumps(mock_call_model_json(prompt, **kwargs))

    llm_module.call_model_json = mock_call_model_json
    llm_module.call_model = mock_call_model

    try:
        tmpdir = scratch_dir()
        db = Database(tmpdir / "test_pipe.db")
        db.connect()
        idx = VectorIndex(dim=128, persist_path=str(tmpdir / "pipe.hnsw"))
        idx.init_fresh()

        emb = RandomEmbedder(dim=128)
        cfg = CompressionConfig(consolidation_threshold=3, max_episode_tokens=500)

        # Create the session record that fragments will reference
        from memory_system.models import Session
        db.upsert_session(Session(id="sess_test_001", project_id="p1"))

        pipeline = ConsolidationPipeline(db, idx, cfg, embedder=emb)

        # Build a realistic transcript
        messages = [
            TranscriptMessage(role="user", content="Fix the Docker build, it's failing with context too large error"),
            TranscriptMessage(role="assistant", content="Let me check your Dockerfile first.", tool_call_chain_len=1),
            TranscriptMessage(role="tool", content="FROM python:3.12-slim\nCOPY . .\nRUN pip install -r requirements.txt"),
            TranscriptMessage(role="assistant", content="You're missing a .dockerignore. Let me create one."),
            TranscriptMessage(role="tool", content="Created .dockerignore with: node_modules, .git, dist, .env"),
            TranscriptMessage(role="user", content="Great, that fixed it! Remember this for next time.",
                              is_explicit_remember=True),
        ]

        new_frags = pipeline.ingest(messages, session_id="sess_test_001", project_id="p1")

        assert len(new_frags) >= 1, f"Pipeline should produce at least 1 fragment, got {len(new_frags)}"
        r.note(f"Pipeline produced {len(new_frags)} fragment(s)")
        r.note(f"LLM calls: distill={call_count['distill']}, entity={call_count['entity']}")

        # Verify fragments are in DB (some may be deprecated by Stage 4 dedup)
        db_frags = db.list_fragments("p1", include_deprecated=True)
        assert len(db_frags) >= 1, f"Fragments should be persisted to DB, found {len(db_frags)}"
        r.note(f"DB has {len(db_frags)} total fragment(s) ({len(db.list_fragments('p1'))} active)")

        # Verify fragments are in vector index
        assert idx.size >= 1, f"Vector index should have entries, has {idx.size}"
        r.note(f"Vector index has {idx.size} entries")

        # Verify entities were extracted
        entities = db.fuzzy_match_entities("p1", "docker", limit=10)
        r.note(f"Entities matching 'docker': {len(entities)}")

        db.close()

    finally:
        # Restore original LLM functions
        llm_module.call_model_json = _original_call_model_json
        llm_module.call_model = _original_call_model

    r.ok("Full 4-stage pipeline ingestion verified")


@_make_test("L6-Pipeline", "Reflection produces a high-confidence fragment")
def test_pipeline_reflection(r: TestResult):
    from memory_system.schema import Database
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.config import CompressionConfig
    from memory_system.pipeline import ConsolidationPipeline, TranscriptMessage
    import memory_system.llm as llm_module
    import memory_system.pipeline as pipeline_module

    _original_llm = llm_module.call_model_json
    _original_pipe = pipeline_module.call_model_json

    def mock_reflect(prompt, **kwargs):
        return {
            "new_facts": [
                "Docker builds need .dockerignore to avoid sending node_modules",
            ],
            "corrected_assumptions": [
                {"was": "COPY . . copies only source files", "now": "COPY . . copies everything in build context"}
            ],
            "lessons": ["Always check for .dockerignore first when build context is large"],
            "proactive_suggestions": [],
        }

    llm_module.call_model_json = mock_reflect
    pipeline_module.call_model_json = mock_reflect

    try:
        tmpdir = scratch_dir()
        db = Database(tmpdir / "test_refl.db")
        db.connect()
        idx = VectorIndex(dim=128, persist_path=str(tmpdir / "refl.hnsw"))
        idx.init_fresh()

        emb = RandomEmbedder(dim=128)
        cfg = CompressionConfig()
        pipeline = ConsolidationPipeline(db, idx, cfg, embedder=emb)

        # Create the session record that the reflection fragment will reference
        from memory_system.models import Session
        db.upsert_session(Session(id="sess_refl", project_id="p1"))

        messages = [
            TranscriptMessage(role="user", content="Fix Docker build"),
            TranscriptMessage(role="assistant", content="Created .dockerignore"),
        ]

        frag = pipeline.reflect(messages, session_id="sess_refl", project_id="p1")
        if frag is None:
            # reflect() may fail if the mock didn't intercept correctly.
            # Check if it's because the mock returns data but reflect()'s 
            # JSON parsing found no items (all empty lists).
            r.note("Reflection returned None - checking if mock data was processed")
            # Call mock directly to verify it returns data
            test_data = mock_reflect("test")
            items = test_data.get("new_facts", []) + test_data.get("lessons", [])
            if items:
                r.fail("Mock returns data but reflect() returned None - possible LLM call routing issue")
                return
            else:
                r.fail("Mock returns empty data")
                return
        assert frag.confidence == 0.85, f"Reflection confidence should be 0.85, got {frag.confidence}"
        assert frag.source_type == "reflection"
        assert "Session reflection" in frag.content
        r.note(f"Reflection fragment: {frag.content[:80]}...")

        db.close()
    finally:
        llm_module.call_model_json = _original_llm
        pipeline_module.call_model_json = _original_pipe

    r.ok("Reflection produces valid high-confidence fragment")


# =============================================================================
#  LAYER 7: Auditor (self-improvement loop)
# =============================================================================

@_make_test("L7-Auditor", "Confidence decay on old unvalidated facts")
def test_auditor_decay(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.config import SelfImprovementConfig
    from memory_system.auditor import MemoryAuditor
    from memory_system.ids import new_id

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_audit.db")
        db.connect()

        # Create a fact that was last accessed 120 days ago
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=120)).isoformat()
        frag = MemoryFragment(
            id=new_id(), project_id="p1", scope="project",
            category="fact", content="Old fact that hasn't been accessed",
            token_count=10, confidence=0.8,
            created_at=old_date, last_accessed=old_date,
            embedding_model="random", embedding_dim=128,
        )
        db.upsert_fragment(frag)

        cfg = SelfImprovementConfig(
            contradiction_detection=False,  # skip LLM calls
            confidence_decay_factor=0.85,
            confidence_decay_after_days=90,
        )
        auditor = MemoryAuditor(db, cfg)
        report = auditor.audit(project_id="p1")

        assert report.fragments_decayed >= 1, f"Should decay at least 1 fragment, decayed {report.fragments_decayed}"
        r.note(f"Decayed {report.fragments_decayed} fragment(s)")

        updated = db.get_fragment(frag.id)
        # Velocity-weighted decay: c * exp(-(λ₀ + κ·v) * Δt_days)
        # With 120 days old, v=0, λ₀=0.003: expected ≈ 0.8 * exp(-0.36) ≈ 0.558
        assert updated.confidence < 0.8, f"Confidence should have decayed below 0.8, got {updated.confidence}"
        assert updated.confidence >= 0.05, f"Confidence should not drop below floor 0.05, got {updated.confidence}"
        r.note(f"Confidence: 0.8 → {updated.confidence:.4f} (velocity-weighted exponential decay)")

        db.close()
        r.ok("Confidence decay verified")


@_make_test("L7-Auditor", "PageRank centrality propagation to fragments")
def test_auditor_pagerank(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment, Entity, Triple
    from memory_system.config import SelfImprovementConfig
    from memory_system.auditor import MemoryAuditor
    from memory_system.ids import new_id

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_pr.db")
        db.connect()

        now = datetime.now(tz=timezone.utc).isoformat()

        # Create entities
        e1 = Entity(id=new_id(), project_id="p1", entity_type="file",
                     qualified_name="app.py", last_seen=now)
        e2 = Entity(id=new_id(), project_id="p1", entity_type="function",
                     qualified_name="main", last_seen=now)
        e3 = Entity(id=new_id(), project_id="p1", entity_type="class",
                     qualified_name="UserService", last_seen=now)
        db.upsert_entity(e1)
        db.upsert_entity(e2)
        db.upsert_entity(e3)

        # Create a fragment linked via triples
        frag = MemoryFragment(
            id=new_id(), project_id="p1", scope="project",
            category="episode", content="Worked on app.py main function",
            token_count=8, created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128,
        )
        db.upsert_fragment(frag)

        # Create triples linking entities and the fragment
        db.upsert_triple(Triple(
            id=new_id(), subject_id=e2.id, predicate="depends_on",
            object_id=e1.id, project_id="p1", source_fragment=frag.id,
            created_at=now,
        ))
        db.upsert_triple(Triple(
            id=new_id(), subject_id=e3.id, predicate="calls",
            object_id=e2.id, project_id="p1", source_fragment=frag.id,
            created_at=now,
        ))

        cfg = SelfImprovementConfig(contradiction_detection=False)
        auditor = MemoryAuditor(db, cfg)
        report = auditor.audit(project_id="p1")

        assert report.centrality_updated >= 1, f"Should update at least 1 centrality, got {report.centrality_updated}"
        r.note(f"Centrality updated for {report.centrality_updated} fragment(s)")

        updated_frag = db.get_fragment(frag.id)
        assert updated_frag.graph_centrality > 0, f"Fragment centrality should be > 0, got {updated_frag.graph_centrality}"
        r.note(f"Fragment centrality: 0.0 → {updated_frag.graph_centrality:.4f}")

        db.close()
        r.ok("PageRank centrality propagation verified")


# =============================================================================
#  LAYER 8: Project + Prompt modules (agent runner prerequisites)
# =============================================================================

@_make_test("L8-Agent", "Project ID resolution and session ID generation")
def test_project_resolution(r: TestResult):
    from memory_system.project import resolve_project_id, new_session_id

    pid = resolve_project_id(Path.cwd())
    assert pid.startswith("proj_"), f"Project ID should start with 'proj_', got '{pid}'"
    assert len(pid) > 6, f"Project ID too short: '{pid}'"
    r.note(f"Resolved CWD to project_id: {pid}")

    # Same CWD should produce same project_id (deterministic)
    pid2 = resolve_project_id(Path.cwd())
    assert pid == pid2, "Same CWD should produce same project_id"

    sid = new_session_id()
    assert sid.startswith("sess_"), f"Session ID should start with 'sess_', got '{sid}'"
    r.note(f"Generated session_id: {sid}")

    # Two session IDs should be different
    sid2 = new_session_id()
    assert sid != sid2, "Sequential session IDs should be unique"

    r.ok("Project/session ID generation verified")


@_make_test("L8-Agent", "Prompt module assembles L0-L3 layers correctly")
def test_prompt_module(r: TestResult):
    from memory_system.prompt import assemble, SYSTEM_PROMPT, format_fragments

    fragments = [
        {
            "id": "frag_001",
            "scope": "project",
            "content": "Docker needs .dockerignore",
            "crs": 0.85,
            "confidence": 0.9,
            "created_at": "2026-05-20T10:00:00",
        },
    ]

    frag_block = format_fragments(fragments)
    assert "<memory" in frag_block, "format_fragments should produce <memory> tags"
    r.note(f"Fragment block length: {len(frag_block)} chars")

    messages = assemble(
        system=SYSTEM_PROMPT,
        global_profile=None,
        project_summary="A Python web service.",
        fragments=fragments,
        file_contents=None,
        query="How to fix Docker?",
        project_id="proj_abc123",
    )

    assert isinstance(messages, list), "assemble() should return a list"
    assert len(messages) >= 1, "Should have at least 1 message"
    r.note(f"Assembled {len(messages)} message(s)")

    # Check that SYSTEM_PROMPT exists and is non-empty
    assert len(SYSTEM_PROMPT) > 50, f"SYSTEM_PROMPT is too short ({len(SYSTEM_PROMPT)} chars)"
    r.note(f"SYSTEM_PROMPT: {len(SYSTEM_PROMPT)} chars")

    r.ok("Prompt module assembly verified")


# =============================================================================
#  INTEGRATION: Full ingest → retrieve round-trip
# =============================================================================

@_make_test("INTEGRATION", "Ingest transcript → retrieve by query (full round-trip)")
def test_full_roundtrip(r: TestResult):
    from memory_system.schema import Database
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.config import CompressionConfig, RetrievalConfig
    from memory_system.pipeline import ConsolidationPipeline, TranscriptMessage
    from memory_system.retrieval import fused_retrieval, assemble_prompt
    import memory_system.llm as llm_module

    _original = llm_module.call_model_json

    def mock_llm(prompt, **kwargs):
        if "Extract structured information" in prompt:
            return {
                "intent": "Set up CI/CD pipeline",
                "outcome": "Configured GitHub Actions with pytest",
                "key_decisions": ["Use pytest-cov for coverage"],
                "tools_used": ["write_file"],
                "files_modified": [".github/workflows/ci.yml"],
                "errors_encountered": [],
                "summary": "Set up GitHub Actions CI pipeline with pytest and coverage reporting.",
                "confidence": 0.88,
            }
        elif "Extract all code entities" in prompt:
            return {
                "entities": [{"type": "config", "name": ".github/workflows/ci.yml"}],
                "relations": [],
            }
        return {"summary": "mock", "confidence": 0.7}

    llm_module.call_model_json = mock_llm

    try:
        tmpdir = scratch_dir()
        db = Database(tmpdir / "roundtrip.db")
        db.connect()
        idx = VectorIndex(dim=128, persist_path=str(tmpdir / "rt.hnsw"))
        idx.init_fresh()
        emb = RandomEmbedder(dim=128)

        # ── INGEST ──────────────────────────────────────────────────────
        # Create the session record that fragments will reference
        from memory_system.models import Session
        db.upsert_session(Session(id="sess_rt", project_id="proj_rt"))

        pipeline = ConsolidationPipeline(
            db, idx, CompressionConfig(), embedder=emb
        )
        messages = [
            TranscriptMessage(role="user", content="Set up CI/CD with GitHub Actions"),
            TranscriptMessage(role="assistant", content="I'll create a workflow file with pytest."),
            TranscriptMessage(role="tool", content="Created .github/workflows/ci.yml"),
            TranscriptMessage(role="user", content="Perfect, thanks!"),
        ]
        ingested = pipeline.ingest(messages, session_id="sess_rt", project_id="proj_rt")
        r.note(f"Ingested {len(ingested)} fragment(s)")

        # ── RETRIEVE ────────────────────────────────────────────────────
        q_emb = emb.embed("CI pipeline GitHub Actions pytest")
        cfg = RetrievalConfig(default_token_budget=500, max_fragments=5, min_crs=0.0)

        result = fused_retrieval(
            db=db, vector_index=idx,
            query_embedding=q_emb, query_text="CI pipeline GitHub Actions pytest",
            project_id="proj_rt", token_budget=500, cfg=cfg,
        )

        assert len(result.fragments) >= 1, \
            f"Round-trip should retrieve at least 1 fragment, got {len(result.fragments)}"
        r.note(f"Retrieved {len(result.fragments)} fragment(s), {result.token_budget_used} tokens")

        # ── ASSEMBLE PROMPT ─────────────────────────────────────────────
        prompt = assemble_prompt(
            system_prompt="You are a helpful assistant.",
            global_profile=None,
            result=result,
            user_query="How did we set up CI?",
        )
        assert "CI" in prompt or "ci" in prompt.lower(), "Prompt should contain CI-related content"
        r.note(f"Final prompt: {len(prompt)} chars")

        db.close()

    finally:
        llm_module.call_model_json = _original

    r.ok("Full ingest → retrieve → assemble round-trip verified")


# =============================================================================
#  BUG-FIX REGRESSION TESTS
# =============================================================================

@_make_test("REGRESSION", "retrieval K defined before structural lane (NameError fix)")
def test_retrieval_k_ordering(r: TestResult):
    """Verifies K is defined before the structural pattern lane uses it."""
    import ast, inspect
    from memory_system import retrieval as retrieval_mod

    src = inspect.getsource(retrieval_mod.fused_retrieval)
    tree = ast.parse(src)

    # Walk all Assign nodes; record first line where K is assigned.
    k_assign_line = None
    k_use_line = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "K":
                    if k_assign_line is None:
                        k_assign_line = node.lineno
        if isinstance(node, ast.BinOp):
            # Look for (K + rank) patterns — K is a Name on the left
            if isinstance(node.left, ast.Name) and node.left.id == "K":
                if k_use_line is None:
                    k_use_line = node.lineno

    assert k_assign_line is not None, "K = cfg.rrf_k not found in fused_retrieval"
    assert k_use_line is not None, "No (K + rank) expression found in fused_retrieval"
    assert k_assign_line < k_use_line, (
        f"K is assigned on line {k_assign_line} but first used on line {k_use_line} — "
        "NameError will occur when structural lane fires"
    )
    r.note(f"K assigned at local line {k_assign_line}, first used at line {k_use_line}")
    r.ok("K is defined before its first use — NameError fixed")


@_make_test("REGRESSION", "structural lane does not crash when patterns_index is provided")
def test_structural_lane_no_crash(r: TestResult):
    """
    Exercises the structural pattern lane path so the K NameError would surface
    if not fixed.  Uses a stub patterns_index so the branch is entered.
    """
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.retrieval import fused_retrieval
    from memory_system.config import RetrievalConfig
    from memory_system.ids import new_id

    emb = RandomEmbedder(dim=128)

    tmpdir = scratch_dir()
    db = Database(tmpdir / "test_struct.db")
    db.connect()
    idx = VectorIndex(dim=128, persist_path=str(tmpdir / "struct.hnsw"))
    idx.init_fresh()

    now = datetime.now(tz=timezone.utc).isoformat()
    fid = new_id()
    vec = emb.embed("some content about structural patterns")
    frag = MemoryFragment(
        id=fid, project_id="p1", scope="project",
        category="fact", content="some content about structural patterns",
        token_count=8, confidence=0.8, created_at=now, last_accessed=now,
        embedding_model="random", embedding_dim=128, embedding=vec,
    )
    db.upsert_fragment(frag)
    idx.add(fid, vec)

    # Build a stub patterns_index whose query returns no results — enough
    # to enter the structural lane branch and trigger the K usage.
    class _StubPatternIndex:
        def query(self, emb, k=5):
            return []  # no hits, but branch is entered

    cfg = RetrievalConfig(
        default_token_budget=500, max_fragments=5, min_crs=0.0,
        structural_gate=1.0,  # always enter structural lane
    )

    # This would raise NameError if K is still defined after the lane block.
    result = fused_retrieval(
        db=db, vector_index=idx,
        query_embedding=emb.embed("query"),
        query_text="query",
        project_id="p1",
        token_budget=500,
        cfg=cfg,
        patterns_index=_StubPatternIndex(),
    )

    db.close()
    assert result is not None
    r.ok("Structural lane executed without NameError — fix confirmed")


@_make_test("REGRESSION", "savings counter accumulates when token counts are passed via ingest")
def test_savings_token_accumulation(r: TestResult):
    """
    Verifies that upsert_session accumulates cost_input_tok / cost_output_tok
    so the savings dashboard can report non-zero values.
    """
    from memory_system.schema import Database
    from memory_system.models import Session
    from memory_system.ids import new_id

    tmpdir = scratch_dir()
    db = Database(tmpdir / "test_savings.db")
    db.connect()

    now = datetime.now(tz=timezone.utc).isoformat()
    sid = new_id()

    # First ingest — no tokens (simulates old MCP behaviour)
    db.upsert_session(Session(
        id=sid, project_id="p1",
        started_at=now, ended_at=now,
        turn_count=2,
        cost_input_tok=0, cost_output_tok=0,
    ))

    # Re-ingest same session with token counts (simulates fix)
    db.upsert_session(Session(
        id=sid, project_id="p1",
        started_at=now, ended_at=now,
        turn_count=3,
        cost_input_tok=1500, cost_output_tok=300,
    ))

    rows = [dict(r) for r in db.fetchall("SELECT * FROM sessions WHERE id=?", (sid,))]
    assert len(rows) == 1, "Should be one session row"
    row = rows[0]
    assert row["cost_input_tok"] == 1500, f"Expected 1500 input tokens, got {row['cost_input_tok']}"
    assert row["cost_output_tok"] == 300, f"Expected 300 output tokens, got {row['cost_output_tok']}"
    assert row["turn_count"] == 5, f"Expected accumulated turn_count=5, got {row['turn_count']}"

    # Simulate savings calculation
    total_input = row["cost_input_tok"]
    tokens_saved = max(0, int(total_input * 4.3) - total_input)
    cost_saved = round((tokens_saved / 1_000_000) * 5.40, 2)
    assert tokens_saved > 0, f"tokens_saved should be > 0, got {tokens_saved}"
    r.note(f"1500 input tokens → {tokens_saved} tokens saved → ${cost_saved:.2f} saved")

    db.close()
    r.ok("Token accumulation in sessions table confirmed — savings counter will work")


@_make_test("REGRESSION", "MCP memory_save signature accepts input_tokens and output_tokens")
def test_mcp_memory_save_signature(r: TestResult):
    """Confirms memory_save accepts the new token count parameters."""
    import inspect
    from memory_system.mcp_server import memory_save

    sig = inspect.signature(memory_save)
    params = sig.parameters

    assert "input_tokens" in params, "memory_save is missing input_tokens parameter"
    assert "output_tokens" in params, "memory_save is missing output_tokens parameter"

    # Defaults should be 0 (backward-compatible)
    assert params["input_tokens"].default == 0, \
        f"input_tokens default should be 0, got {params['input_tokens'].default}"
    assert params["output_tokens"].default == 0, \
        f"output_tokens default should be 0, got {params['output_tokens'].default}"

    r.note("memory_save(content, project_id='', input_tokens=0, output_tokens=0)")
    r.ok("MCP memory_save signature is backward-compatible with token tracking")


@_make_test("L8-Agent", "prompt-cache breakpoint marks the last block without mutating input")
def test_cache_breakpoint(r: TestResult):
    """_apply_cache_breakpoint must add an ephemeral cache_control marker to the
    final content block so Anthropic caches the stable prefix — and must never
    mutate the caller's messages (build_segments relies on plain-string content).
    """
    from memory_system.agent import _apply_cache_breakpoint

    # String content (the assembled memory msg0) → wrapped into a marked text block
    msgs = [{"role": "user", "content": "<user_profile>...</user_profile>\n\nhello"}]
    out = _apply_cache_breakpoint(msgs)
    block = out[-1]["content"][-1]
    assert block["cache_control"] == {"type": "ephemeral"}, "marker missing on string content"
    assert block["text"].endswith("hello"), "text content lost"
    assert msgs[0]["content"] == "<user_profile>...</user_profile>\n\nhello", \
        "original message was mutated (must stay a plain string)"
    r.note("string content → marked text block, original untouched")

    # List content (tool_results) → only the LAST block gets the marker
    msgs2 = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "b", "content": "r2"},
    ]}]
    out2 = _apply_cache_breakpoint(msgs2)
    assert "cache_control" not in out2[-1]["content"][0], "first block should not be marked"
    assert out2[-1]["content"][1]["cache_control"] == {"type": "ephemeral"}, "last block not marked"
    assert "cache_control" not in msgs2[0]["content"][1], "original list block was mutated"

    # Empty input is a no-op (no crash)
    assert _apply_cache_breakpoint([]) == [], "empty messages should pass through"
    r.ok("Cache breakpoint applied to last block only, inputs never mutated")


@_make_test("REGRESSION", "prompt-cache tokens accumulate and cache_hit_rate is computed")
def test_cache_token_accumulation(r: TestResult):
    """Cache read/write tokens must accumulate across re-ingests of one session
    (mirroring cost_input_tok), and cache_stats must report a hit rate so
    `mem status` can surface it as a flywheel vital sign."""
    from memory_system.schema import Database
    from memory_system.models import Session
    from memory_system.ids import new_id

    tmpdir = scratch_dir()
    db = Database(tmpdir / "test_cache.db")
    db.connect()

    now = datetime.now(tz=timezone.utc).isoformat()
    sid = new_id()

    # Turn 1: a cache write (cold prefix) plus some uncached input
    db.upsert_session(Session(
        id=sid, project_id="p1", started_at=now, ended_at=now,
        turn_count=1, cost_input_tok=200,
        cache_read_tok=0, cache_creation_tok=1000,
    ))
    # Turn 2 (same session): the prefix is now warm → cache reads
    db.upsert_session(Session(
        id=sid, project_id="p1", started_at=now, ended_at=now,
        turn_count=1, cost_input_tok=100,
        cache_read_tok=3000, cache_creation_tok=0,
    ))

    row = dict(db.fetchone("SELECT * FROM sessions WHERE id=?", (sid,)))
    assert row["cache_read_tok"] == 3000, f"read accumulation wrong: {row['cache_read_tok']}"
    assert row["cache_creation_tok"] == 1000, f"write accumulation wrong: {row['cache_creation_tok']}"

    stats = db.cache_stats()
    # hit_rate = read / (read + write) = 3000 / 4000 = 0.75
    assert stats["cache_hit_rate"] == 0.75, f"expected 0.75, got {stats['cache_hit_rate']}"
    assert stats["cache_read_tok"] == 3000 and stats["cache_creation_tok"] == 1000
    r.note(f"3000 read / 1000 write → {stats['cache_hit_rate']*100:.0f}% hit rate")

    db.close()
    r.ok("Cache tokens accumulate and cache_hit_rate is reported for mem status")


@_make_test("REGRESSION", "failed distillation leaves producer NULL (no provenance lie)")
def test_failed_distillation_no_producer_stamp(r: TestResult):
    """When the distillation model is unreachable, the pipeline stores a raw
    excerpt instead. That excerpt was judged by NO model, so it must NOT be
    stamped with a producer — otherwise the upgrade detector treats raw,
    unjudged text as already processed by a capable model and skips it."""
    from memory_system.schema import Database
    from memory_system.vector_index import VectorIndex
    from memory_system.config import CompressionConfig
    from memory_system.pipeline import ConsolidationPipeline, TranscriptMessage
    from memory_system.embedder import RandomEmbedder
    from memory_system.models import Session

    # A router that names a capable model but FAILS every distillation call —
    # exactly the "configured cloud model, no API key" situation on this box.
    class DeadRouter:
        def model_for(self, role): return "claude-haiku-4-5-20251001"
        def call_json(self, role, prompt, system="", max_tokens=200):
            raise ConnectionError("simulated: distillation model unreachable")

    tmpdir = scratch_dir()
    db = Database(tmpdir / "no_stamp.db")
    db.connect()
    try:
        idx = VectorIndex(dim=128, persist_path=str(tmpdir / "no_stamp.hnsw"))
        idx.init_fresh()
        emb = RandomEmbedder(dim=128)
        cfg = CompressionConfig(consolidation_threshold=3, max_episode_tokens=500)
        db.upsert_session(Session(id="sess_dead", project_id="p1"))

        pipeline = ConsolidationPipeline(db, idx, cfg, embedder=emb, router=DeadRouter())
        assert pipeline._producer_model == "claude-haiku-4-5-20251001", \
            "guard the premise: a successful distill WOULD stamp this producer"

        messages = [
            TranscriptMessage(role="user", content="Fix the Docker build, context too large"),
            TranscriptMessage(role="assistant", content="Missing a .dockerignore — creating one."),
            TranscriptMessage(role="user", content="Great, that fixed it! Remember this.",
                              is_explicit_remember=True),
        ]
        frags = pipeline.ingest(messages, session_id="sess_dead", project_id="p1")

        distilled = [f for f in frags if f.source_type == "distillation"]
        assert distilled, "fallback should still store a raw excerpt fragment"
        for f in distilled:
            assert f.producer_model is None, \
                f"unjudged fallback fragment must not be stamped, got {f.producer_model!r}"
            assert f.producer_version is None, \
                f"unjudged fallback fragment must not carry a producer version"
        r.note(f"{len(distilled)} fallback fragment(s) stored with producer NULL")
        r.ok("Failed distillation leaves producer NULL — upgrade detector won't be fooled")
    finally:
        db.close()


# =============================================================================
#  RUN ALL TESTS
# =============================================================================

@_make_test("L8-Mirror", "Goal proposal lifecycle: confirm promotes, dismiss is source-scoped")
def test_goal_proposal_lifecycle(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import Goal
    from datetime import datetime, timezone

    def now():
        return datetime.now(tz=timezone.utc).isoformat()

    db = Database(scratch_dir() / "goals.db")
    db.connect()
    try:
        db.insert_goal(Goal(id="g_user", project_id="p1", statement="ship the thing",
                            source="user", created_at=now()))
        db.insert_goal(Goal(id="g_prop", project_id="p1", statement="write more tests",
                            source="proposed", created_at=now()))

        # Listing partitions cleanly by source.
        user_open = db.list_goals("p1", status="open", source="user")
        prop_open = db.list_goals("p1", status="open", source="proposed")
        assert [g.id for g in user_open] == ["g_user"], f"user list wrong: {[g.id for g in user_open]}"
        assert [g.id for g in prop_open] == ["g_prop"], f"proposal list wrong: {[g.id for g in prop_open]}"
        r.note("list_goals partitions user vs proposed correctly")

        # confirm promotes proposed -> user, is idempotent, and refuses a user goal.
        assert db.confirm_goal("g_prop", "p1") is True, "confirm of proposal should succeed"
        assert db.confirm_goal("g_prop", "p1") is False, "confirm should be idempotent"
        assert db.confirm_goal("g_user", "p1") is False, "confirm must refuse a user goal"
        assert {g.id for g in db.list_goals("p1", status="open", source="user")} == {"g_user", "g_prop"}
        assert db.list_goals("p1", status="open", source="proposed") == []
        r.note("confirm promotes once, refuses non-proposals")

        # dismiss_proposal is source-scoped: it must NOT close a user-owned goal.
        assert db.dismiss_proposal("g_user", now(), "p1") is False, \
            "dismiss must refuse a user-owned goal (Finding 1)"
        assert db.get_goal("g_user").status == "open", "user goal must stay open after dismiss attempt"
        r.note("dismiss refuses to close a user-owned goal")

        # dismiss only retires a genuine pending proposal.
        db.insert_goal(Goal(id="g_prop2", project_id="p1", statement="tidy the docs",
                            source="proposed", created_at=now()))
        assert db.dismiss_proposal("g_prop2", now(), "p1") is True
        assert db.get_goal("g_prop2").status == "closed"
        r.note("dismiss closes a pending proposal")

        # project scoping: a goal in another project is untouched (Finding 2).
        db.insert_goal(Goal(id="g_other", project_id="OTHER", statement="x",
                            source="proposed", created_at=now()))
        assert db.confirm_goal("g_other", "p1") is False, "confirm must respect project_id"
        assert db.get_goal("g_other").source == "proposed", "cross-project goal must be untouched"
        r.note("confirm/dismiss respect project_id")

        r.ok("Goal proposal lifecycle verified")
    finally:
        db.close()


@_make_test("L9-Reindex", "Reindex health snapshot captures before/after deltas")
def test_reindex_health_snapshot(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment, Goal
    from memory_system.reindex import ModelUpgradeReindexJob, HealthSnapshot
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    def now():
        return datetime.now(tz=timezone.utc).isoformat()

    def frag(content, conf, category="fact", deprecated=False):
        return MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category=category,
            content=content, token_count=10, confidence=conf,
            source_type="distillation", created_at=now(), last_accessed=now(),
            embedding_model="random", embedding_dim=128,
            is_deprecated=deprecated,
        )

    db = Database(scratch_dir() / "reindex_health.db")
    db.connect()
    try:
        # Seed: 2 healthy facts, 1 vague fact (<0.70), 1 already-deprecated dup.
        db.upsert_fragment(frag("Builds need a lockfile.", 0.90))
        db.upsert_fragment(frag("Tests run in CI.", 0.85))
        vague = frag("something about deploys maybe", 0.50)
        db.upsert_fragment(vague)
        db.upsert_fragment(frag("dup of lockfile", 0.90, deprecated=True))
        db.insert_goal(Goal(id="g1", project_id="p1", statement="ship", created_at=now()))
        db.insert_goal(Goal(id="g2", project_id="p1", statement="learn", created_at=now()))

        # The job's snapshot is read-only; build a bare instance to call it.
        job = ModelUpgradeReindexJob.__new__(ModelUpgradeReindexJob)
        job.db = db

        before = job._snapshot_health("p1")
        assert before.active_fragments == 3, f"active: {before.active_fragments}"
        assert before.low_confidence_facts == 1, f"low-conf: {before.low_confidence_facts}"
        assert before.deprecated_fragments == 1, f"deprecated: {before.deprecated_fragments}"
        assert before.open_goals == 2, f"goals: {before.open_goals}"
        assert 0.74 < before.avg_confidence < 0.76, f"avg: {before.avg_confidence}"
        r.note(f"before: {before.as_dict()}")

        # Simulate a good crank: lift the vague fact, close a goal, merge a dup.
        db.execute("UPDATE memory_fragments SET confidence=0.85 WHERE id=?", (vague.id,))
        db.close_goal("g1", now())
        db.upsert_fragment(frag("newly merged dup", 0.90, deprecated=True))

        after = job._snapshot_health("p1")
        assert after.low_confidence_facts == 0, f"low-conf after: {after.low_confidence_facts}"
        assert after.open_goals == 1, f"goals after: {after.open_goals}"
        assert after.deprecated_fragments == 2, f"deprecated after: {after.deprecated_fragments}"
        assert after.avg_confidence > before.avg_confidence, "confidence should rise"
        r.note(f"after : {after.as_dict()}")

        # Snapshot must never mutate the store (read-only invariant).
        assert job._snapshot_health("p1").as_dict() == after.as_dict(), \
            "snapshot must be idempotent / read-only"

        # The scorecard renderer survives empty/missing input without raising.
        from memory_system.cli import _print_reindex_scorecard
        _print_reindex_scorecard(None, after.as_dict())
        _print_reindex_scorecard(before.as_dict(), after.as_dict())

        r.ok("Health snapshot tracks confidence, goals, dedup, and is read-only")
    finally:
        db.close()


@_make_test("L9-Reindex", "Model-upgrade detector ranks models and flags a real gap")
def test_upgrade_detector(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.upgrade import detect_upgrade, model_rank
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    # Ranking: NULL < unknown-equivalent < haiku < sonnet < opus, version-aware.
    assert model_rank(None) < model_rank("claude-haiku-4-5"), "NULL must rank below any real model"
    assert model_rank("claude-haiku-4-5-20251001") == model_rank("claude-haiku-4-5"), \
        "dated id must resolve to its undated stem"
    assert model_rank("claude-opus-4-8") > model_rank("claude-sonnet-4-6") > model_rank("claude-haiku-4-5"), \
        "tier ordering wrong"
    assert model_rank("claude-opus-4-8") > model_rank("claude-opus-4-7"), "version ordering wrong"
    r.note("model_rank: NULL < haiku < sonnet < opus, version-aware, date-tolerant")

    def now():
        return datetime.now(tz=timezone.utc).isoformat()

    def frag(producer, conf=0.9):
        return MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="fact",
            content="x", token_count=5, confidence=conf, source_type="distillation",
            created_at=now(), last_accessed=now(),
            embedding_model="random", embedding_dim=128,
            producer_model=producer,
        )

    db = Database(scratch_dir() / "upgrade.db")
    db.connect()
    try:
        # Store built by a mix of NULL (pre-provenance) and an older model.
        db.upsert_fragment(frag(None))
        db.upsert_fragment(frag(None))
        db.upsert_fragment(frag("claude-haiku-4-5"))

        # Running a stronger model now -> upgrade available, all 3 behind.
        up = detect_upgrade(db, "claude-opus-4-8", project_id="p1")
        assert up.upgrade_available is True, "should detect an upgrade"
        assert up.fragments_total == 3, f"total: {up.fragments_total}"
        assert up.fragments_behind == 3, f"behind: {up.fragments_behind}"
        assert up.stored_rank < up.current_rank, "stored must rank below current"
        r.note(f"behind={up.fragments_behind}/{up.fragments_total} -> {up.note[:60]}...")

        # Running the SAME weak model that built it -> nothing to gain.
        same = detect_upgrade(db, "claude-haiku-4-5", project_id="p1")
        # NULL fragments still rank below haiku, so they remain 'behind' — but a
        # store built entirely by the current model must report no upgrade:
        db2 = Database(scratch_dir() / "upgrade_current.db"); db2.connect()
        try:
            db2.upsert_fragment(frag("claude-opus-4-8"))
            cur = detect_upgrade(db2, "claude-opus-4-8", project_id="p1")
            assert cur.upgrade_available is False, "current store must report no upgrade"
            assert cur.fragments_behind == 0, f"behind should be 0: {cur.fragments_behind}"
        finally:
            db2.close()
        r.note("no false upgrade when store is already current")

        r.ok("Upgrade detector ranks models and flags only genuine gaps")
    finally:
        db.close()


@_make_test("L9-Reindex", "Resynthesis routes through the local router and stamps provenance")
def test_resynthesis_uses_router(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.reindex import ModelUpgradeReindexJob
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()

    # A fake router that stands in for a reachable local model.
    class FakeRouter:
        def __init__(self): self.calls = 0
        def model_for(self, role): return "ollama/qwen3:8b"
        def call_json(self, role, prompt, max_tokens=200):
            self.calls += 1
            assert role == "medium", f"resynthesis must use 'medium', got {role!r}"
            return {"rewritten": "Crisp, de-vagued restatement of the fact."}

    db = Database(scratch_dir() / "resynth.db")
    db.connect()
    try:
        low = MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="fact",
            content="vague mumble about something", token_count=6, confidence=0.50,
            source_type="distillation", created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, producer_model=None,
        )
        db.upsert_fragment(low)
        # A high-confidence fact must be left untouched (only <0.70 are candidates).
        high = MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="fact",
            content="already crisp fact", token_count=4, confidence=0.95,
            source_type="distillation", created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, producer_model=None,
        )
        db.upsert_fragment(high)

        fake = FakeRouter()
        job = ModelUpgradeReindexJob.__new__(ModelUpgradeReindexJob)
        job.db = db
        job._router = fake
        job._progress_cb = None  # built via __new__, so init never set this

        n = job._resynthesize_facts("p1")
        assert n == 1, f"exactly one low-conf fact should be rewritten, got {n}"
        assert fake.calls == 1, "router must be the path used (not the cloud fallback)"

        got = db.get_fragment(low.id)
        assert got.content == "Crisp, de-vagued restatement of the fact.", "content not rewritten"
        assert abs(got.confidence - 0.50) < 1e-6, \
            f"rewriting must NOT change confidence (stays 0.50), got {got.confidence}"
        assert got.producer_model == "ollama/qwen3:8b", \
            f"rewritten fact must be stamped with the producer, got {got.producer_model!r}"
        r.note("low-conf fact rewritten via 'medium' role, provenance stamped, confidence untouched")

        untouched = db.get_fragment(high.id)
        assert untouched.confidence == 0.95 and untouched.producer_model is None, \
            "high-confidence fact must be left alone"
        r.note("high-confidence fact left untouched")

        r.ok("Resynthesis uses the local router and stamps provenance off NULL")
    finally:
        db.close()


@_make_test("L9-Reindex", "Resynthesis rewords but never changes confidence")
def test_resynthesis_never_inflates_confidence(r: TestResult):
    """Rewriting a fact updates its wording + provenance but must leave confidence
    untouched. A rephrase is not evidence of truth, so it can't raise trust — even
    a clearly different (substantive) rewrite leaves the score where it was."""
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.reindex import ModelUpgradeReindexJob
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()
    original = "vague mumble about something"
    rewrite = "A clear, de-vagued restatement that reads nothing like the original."

    class FakeRouter:
        def model_for(self, role): return "ollama/qwen3:8b"
        def call_json(self, role, prompt, max_tokens=200):
            return {"rewritten": rewrite}

    db = Database(scratch_dir() / "noinflate.db"); db.connect()
    try:
        frag = MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="fact",
            content=original, token_count=6, confidence=0.50,
            source_type="distillation", created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, producer_model=None,
        )
        db.upsert_fragment(frag)

        job = ModelUpgradeReindexJob.__new__(ModelUpgradeReindexJob)
        job.db = db
        job._router = FakeRouter()
        job._progress_cb = None

        n = job._resynthesize_facts("p1")
        assert n == 1, f"a genuinely changed rewrite is still adopted, got {n}"

        got = db.get_fragment(frag.id)
        assert got.content == rewrite, "wording should be updated"
        assert got.producer_model == "ollama/qwen3:8b", "provenance should be stamped"
        assert abs(got.confidence - 0.50) < 1e-6, \
            f"confidence must stay 0.50 even on a substantive rewrite, got {got.confidence}"
        r.ok("Reword updated content + provenance, left confidence at 0.50")
    finally:
        db.close()


@_make_test("L9-Reindex", "Reindex reports progress to a callback (background-job contract)")
def test_reindex_progress_callback(r: TestResult):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.reindex import ModelUpgradeReindexJob
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.config import SelfImprovementConfig
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "prog.db"); db.connect()
        idx = VectorIndex(dim=128, persist_path=str(Path(tmpdir) / "prog.hnsw"))
        idx.init_fresh()
        try:
            for i in range(7):
                db.upsert_fragment(MemoryFragment(
                    id=new_id(), project_id="p1", scope="project", category="episode",
                    content=f"fragment number {i}", token_count=5, confidence=0.8,
                    source_type="distillation", created_at=now, last_accessed=now,
                    embedding_model="random", embedding_dim=128,
                ))

            events: list[tuple] = []
            job = ModelUpgradeReindexJob(
                db=db, vector_index=idx, new_embedder=RandomEmbedder(dim=128),
                cfg=SelfImprovementConfig(),
                progress_cb=lambda phase, done, total: events.append((phase, done, total)),
            )
            report = job.execute("p1")

            assert report.fragments_reindexed == 7, f"reindexed: {report.fragments_reindexed}"
            assert events, "progress callback was never called"
            phases = {e[0] for e in events}
            assert "re-embedding" in phases, f"missing re-embedding phase: {phases}"
            # Progress must reach the full count and never overshoot the total.
            reembed = [e for e in events if e[0] == "re-embedding"]
            assert reembed[-1][1] == 7, f"final done should be 7: {reembed[-1]}"
            assert all(done <= total for _, done, total in reembed), "done overshot total"
            r.note(f"progress events: {len(events)}, final={reembed[-1]}")
            r.ok("Reindex emits monotonic progress a daemon can relay to a poller")
        finally:
            db.close()


def main():
    print()
    print("=" * 78)
    print("  MEMORY SYSTEM -- END-TO-END SMOKE TEST SUITE")
    print("=" * 78)
    print()

    # Collect and run all test functions from this module's globals
    test_fns = [
        v for v in globals().values()
        if callable(v) and getattr(v, "_test", False)
    ]

    # Sort by layer name for deterministic ordering
    test_fns.sort(key=lambda fn: (fn._layer, fn._name))

    for fn in test_fns:
        fn()

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("-" * 78)
    current_layer = ""
    passed = 0
    failed = 0

    for r in results:
        if r.layer != current_layer:
            current_layer = r.layer
            print(f"\n  +-- {current_layer} {'-' * (70 - len(current_layer))}")

        status = "PASS" if r.passed else "FAIL"
        print(f"  |  [{status}]  {r.name}  ({r.elapsed_ms:.0f}ms)")
        for detail in r.details:
            print(f"  |         -> {detail}")
        if r.error:
            # Print first 3 lines of error
            for line in r.error.strip().split("\n")[:3]:
                print(f"  |         !! {line}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print(f"\n  +{'-' * 76}")
    print()
    print("=" * 78)
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} TESTS PASSED -- your memory system is fully wired!")
    else:
        print(f"  WARNING: {failed}/{total} TESTS FAILED -- review errors above")
    print("=" * 78)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
