"""
Embedder protocol + implementations.

Plug in whichever backend is available:
  RandomEmbedder    — deterministic random vectors for testing (no deps)
  OllamaEmbedder    — local models via Ollama REST API (recommended)
  OpenAIEmbedder    — text-embedding-3-large via openai SDK (optional)

The pipeline calls embedder.embed() / embedder.embed_batch() and stores the
resulting vectors in the VectorIndex.  The embedding_model name is stored in
every MemoryFragment so the index can be re-built when the model changes.
"""

from __future__ import annotations
import hashlib
import math
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ── Random (test / offline) ──────────────────────────────────────────────────

class RandomEmbedder:
    """
    Deterministic pseudo-random embeddings derived from text hash.
    Same text always produces the same unit vector — useful for testing
    the full pipeline without an API key or local model.
    """
    model_name = "random"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        # Seed numpy RNG from SHA256 of text — always produces finite unit vectors
        seed_bytes = hashlib.sha256(text.encode()).digest()
        seed_int = int.from_bytes(seed_bytes, "big") % (2**32)
        rng = np.random.default_rng(seed_int)
        vec = rng.standard_normal(self.dim).astype(np.float64)
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ── Ollama (local, recommended) ──────────────────────────────────────────────

class OllamaEmbedder:
    """
    Uses Ollama's /api/embed endpoint for local embedding models.
    Requires Ollama running at http://localhost:11434 (or custom endpoint).
    Default model: nomic-embed-text (768-dim, ~274MB).

    embed_batch() sends one request for the entire batch using the /api/embed
    array input — Ollama processes all texts in a single GPU forward pass.
    Falls back to parallel single requests if the batch endpoint is unavailable
    (Ollama < 0.1.34).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        endpoint: str = "http://localhost:11434",
        dim: int = 768,
    ):
        import urllib.request as _req  # stdlib — no extra deps
        self._req = _req
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self.model_name = f"ollama/{model}"
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.error
        if not texts:
            return []
        # /api/embed accepts {"model": ..., "input": [str, ...]} and returns
        # {"embeddings": [[float, ...], ...]} — one GPU pass for the whole batch.
        body = json.dumps({"model": self._model, "input": texts}).encode()
        req = self._req.Request(
            f"{self._endpoint}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._req.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            return result["embeddings"]
        except (urllib.error.HTTPError, KeyError):
            # HTTPError 404 → old Ollama without /api/embed; KeyError → response
            # has no "embeddings" key (old single-result format). Fall back to
            # parallel single requests via the legacy /api/embeddings endpoint.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(self._embed_single, texts))

    def _embed_single(self, text: str) -> list[float]:
        import json
        body = json.dumps({"model": self._model, "prompt": text}).encode()
        req = self._req.Request(
            f"{self._endpoint}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._req.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return result["embedding"]


# ── OpenAI (optional) ────────────────────────────────────────────────────────

class OpenAIEmbedder:
    """
    Wraps the openai SDK. Requires:  pip install openai
    Uses batching to stay within the API rate limits.
    """
    model_name = "text-embedding-3-large"
    dim = 1024  # native dimension for text-embedding-3-large

    def __init__(self, api_key: str | None = None, batch_size: int = 100):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install openai  to use OpenAIEmbedder")
        import os
        self._client = _openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._batch_size = batch_size

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(
                model=self.model_name,
                input=chunk,
                dimensions=self.dim,
            )
            results.extend(item.embedding for item in resp.data)
        return results


# ── Cached wrapper ───────────────────────────────────────────────────────────

class CachedEmbedder:
    """
    LRU-like in-memory cache around any Embedder.
    Shared across all sessions — matches the daemon's shared embedding cache
    described in Section 3.2 of the architecture.
    """

    def __init__(self, inner: Embedder, max_entries: int = 10_000):
        self._inner = inner
        self._max = max_entries
        self._cache: dict[str, list[float]] = {}
        self._order: list[str] = []

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        vec = self._inner.embed(text)
        self._store(text, vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
            else:
                results.append([])  # placeholder
                missing_idx.append(i)
                missing_texts.append(t)
        if missing_texts:
            vecs = self._inner.embed_batch(missing_texts)
            for i, (idx, text, vec) in enumerate(zip(missing_idx, missing_texts, vecs)):
                results[idx] = vec
                self._store(text, vec)
        return results

    def _store(self, text: str, vec: list[float]) -> None:
        if len(self._cache) >= self._max:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[text] = vec
        self._order.append(text)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_embedder(cfg) -> "CachedEmbedder":
    """Construct the configured embedder, wrapped in the shared cache.

    Selection by cfg.embedding.model:
      'ollama/<name>'        → OllamaEmbedder   (local, recommended)
      'text-embedding*' + key → OpenAIEmbedder  (when OPENAI_API_KEY is set)
      anything else          → RandomEmbedder   (offline / test fallback)
    """
    import logging
    import os
    log = logging.getLogger("memoryd")
    model = cfg.embedding.model

    if model.startswith("ollama/"):
        local_model = model[len("ollama/"):]
        inner = OllamaEmbedder(model=local_model, dim=cfg.embedding.dimensions)
        log.info("embedder: Ollama %s dim=%d", inner.model_name, inner.dim)
        return CachedEmbedder(inner, max_entries=cfg.embedding.cache_size)

    if model.startswith("text-embedding") and os.environ.get("OPENAI_API_KEY"):
        try:
            inner = OpenAIEmbedder(batch_size=cfg.embedding.batch_size)
            log.info("embedder: OpenAI %s dim=%d", inner.model_name, inner.dim)
            return CachedEmbedder(inner, max_entries=cfg.embedding.cache_size)
        except ImportError:
            log.warning("openai SDK not installed — falling back to RandomEmbedder")

    log.warning("no embedder matched config model=%r — using RandomEmbedder", model)
    inner = RandomEmbedder(dim=cfg.embedding.dimensions)
    return CachedEmbedder(inner, max_entries=cfg.embedding.cache_size)


# ── Embedding math (single canonical location) ───────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Returns cosine similarity in [-1, 1]. Returns 0.0 on empty or mismatched vecs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    result = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, result))
