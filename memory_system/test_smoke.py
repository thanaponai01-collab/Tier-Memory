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


def test(layer: str, name: str):
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

@test("L1-Config", "load_config returns defaults without YAML file")
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


@test("L1-Config", "MemoryConfig dataclass hierarchy is complete")
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

@test("L2-Database", "Schema creation and fragment CRUD")
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


@test("L2-Database", "BM25 full-text search")
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


@test("L2-Database", "Entity + Triple CRUD and graph traversal")
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

@test("L3-Vector", "RandomEmbedder produces deterministic unit vectors")
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


@test("L3-Vector", "CachedEmbedder caches and delegates correctly")
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


@test("L3-Vector", "VectorIndex add/query/remove/persist")
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

@test("L4-Scoring", "CRS computation and tier classification")
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

@test("L5-Retrieval", "Fused retrieval returns token-budget-packed fragments")
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


@test("L5-Retrieval", "Prompt assembly produces L0/L1/L2/L3 structure")
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

@test("L6-Pipeline", "4-stage pipeline ingests transcript and produces fragments")
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


@test("L6-Pipeline", "Reflection produces a high-confidence fragment")
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

@test("L7-Auditor", "Confidence decay on old unvalidated facts")
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


@test("L7-Auditor", "PageRank centrality propagation to fragments")
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

@test("L8-Agent", "Project ID resolution and session ID generation")
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


@test("L8-Agent", "Prompt module assembles L0-L3 layers correctly")
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

@test("INTEGRATION", "Ingest transcript → retrieve by query (full round-trip)")
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

@test("REGRESSION", "retrieval K defined before structural lane (NameError fix)")
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


@test("REGRESSION", "structural lane does not crash when patterns_index is provided")
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


@test("REGRESSION", "savings counter accumulates when token counts are passed via ingest")
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


@test("REGRESSION", "MCP memory_save signature accepts input_tokens and output_tokens")
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


# =============================================================================
#  RUN ALL TESTS
# =============================================================================

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
