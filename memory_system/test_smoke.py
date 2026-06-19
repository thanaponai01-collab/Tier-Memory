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
from datetime import datetime, timezone, timedelta
from pathlib import Path
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
# These tests run under pytest. Each takes the `r` fixture below — a thin
# recorder kept so the existing bodies' r.note()/r.ok() calls still read well.
# The load-bearing change from the old homegrown runner: r.fail() RAISES (and
# uncaught assertions propagate), so a failing test actually fails the pytest
# run instead of being swallowed into a report the runner ignored.

import pytest


class _Recorder:
    """Per-test scratchpad. note()/ok() are informational (surface via -v on
    failure); fail() raises so the failure reaches the test runner."""

    def __init__(self):
        self.details: list[str] = []

    def ok(self, detail: str = ""):
        if detail:
            self.details.append(detail)

    def note(self, detail: str):
        self.details.append(detail)

    def fail(self, error: str):
        raise AssertionError(error)


@pytest.fixture
def r():
    return _Recorder()


# =============================================================================
#  LAYER 1: Configuration
# =============================================================================

def test_config_defaults(r):
    from memory_system.config import load_config
    cfg = load_config(path="/nonexistent/path/memory.yaml")
    assert cfg.compression.distillation_model == "claude-haiku-4-5-20251001", \
        f"Unexpected model: {cfg.compression.distillation_model}"
    assert cfg.retrieval.default_token_budget == 4000
    assert cfg.eviction.recency_decay_lambda == 0.003
    r.note(f"Compression model: {cfg.compression.distillation_model}")
    r.note(f"Token budget: {cfg.retrieval.default_token_budget}")
    r.ok("All default config values correct")


def test_config_structure(r):
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

def test_db_schema_and_crud(r):
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


def test_embedder_health_watchdog(r):
    """The embedder watchdog must record only state transitions and report an
    honest 'down since / last ok' so the dashboard can't show green while the
    embedder is silently dead."""
    from memory_system.schema import Database

    def ts(h):  # deterministic ISO timestamps, h hours apart
        return (datetime(2026, 6, 6, tzinfo=timezone.utc) + timedelta(hours=h)).isoformat()

    db = Database(scratch_dir() / "health.db")
    db.connect()

    # No history yet → unknown, not falsely green/red.
    h = db.embedder_health()
    assert h["down_since"] is None and h["last_ok_at"] is None, h

    # First alive reading is a transition; an identical follow-up is not.
    assert db.record_embedder_health(True, "ok", ts(0)) is True
    assert db.record_embedder_health(True, "ok", ts(1)) is False, "no flip => no row"
    h = db.embedder_health()
    assert h["last_ok_at"] == ts(0) and h["down_since"] is None, h
    r.note("alive recorded once; repeats suppressed")

    # It dies → one transition row; the down streak starts here.
    assert db.record_embedder_health(False, "WinError 10061", ts(2)) is True
    assert db.record_embedder_health(False, "WinError 10061", ts(3)) is False
    h = db.embedder_health()
    assert h["down_since"] == ts(2), f"down_since should be flip time: {h}"
    assert h["last_ok_at"] == ts(0), f"last ok preserved across outage: {h}"
    r.note(f"DOWN since {h['down_since'][:19]}, last ok {h['last_ok_at'][:19]}")

    # Recovery clears down_since and advances last_ok.
    assert db.record_embedder_health(True, "ok", ts(5)) is True
    h = db.embedder_health()
    assert h["down_since"] is None and h["last_ok_at"] == ts(5), h

    # Transition-only writes keep the log bounded: 3 flips (alive→down→alive),
    # not the 6 readings we fed in.
    rows = db.get_events(kind="embedder_health", limit=100)
    assert len(rows) == 3, f"expected 3 transition rows, got {len(rows)}"

    db.close()


def test_issue_log(r):
    """The system issue log is the watchdog grown up: any caught failure becomes
    a durable, dashboard-visible row. It must dedup a flapping failure, count by
    severity for the health pill, and read back newest-first for the panel."""
    from memory_system.schema import Database

    def ts(m):  # deterministic ISO timestamps, m minutes apart
        return (datetime(2026, 6, 6, tzinfo=timezone.utc) + timedelta(minutes=m)).isoformat()

    db = Database(scratch_dir() / "issues.db")
    db.connect()

    # Clean store → empty summary, not a falsely-alarming one.
    s = db.issue_summary(ts(-60))
    assert s["total"] == 0 and s["last_at"] is None, s

    # Three distinct issues land (explicit timestamps so the summary/panel
    # assertions are deterministic; dedup_window_secs=0 disables the
    # against-now dedup that those synthetic timestamps would otherwise fight).
    assert db.record_issue("ingest", "boom", "error", ts(0), dedup_window_secs=0) is True
    assert db.record_issue("audit", "audit crashed", "error", ts(1), dedup_window_secs=0) is True
    assert db.record_issue("reranker", "refit skipped", "warn", ts(2), dedup_window_secs=0) is True

    # Dedup against real now: an identical failure twice in a row records once,
    # so one flapping error can't bury the panel. (Both use default ts ≈ now,
    # so they fall inside the same window deterministically.)
    assert db.record_issue("flapper", "same error") is True
    assert db.record_issue("flapper", "same error") is False, "duplicate within window suppressed"
    r.note("duplicate flapping error suppressed within dedup window")

    # Summary counts errors and warnings separately for the pill.
    s = db.issue_summary(ts(-60))
    assert s["errors"] == 3 and s["warnings"] == 1, s
    assert s["total"] == 4, s
    r.note(f"summary: {s['errors']} errors, {s['warnings']} warnings")

    # Panel read-back is newest-first and carries component + severity + detail.
    # The flapper row is newest of all (≈now), then reranker/audit/ingest.
    issues = db.recent_issues(since_iso=ts(-60), limit=50)
    assert len(issues) == 4, issues
    assert issues[0]["component"] == "flapper", issues[0]
    assert all({"ts", "component", "severity", "detail"} <= set(i) for i in issues)

    # The since-window actually filters: only flapper (≈now) is newer than ts(2.5).
    recent = db.recent_issues(since_iso=ts(2.5), limit=50)
    assert len(recent) == 1 and recent[0]["component"] == "flapper", recent

    db.close()
    r.ok("watchdog records transitions only and reports honest down/last-ok")


def test_curate_note_to_fragment(r):
    from memory_system.curate import note_to_fragment, build_fragments_from_dir
    from memory_system.schema import Database

    note = (
        "---\n"
        "name: flywheel-north-star\n"
        "description: the user's real goal — a compounding flywheel memory\n"
        "metadata:\n  type: project\n"
        "---\n\n"
        "First focus is the flywheel; upgrade trigger is the 'safe mix'.\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        (notes_dir / "flywheel.md").write_text(note, encoding="utf-8")
        (notes_dir / "MEMORY.md").write_text("# index\n- ignore me\n", encoding="utf-8")

        frag = note_to_fragment(notes_dir / "flywheel.md", "p1")
        assert frag is not None, "expected a fragment"
        # The regression that bit absorb: category must satisfy the DB CHECK.
        assert frag["category"] in ("fact", "episode", "preference", "correction"), \
            f"invalid category {frag['category']!r} would fail the schema CHECK"
        assert "compounding flywheel" in frag["content"], "description should lead content"
        assert "safe mix" in frag["content"], "body should be included"
        assert frag["id"].startswith("cur_"), "curated ids are prefixed cur_"
        assert frag["confidence"] == 1.0, "curated notes are authoritative"
        r.note(f"fragment id {frag['id']}, category {frag['category']}")

        # Stable id → idempotent (same note yields same id).
        again = note_to_fragment(notes_dir / "flywheel.md", "p1")
        assert again["id"] == frag["id"], "id must be stable for idempotent re-absorb"

        # MEMORY.md is skipped; one real note produces one fragment.
        frags = build_fragments_from_dir(notes_dir, "p1")
        assert len(frags) == 1, f"expected 1 fragment (MEMORY.md skipped), got {len(frags)}"

        # And it actually upserts (the real regression was a CHECK failure here).
        db = Database(Path(tmpdir) / "c.db")
        db.connect()
        from memory_system.models import MemoryFragment
        db.upsert_fragment(MemoryFragment(
            id=frag["id"], project_id="p1", scope=frag["scope"],
            category=frag["category"], content=frag["content"],
            token_count=frag["token_count"], confidence=frag["confidence"],
            graph_centrality=frag["graph_centrality"], source_type=frag["source_type"],
        ))
        assert db.get_fragment(frag["id"]) is not None, "fragment should persist"
        db.close()
        r.ok("Curated note parsed, schema-valid, idempotent, persists")


def test_bm25_search(r):
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


def test_recent_temporal_recall(r):
    from memory_system.schema import Database
    from memory_system.models import Session, MemoryFragment
    from memory_system.ids import new_id
    from memory_system.cli import parse_time_window

    # ── the time-window grammar ──────────────────────────────────────────────
    since, limit, _ = parse_time_window(None)
    assert since is not None and limit == 10, "default should be a 7-day window"
    since, limit, _ = parse_time_window("5")
    assert since is None and limit == 5, "bare count => last-N-sessions, no date floor"
    since, limit, _ = parse_time_window("last 3 sessions")
    assert since is None and limit == 3, f"'last 3 sessions' => limit 3, got {limit}"
    since, _, label = parse_time_window("since 2026-06-01")
    assert since.startswith("2026-06-01"), f"ISO date floor not honoured: {since}"
    since_w, _, _ = parse_time_window("last week")
    since_d, _, _ = parse_time_window("3d")
    assert since_w is not None and since_d is not None, "relative windows must set a floor"
    r.note("parse_time_window covers default / count / sessions / ISO / relative")

    # ── the date-ordered query ───────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_recent.db")
        db.connect()

        # Three sessions across three days; newest must come first.
        days = ["2026-06-01T09:00:00+00:00",
                "2026-06-05T09:00:00+00:00",
                "2026-06-08T09:00:00+00:00"]
        ids = []
        for i, started in enumerate(days):
            sid = new_id()
            ids.append(sid)
            db.upsert_session(Session(
                id=sid, project_id="p1", started_at=started,
                summary=(f"did thing {i}" if i != 1 else None),  # middle has no summary
                turn_count=i + 1,
            ))

        # Give the summary-less middle session a fragment to surface as a highlight.
        # Multi-line, answer-shaped: only the first line (headline) should surface;
        # the supporting body must be dropped from the temporal digest.
        now = datetime.now(tz=timezone.utc).isoformat()
        db.upsert_fragment(MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="fact",
            content="middle session produced this fact\n"
                    "The user did supporting detail that should be dropped.",
            token_count=8,
            source_session=ids[1], confidence=0.9,
            created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128,
        ))
        # A lower-confidence EPISODE from the same session. The digest answers
        # "what did I work on", so the descriptive episode must win over the
        # higher-confidence generic fact — episode-preference beats confidence.
        db.upsert_fragment(MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="episode",
            content="extended the temporal recall feature in the read reflex",
            token_count=8,
            source_session=ids[1], confidence=0.5,
            created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128,
        ))

        rows = db.recent_sessions("p1", since_iso=None, limit=10)
        assert len(rows) == 3, f"expected 3 sessions, got {len(rows)}"
        assert rows[0]["started_at"].startswith("2026-06-08"), "must be newest-first"
        assert rows[2]["started_at"].startswith("2026-06-01"), "oldest must be last"
        assert rows[1]["frag_count"] == 2, f"middle frag_count wrong: {rows[1]['frag_count']}"

        # since_iso floor excludes the oldest session.
        windowed = db.recent_sessions("p1", since_iso="2026-06-04T00:00:00+00:00", limit=10)
        assert len(windowed) == 2, f"date floor should leave 2 sessions, got {len(windowed)}"

        highlights = db.fragments_in_session(ids[1], limit=3)
        assert highlights, "highlight fallback missing"
        # Episode-preference: the lower-confidence episode outranks the
        # higher-confidence generic fact — the digest reads like a worklog.
        assert "temporal recall" in highlights[0], \
            f"episode must win over higher-confidence fact, got: {highlights[0]!r}"
        assert "supporting detail" not in " ".join(highlights), "highlights must be headline-only"

        db.close()
        r.ok("temporal recall: date-ordered sessions + window floor + highlights")


def test_entity_graph(r):
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

def test_random_embedder(r):
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


def test_cached_embedder(r):
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


def test_ollama_embedder_handles_overlength_input(r):
    """Regression: a turn longer than the model's context window must NOT 400 the
    whole batch (which silently dropped distillation → sessions marked 'failed').
    Newer Ollama rejects over-length embed input with HTTP 400 instead of
    truncating; the embedder must cap inputs and recover per-item."""
    import io
    import json as _json
    import urllib.error
    from memory_system.embedder import OllamaEmbedder

    LIMIT = 20  # pretend the model context is tiny so we exercise the cap/halving

    class _Resp:
        def __init__(self, payload): self._p = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return _json.dumps(self._p).encode()

    class _FakeReq:
        def __init__(self): self.last_inputs = None
        def Request(self, url, data=None, headers=None, method=None):
            body = _json.loads(data.decode())
            self._inputs = body.get("input") or [body.get("prompt")]
            self.last_inputs = self._inputs
            return self
        def urlopen(self, req, timeout=None):
            # Reject if any single input is over the pretend context window.
            if any(len(t) > LIMIT for t in self._inputs):
                raise urllib.error.HTTPError(
                    "http://x/api/embed", 400, "Bad Request", {},
                    io.BytesIO(b'{"error":"the input length exceeds the context length"}'))
            return _Resp({"embeddings": [[0.1] * 8 for _ in self._inputs]})

    emb = OllamaEmbedder(model="nomic-embed-text", dim=8, max_chars=LIMIT)
    emb._req = _FakeReq()

    # A batch with one massively over-length turn must not raise, and must return
    # one vector per input (the long one truncated/halved down to fit).
    vecs = emb.embed_batch(["short", "x" * 500, "also short"])
    assert len(vecs) == 3, f"expected 3 vectors, got {len(vecs)}"
    assert all(len(v) == 8 for v in vecs), "every input must yield a vector"

    # And the primary path must cap before sending, never shipping the raw 500.
    big = "y" * 9000
    _ = emb.embed_batch([big])
    assert all(len(t) <= LIMIT for t in emb._req.last_inputs), \
        "embedder must cap input length to the context budget before sending"

    r.note(f"over-length recovered; cap={LIMIT}")
    r.ok("OllamaEmbedder caps and recovers over-length input without dropping the batch")


def test_ingest_scrubs_harness_scaffolding(r):
    """Regression: slash-command tags and skill bodies are harness plumbing, not
    the user's words. Left in, the distiller summarized them into junk headlines
    ("Use the /next command…", "Chief Engineer skill routes requests…"). The
    ingest scrub must drop pure-scaffolding turns and keep real content."""
    import hook_ingest  # repo root, already on sys.path

    # A bare slash-command invocation is all scaffolding → dropped.
    cmd_turn = "<command-message>next</command-message>\n<command-name>/next</command-name>"
    assert hook_ingest._scrub_scaffolding(cmd_turn) == "", \
        "command-tag-only turn must scrub to empty (skipped)"

    # A skill body expansion → dropped whole.
    skill_turn = ("Base directory for this skill: C:/Users/x/.claude/skills/next\n\n"
                  "# next — you judge, I drive\n\nChief Engineer skill routes requests...")
    assert hook_ingest._scrub_scaffolding(skill_turn) == "", \
        "skill-body turn must scrub to empty (skipped)"

    # A system-reminder embedded in a real turn → reminder gone, content kept.
    mixed = "Fix the redrive timeout.\n<system-reminder>memory is 4 days old</system-reminder>"
    cleaned = hook_ingest._scrub_scaffolding(mixed)
    assert cleaned == "Fix the redrive timeout.", f"got {cleaned!r}"

    # A genuine user turn passes through untouched.
    real = "Why is mem status crashing on long ops?"
    assert hook_ingest._scrub_scaffolding(real) == real, "real content must survive"

    r.note("command-tags + skill-body dropped; real turn + content preserved")
    r.ok("ingest scrub strips harness scaffolding, keeps real turns")


def test_vector_index(r):
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

def test_crs_and_tiers(r):
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

def test_fused_retrieval(r):
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


def test_crs_semantic_roundtrip(r):
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


def test_semantic_gate_roundtrip(r):
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


def test_db_concurrency(r):
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


def test_prompt_assembly(r):
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

def test_pipeline_with_mock_llm(r):
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


def test_pipeline_reflection(r):
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

def test_auditor_decay(r):
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


def test_auditor_pagerank(r):
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

def test_project_resolution(r):
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


def test_prompt_module(r):
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

def test_full_roundtrip(r):
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

def test_retrieval_k_ordering(r):
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


def test_structural_lane_no_crash(r):
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


def test_savings_token_accumulation(r):
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


def test_mcp_memory_save_signature(r):
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


def test_cache_breakpoint(r):
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


def test_cache_token_accumulation(r):
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


def test_failed_distillation_drops_fragment(r):
    """When the distillation model is unreachable, the pipeline must DROP the
    episode — not fall back to storing the raw transcript. A 2026-06-11 audit
    found 509 raw "User:/Assistant:" fragments (16% of live memory) from past
    API-budget outages, all stored by the old fallback and all ranked first by
    the temporal digest (episodes lead). A failed episode has no durable value;
    dropping it keeps memory clean. The LLM failure is surfaced upstream by
    llm.call_model's budget Issue-Catcher, so the drop is not silent."""
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
        assert not distilled, \
            f"failed distillation must store NO fragment, got {len(distilled)}: " \
            f"{[f.content[:60] for f in distilled]}"
        # And nothing raw leaked into the store either.
        raw = db.fetchall(
            "SELECT content FROM memory_fragments WHERE source_session='sess_dead'")
        assert not raw, f"no fragment should persist for a failed distill, found {len(raw)}"
        r.note("failed distillation produced 0 fragments (raw transcript dropped)")
        r.ok("Failed distillation drops the episode — no raw junk enters memory")
    finally:
        db.close()


# =============================================================================
#  RUN ALL TESTS
# =============================================================================

def test_goal_proposal_lifecycle(r):
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


def test_reindex_health_snapshot(r):
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


def test_upgrade_detector(r):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.upgrade import detect_upgrade, model_rank
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    # Ranking: NULL < unknown-equivalent < haiku < sonnet < claude-code < opus, version-aware.
    assert model_rank(None) < model_rank("claude-haiku-4-5"), "NULL must rank below any real model"
    assert model_rank("claude-haiku-4-5-20251001") == model_rank("claude-haiku-4-5"), \
        "dated id must resolve to its undated stem"
    assert model_rank("claude-opus-4-8") > model_rank("claude-sonnet-4-6") > model_rank("claude-haiku-4-5"), \
        "tier ordering wrong"
    assert model_rank("claude-opus-4-8") > model_rank("claude-opus-4-7"), "version ordering wrong"
    assert model_rank("claude-code") > model_rank("claude-sonnet-4-6"), \
        "claude-code subscription must rank above pinned sonnet-4-6"
    assert model_rank("claude-code") < model_rank("claude-opus-4-6"), \
        "claude-code subscription must rank below opus"
    r.note("model_rank: NULL < haiku < sonnet < claude-code < opus, version-aware, date-tolerant")

    def now():
        return datetime.now(tz=timezone.utc).isoformat()

    def frag(producer, conf=0.9, category="fact"):
        return MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category=category,
            content="x", token_count=5, confidence=conf, source_type="distillation",
            created_at=now(), last_accessed=now(),
            embedding_model="random", embedding_dim=128,
            producer_model=producer,
        )

    db = Database(scratch_dir() / "upgrade.db")
    db.connect()
    try:
        # A realistic mix mirroring the real store:
        #  - 2 weak low-confidence FACTS (re-synthesis can genuinely improve these)
        #  - 1 high-confidence fact (re-synth skips: only touches conf<0.70)
        #  - 3 raw episodes (never re-synthesized, must NOT be counted)
        db.upsert_fragment(frag(None,                conf=0.5))   # improvable
        db.upsert_fragment(frag("claude-haiku-4-5",  conf=0.5))   # improvable (behind sonnet)
        db.upsert_fragment(frag("claude-haiku-4-5",  conf=0.95))  # high-conf: not a candidate
        db.upsert_fragment(frag(None, conf=0.5, category="episode"))
        db.upsert_fragment(frag(None, conf=0.5, category="episode"))
        db.upsert_fragment(frag("claude-haiku-4-5", conf=0.5, category="episode"))

        # Re-synthesis would run with sonnet -> only the 2 weak low-conf FACTS
        # genuinely level up. fragments_behind still counts the raw producer gap
        # (5 of 6, incl. episodes) but that must NOT drive the nag anymore.
        up = detect_upgrade(db, "claude-haiku-4-5", project_id="p1",
                            resynthesis_model="claude-sonnet-4-6")
        assert up.upgrade_available is True, "weak facts should be improvable"
        assert up.improvable_facts == 2, f"improvable should be 2 facts, got {up.improvable_facts}"
        assert up.low_confidence_facts == 2, f"low-conf facts: {up.low_confidence_facts}"
        assert up.fragments_total == 6, f"total: {up.fragments_total}"
        r.note(f"improvable={up.improvable_facts} (NOT fragments_behind={up.fragments_behind}) -> honest")

        # THE BUG WE FIXED: facts already built by the re-synthesis model must
        # report NOTHING to gain — re-judging sonnet output with sonnet is a no-op,
        # even though raw episodes remain 'behind' forever.
        db2 = Database(scratch_dir() / "upgrade_current.db"); db2.connect()
        try:
            db2.upsert_fragment(frag("claude-sonnet-4-6", conf=0.5))   # already at strong
            db2.upsert_fragment(frag(None, conf=0.5, category="episode"))  # un-improvable
            cur = detect_upgrade(db2, "claude-haiku-4-5", project_id="p1",
                                 resynthesis_model="claude-sonnet-4-6")
            assert cur.improvable_facts == 0, f"improvable should be 0: {cur.improvable_facts}"
            assert cur.upgrade_available is False, "must not nag when facts already at re-synth model"
            assert "already current" in cur.note, f"note should say current: {cur.note}"
        finally:
            db2.close()
        r.note("no false 'could level up' nag when facts already at the re-synthesis model")

        r.ok("Upgrade detector counts only genuinely-improvable facts, not raw episodes")
    finally:
        db.close()


def test_budget_fallback(r):
    """When the paid API runs out of credit, LLM work must degrade to the local
    model — but transient failures (rate limit, 5xx) must NOT trigger a downgrade."""
    from memory_system import llm

    saved = (llm._call_anthropic, llm._call_ollama, llm._budget_exhausted_until,
             llm._BUDGET_FALLBACK_MODEL, llm._BUDGET_FALLBACK_ENDPOINT, llm._on_budget_fallback)
    try:
        llm._budget_exhausted_until = 0.0
        llm.set_budget_fallback("ollama/qwen3:8b", "http://localhost:11434")
        fired: list = []
        llm.register_budget_fallback_hook(lambda m, d: fired.append((m, d)))
        # Local model stub — proves we routed to ollama without a network call.
        llm._call_ollama = lambda prompt, system, model, base, mt, jm: f"LOCAL:{model}"

        # 1) A transient rate-limit error must propagate, NOT silently downgrade.
        class FakeRateLimit(Exception):
            type = "rate_limit_error"
        llm._call_anthropic = lambda *a, **k: (_ for _ in ()).throw(FakeRateLimit("429"))
        raised = False
        try:
            llm.call_model("hi", model="claude-haiku-4-5")
        except FakeRateLimit:
            raised = True
        assert raised, "transient rate-limit must propagate, not downgrade"
        assert not llm.budget_fallback_active(), "rate limit must not latch the fallback"
        r.note("transient rate-limit raises (no silent downgrade)")

        # 2) Out-of-credit (billing) → fall back to local, latch, fire alarm once.
        class FakeBilling(Exception):
            type = "billing_error"
            message = "Your credit balance is too low to access the API"
        llm._call_anthropic = lambda *a, **k: (_ for _ in ()).throw(FakeBilling())
        out = llm.call_model("hi", model="claude-sonnet-4-6")
        assert out == "LOCAL:qwen3:8b", f"should route to local fallback, got {out!r}"
        assert llm.budget_fallback_active(), "billing error must latch the fallback"
        assert fired and fired[0][0] == "claude-sonnet-4-6", "alarm hook should fire"
        r.note("out-of-credit → local fallback, latched, alarm fired")

        # 3) While latched, the paid path is skipped entirely (no second alarm).
        llm._call_anthropic = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("paid API must not be called while latched"))
        out2 = llm.call_model("hi", model="claude-sonnet-4-6")
        assert out2 == "LOCAL:qwen3:8b", "latched path must use local"
        assert len(fired) == 1, "alarm fires once per latch, not per call"

        # 4) The classifier must catch the REAL SDK exception shape, not just our
        # hand-made stub. A real out-of-credit error is a 400 BadRequestError
        # with NO usable top-level .type — recognition leans on the body text, so
        # assert against an actually-constructed SDK exception or this guard is
        # theatre (the stub above sets .type which the real error lacks).
        try:
            import anthropic as _a, httpx as _h
        except ImportError:
            _a = _h = None
        if _a is not None and _h is not None:
            _req = _h.Request("POST", "https://api.anthropic.com/v1/messages")
            def _mk(status, etype, message, cls):
                err = {"type": etype, "message": message}
                resp = _h.Response(status, request=_req,
                                   json={"type": "error", "error": err})
                return cls(f"Error code: {status}", response=resp, body=err)
            real_billing = _mk(400, "invalid_request_error",
                               "Your credit balance is too low to access the Anthropic API.",
                               _a.BadRequestError)
            real_auth = _mk(401, "authentication_error", "invalid x-api-key",
                            _a.AuthenticationError)
            real_overload = _mk(529, "overloaded_error", "Overloaded",
                                _a.InternalServerError)
            assert llm._is_budget_or_auth_error(real_billing), \
                "real out-of-credit BadRequestError must be caught"
            assert llm._is_budget_or_auth_error(real_auth), \
                "real 401 auth error must be caught"
            assert not llm._is_budget_or_auth_error(real_overload), \
                "transient 529 overload must NOT latch the fallback"
            r.note("classifier verified against real SDK BadRequestError/401/529")

            # Hardened recognition (2026-06-07): must survive wording changes.
            # (a) 402 Payment Required with NO billing phrase → caught by STATUS.
            real_402 = _mk(402, "request_error", "Account action needed.",
                           _a.APIStatusError)
            assert llm._is_budget_or_auth_error(real_402), \
                "402 Payment Required must be caught by status, regardless of wording"
            # (b) 429 rate limit whose prose mentions a 'limit' must NOT degrade
            # us (transient excluded by type before any phrase match).
            real_429 = _mk(429, "rate_limit_error",
                           "You have exceeded your usage limit", _a.RateLimitError)
            assert not llm._is_budget_or_auth_error(real_429), \
                "429 rate limit must never latch the fallback"
            assert not llm._is_unrecognized_paid_rejection(real_429), \
                "429 is transient — not an unrecognized rejection"
            # (c) a REWORDED out-of-credit 400 that matches NO known phrase: we
            # can't classify it as budget, but it must raise the 'can't be sure'
            # alarm so it never stalls silently.
            real_reworded = _mk(400, "invalid_request_error",
                                "Your account is no longer permitted to spend.",
                                _a.BadRequestError)
            assert not llm._is_budget_or_auth_error(real_reworded), \
                "unmatched wording is (correctly) not auto-classified as budget"
            assert llm._is_unrecognized_paid_rejection(real_reworded), \
                "a persistent unrecognized 4xx must trip the uncertainty alarm"
            assert not llm._is_unrecognized_paid_rejection(real_billing), \
                "a recognized billing 400 is handled, not flagged uncertain"
            # (d) Anthropic's quota/usage-limit wording (2026-06-10): must be
            # caught as out-of-credit, NOT fire the 'unrecognized' alarm.
            real_usage_limit = _mk(400, "invalid_request_error",
                                   "You have reached your specified API usage limits. "
                                   "You will regain access on 2026-07-01 at 00:00 UTC.",
                                   _a.BadRequestError)
            assert llm._is_budget_or_auth_error(real_usage_limit), \
                "Anthropic usage-limit 400 must be caught as out-of-credit"
            assert not llm._is_unrecognized_paid_rejection(real_usage_limit), \
                "recognized usage-limit must not trip the uncertainty alarm"
            r.note("hardened: Anthropic 'usage limits' phrase caught as out-of-credit")
            # And the alarm actually fires through the registered hook (once).
            alarms = []
            _saved_hook = llm._on_paid_error
            llm._paid_error_alarm_until = 0.0
            llm._budget_exhausted_until = 0.0  # clear the latch from step 2 so the paid path runs
            llm.register_paid_error_hook(lambda m, d: alarms.append((m, d)))
            try:
                llm._call_anthropic = lambda *a, **k: (_ for _ in ()).throw(real_reworded)
                try:
                    llm.call_model("hi", model="claude-sonnet-4-6")
                except _a.BadRequestError:
                    pass
                assert len(alarms) == 1 and alarms[0][0] == "claude-sonnet-4-6", \
                    "unrecognized paid rejection must fire the alarm and re-raise"
            finally:
                llm.register_paid_error_hook(_saved_hook)
            r.note("hardened: 402-by-status, 429 immune, reworded-400 alarms not stalls")

        r.ok("Budget fallback: degrades to local on credit exhaustion, ignores transient errors")
    finally:
        (llm._call_anthropic, llm._call_ollama, llm._budget_exhausted_until,
         llm._BUDGET_FALLBACK_MODEL, llm._BUDGET_FALLBACK_ENDPOINT, llm._on_budget_fallback) = saved


def test_budget_fallback_cli(r):
    """When fallback is 'claude-code', out-of-credit routes through the Claude
    Code CLI (subscription); if the CLI is missing or the runaway circuit-breaker
    trips, it degrades to the local model."""
    import time as _t
    from memory_system import llm

    saved = (llm._call_anthropic, llm._call_ollama, llm._call_claude_cli,
             llm._budget_exhausted_until, llm._BUDGET_FALLBACK_MODEL,
             llm._BUDGET_FALLBACK_LOCAL_MODEL, llm._on_budget_fallback,
             list(llm._cli_call_times))
    try:
        llm._budget_exhausted_until = 0.0
        llm._cli_call_times.clear()
        llm.set_budget_fallback("claude-code", local_model="ollama/qwen3:8b")
        llm.register_budget_fallback_hook(lambda m, d: None)
        llm._call_ollama = lambda *a, **k: "LOCAL"

        class FakeBilling(Exception):
            type = "billing_error"
            message = "credit balance too low"
        llm._call_anthropic = lambda *a, **k: (_ for _ in ()).throw(FakeBilling())

        # CLI reachable → subscription path is used.
        llm._call_claude_cli = lambda prompt, system, mt, timeout=180: "CLI:ok"
        out = llm.call_model("hi", model="claude-sonnet-4-6")
        assert out == "CLI:ok", f"should route through claude-code CLI, got {out!r}"
        r.note("out-of-credit → Claude Code CLI (subscription) path")

        # CLI not installed → degrade to local.
        llm._call_claude_cli = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("claude"))
        out2 = llm._budget_fallback_call("hi", "", 100, False)
        assert out2 == "LOCAL", f"missing CLI must degrade to local, got {out2!r}"
        r.note("claude CLI missing → degrades to local")

        # Runaway circuit-breaker tripped → degrade to local even though CLI 'works'.
        llm._call_claude_cli = lambda *a, **k: "CLI:ok"
        llm._cli_call_times[:] = [_t.monotonic()] * llm._CLI_MAX_IN_WINDOW
        out3 = llm._budget_fallback_call("hi", "", 100, False)
        assert out3 == "LOCAL", "tripped circuit-breaker must degrade to local"
        r.ok("Budget fallback: prefers subscription CLI, degrades to local on missing CLI / runaway")
    finally:
        (llm._call_anthropic, llm._call_ollama, llm._call_claude_cli,
         llm._budget_exhausted_until, llm._BUDGET_FALLBACK_MODEL,
         llm._BUDGET_FALLBACK_LOCAL_MODEL, llm._on_budget_fallback, _times) = saved
        llm._cli_call_times[:] = _times


def test_claude_code_primary_route(r):
    """model='claude-code' routes through the CLI directly — never touches the
    paid Anthropic SDK, even when the budget latch is clear."""
    import time as _t
    from memory_system import llm

    saved = (llm._call_anthropic, llm._call_ollama, llm._call_claude_cli,
             llm._budget_exhausted_until, list(llm._cli_call_times))
    try:
        llm._budget_exhausted_until = 0.0
        llm._cli_call_times.clear()
        llm._call_anthropic = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("paid API must not be called for model=claude-code"))
        llm._call_ollama = lambda *a, **k: f"LOCAL:{a[2]}"

        # CLI available → subscription path used.
        llm._call_claude_cli = lambda prompt, system, mt, timeout=180: "CLI:ok"
        out = llm.call_model("hi", model="claude-code")
        assert out == "CLI:ok", f"should route through CLI, got {out!r}"
        r.note("model=claude-code routes through CLI, never paid SDK")

        # CLI missing → degrade to local (not an error).
        llm._call_claude_cli = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("claude"))
        out2 = llm.call_model("hi", model="claude-code")
        assert out2.startswith("LOCAL:"), f"missing CLI must degrade to local, got {out2!r}"
        r.note("model=claude-code + missing CLI → local fallback, no crash")

        # Runaway guard tripped → also degrades to local.
        llm._call_claude_cli = lambda *a, **k: "CLI:ok"
        llm._cli_call_times[:] = [_t.monotonic()] * llm._CLI_MAX_IN_WINDOW
        out3 = llm.call_model("hi", model="claude-code")
        assert out3.startswith("LOCAL:"), "tripped circuit must degrade to local"
        r.ok("claude-code primary route: CLI → local gracefully, paid SDK never called")
    finally:
        (llm._call_anthropic, llm._call_ollama, llm._call_claude_cli,
         llm._budget_exhausted_until, _times) = saved
        llm._cli_call_times[:] = _times


def test_resynthesis_uses_router(r):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.reindex import ModelUpgradeReindexJob
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()

    # A fake router that stands in for a reachable model. Resynthesis must route
    # through the 'strong' role — the point of a model upgrade is to re-judge
    # memory with the sharpest reachable model, not a local no-op (commit f3501ab).
    class FakeRouter:
        def __init__(self): self.calls = 0
        def model_for(self, role): return "ollama/qwen3:8b"
        def call_json(self, role, prompt, max_tokens=200):
            self.calls += 1
            assert role == "strong", f"resynthesis must use 'strong', got {role!r}"
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
        job._resynthesis_role = "strong"  # ditto — init never ran

        n = job._resynthesize_facts("p1")
        assert n == 1, f"exactly one low-conf fact should be rewritten, got {n}"
        assert fake.calls == 1, "router must be the path used (not the cloud fallback)"

        got = db.get_fragment(low.id)
        assert got.content == "Crisp, de-vagued restatement of the fact.", "content not rewritten"
        assert abs(got.confidence - 0.50) < 1e-6, \
            f"rewriting must NOT change confidence (stays 0.50), got {got.confidence}"
        assert got.producer_model == "ollama/qwen3:8b", \
            f"rewritten fact must be stamped with the producer, got {got.producer_model!r}"
        r.note("low-conf fact rewritten via 'strong' role, provenance stamped, confidence untouched")

        untouched = db.get_fragment(high.id)
        assert untouched.confidence == 0.95 and untouched.producer_model is None, \
            "high-confidence fact must be left alone"
        r.note("high-confidence fact left untouched")

        r.ok("Resynthesis uses the strong router and stamps provenance off NULL")
    finally:
        db.close()


def test_resynthesis_never_inflates_confidence(r):
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
        job._resynthesis_role = "strong"  # built via __new__, so init never set this

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


def test_reindex_progress_callback(r):
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


def test_eval_scoring(r):
    from memory_system.eval import EvalCase, score, parse_cases

    # Three cases: hit at rank 1, hit at rank 3, and a miss.
    frags_a = [{"id": "f1", "content": "Docker builds need a .dockerignore"},
               {"id": "f2", "content": "CI runs pytest"}]
    frags_b = [{"id": "x1", "content": "irrelevant"},
               {"id": "x2", "content": "also irrelevant"},
               {"id": "x3", "content": "the durability invariant is load-bearing"}]
    frags_c = [{"id": "z1", "content": "nothing matches here"}]

    case_outputs = [
        (EvalCase(query="docker", expect=["dockerignore"]), frags_a),   # rank 1
        (EvalCase(query="invariant", expect=["durability invariant"]), frags_b),  # rank 3
        (EvalCase(query="missing", expect=["never appears"]), frags_c),  # miss
    ]
    report = score(case_outputs, k_values=(1, 3, 5))

    assert report.total == 3
    assert report.matched == 2, f"expected 2 matched, got {report.matched}"
    assert report.results[0].rank == 1, report.results[0].rank
    assert report.results[1].rank == 3, report.results[1].rank
    assert report.results[2].rank is None
    # recall@1 = only the first case hits at 1 -> 1/3
    assert abs(report.recall_at(1) - 1/3) < 1e-9, report.recall_at(1)
    # recall@3 = first two hit within 3 -> 2/3
    assert abs(report.recall_at(3) - 2/3) < 1e-9, report.recall_at(3)
    # MRR = (1/1 + 1/3 + 0) / 3
    assert abs(report.mrr - (1.0 + 1/3) / 3) < 1e-9, report.mrr
    r.note(f"recall@3={report.recall_at(3):.2f}, MRR={report.mrr:.3f}")

    # expect_id matching + any-of substrings + malformed-file rejection
    by_id = score([(EvalCase(query="q", expect_id=["f2"]), frags_a)], (1, 3))
    assert by_id.results[0].rank == 2, by_id.results[0].rank
    try:
        parse_cases({"cases": [{"query": "no expectation"}]})
        assert False, "should have rejected a case with no expect/expect_id"
    except ValueError:
        pass
    r.ok("Rank, recall@k, MRR, id-match and validation all correct")


def test_read_only_retrieval(r):
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment
    from memory_system.vector_index import VectorIndex
    from memory_system.embedder import RandomEmbedder
    from memory_system.retrieval import fused_retrieval
    from memory_system.config import RetrievalConfig
    from memory_system.ids import new_id

    emb = RandomEmbedder(dim=128)
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_ro.db")
        db.connect()
        idx = VectorIndex(dim=128, persist_path=str(Path(tmpdir) / "ro.hnsw"))
        idx.init_fresh()

        now = datetime.now(tz=timezone.utc).isoformat()
        fid = new_id()
        vec = emb.embed("the durability invariant separates record from judgment")
        db.upsert_fragment(MemoryFragment(
            id=fid, project_id="p1", scope="project", category="fact",
            content="the durability invariant separates record from judgment",
            token_count=10, confidence=0.8, created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128, embedding=vec,
        ))
        idx.add(fid, vec)

        cfg = RetrievalConfig(default_token_budget=500, max_fragments=5, min_crs=0.0)
        q_emb = emb.embed("durability invariant")

        def _event_count() -> int:
            return db.fetchone("SELECT COUNT(*) AS n FROM retrieval_events")["n"]

        # read_only=True: must surface the fragment but touch nothing.
        res = fused_retrieval(
            db=db, vector_index=idx, query_embedding=q_emb,
            query_text="durability invariant", project_id="p1",
            token_budget=500, cfg=cfg, read_only=True,
        )
        assert len(res.fragments) >= 1, "read_only retrieval still returns results"
        assert db.get_fragment(fid).access_count == 0, "read_only must NOT touch access_count"
        assert _event_count() == 0, "read_only must NOT log a retrieval event"
        r.note("read_only: 1 result, access_count=0, events=0")

        # Default (read_only=False): the side-effects DO happen.
        fused_retrieval(
            db=db, vector_index=idx, query_embedding=q_emb,
            query_text="durability invariant", project_id="p1",
            token_budget=500, cfg=cfg,
        )
        assert db.get_fragment(fid).access_count == 1, "normal retrieval should touch"
        assert _event_count() == 1, "normal retrieval should log one event"
        r.note("normal: access_count=1, events=1")

        db.close()
        r.ok("read_only suppresses both side-effects; normal path keeps them")


def test_read_reflex_hook(r):
    import hook_read  # lives at repo root; _project_root is on sys.path

    # Gating: acks, slash-commands, shell escapes and tiny prompts are skipped;
    # a real question is not.
    assert hook_read._should_skip("ok") is True
    assert hook_read._should_skip("Thanks!") is True
    assert hook_read._should_skip("/review") is True
    assert hook_read._should_skip("!ls -la") is True
    assert hook_read._should_skip("hi") is True
    assert hook_read._should_skip("why is retrieval ranking the goal fragment so low") is False
    r.note("skip set, slash/shell escapes, length gate all correct")

    # Formatting: a clearly-labelled block carrying crs + id + content, and the
    # caller-side cap is respected.
    frags = [
        {"id": "frag_a", "crs": 0.82, "content": "the   durability\n invariant"},
        {"id": "frag_b", "crs": 0.55, "content": "second fragment"},
    ]
    goals = ["Build a memory system that knows me"]
    block = hook_read._format_block(goals, frags, [])
    assert block.startswith("<recalled_memory"), block[:40]
    assert block.rstrip().endswith("</recalled_memory>")
    assert "frag_a" in block and "0.82" in block
    assert "durability invariant" in block, "whitespace should be collapsed"
    # Intent steering: open goals render first, before the recalled fragments.
    assert "Build a memory system that knows me" in block, "goals must be injected"
    assert block.index("knows me") < block.index("frag_a"), "goals come before fragments"
    assert hook_read._MAX_FRAGMENTS >= 1
    r.note("block carries goals first, then id+crs-tagged fragments, ws-collapsed")

    # Temporal lane: 'what did I work on' prompts are detected and route to a
    # date-walk, not a vector search; ordinary questions do not.
    assert hook_read._is_temporal_query("what did I work on last week") is True
    assert hook_read._is_temporal_query("remind me where we were") is True
    assert hook_read._is_temporal_query("what have I been doing recently") is True
    assert hook_read._is_temporal_query("how does the reranker weight citations") is False
    # The recent digest renders as its own labelled section, between goals and
    # topical fragments, newest-first.
    recent = ["- 2026-06-10: shipped the temporal dashboard panel",
              "- 2026-06-09: built mem recent"]
    tblock = hook_read._format_block(goals, frags, recent)
    assert "worked on recently" in tblock, "temporal section must be labelled"
    assert "2026-06-10" in tblock
    assert tblock.index("knows me") < tblock.index("2026-06-10") < tblock.index("frag_a"), \
        "order must be goals -> recent -> fragments"
    r.note("temporal query detected; recent digest renders between goals and fragments")
    r.ok("Read reflex gating + formatting + temporal lane verified")


def test_citation_detector(r):
    from memory_system.citation import detect_cited, content_tokens

    injected = {
        "frag_used": "The durability invariant separates the durable Record from the "
                     "disposable Judgment so a fast layer never destroys a slow layer input.",
        "frag_topic_only": "Docker multi-stage builds produce smaller production images.",
        "frag_echo": "The user prefers concise outcome-focused summaries.",
    }
    prompt = "remind me about the user preference for concise summaries"
    # Answer draws on frag_used (novel content), not on the others.
    response = ("The durability invariant keeps the durable Record separate from the "
                "disposable Judgment, so a fast layer never destroys the slow layer's input.")

    cited = detect_cited(injected, prompt, response)
    assert "frag_used" in cited, f"grounded fragment should be cited: {cited}"
    assert "frag_topic_only" not in cited, "unused off-topic fragment must not be cited"
    # frag_echo's distinctive words ('concise','summaries','preference') are all in
    # the PROMPT, so even if echoed they must not count as a citation.
    assert "frag_echo" not in cited, "prompt-supplied words must not earn a citation"
    r.note(f"cited={cited}")

    # min_novel floor: a single shared word is coincidence, not grounding.
    weak = detect_cited({"f": "alpha beta gamma delta epsilon zeta"},
                        prompt="", response="alpha", min_novel=4)
    assert weak == [], "a lone shared token must not be a citation"
    assert "durability" in content_tokens("The Durability Invariant")
    r.ok("Citation detection grounded, prompt-subtracted, and floor-gated")


def test_citation_feeds_v4_label(r):
    import json as _json
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment, RetrievalEvent
    from memory_system.ids import new_id
    from memory_system.v4_reranker import _fetch_training_rows, RerankerConfig, COMPONENTS

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_cite.db")
        db.connect()
        now = datetime.now(tz=timezone.utc).isoformat()

        fid = new_id()
        db.upsert_fragment(MemoryFragment(
            id=fid, project_id="p1", scope="project", category="fact",
            content="cited fact", token_count=3, confidence=0.8,
            created_at=now, last_accessed=now,
            embedding_model="random", embedding_dim=128,
        ))

        # A retrieval event that surfaced this fragment, with full CRS components.
        comps = {c: 0.5 for c in COMPONENTS}
        db.log_retrieval_event(RetrievalEvent(
            query_hash="qh", project_id="p1",
            fragment_ids_json=_json.dumps([fid]),
            crs_components_json=_json.dumps({fid: comps}),
            returned_at=now,
        ))

        cfg = RerankerConfig()
        # Before any citation: not re-touched, not pinned, no feedback -> label 0.
        data0 = _fetch_training_rows(db._conn, "p1", cfg)
        assert len(data0.y) == 1, f"expected 1 training row, got {len(data0.y)}"
        assert data0.y[0] == 0, "uncited fragment should be a negative example"

        # Mark it cited (what the Stop hook's _report_citations triggers).
        n = db.mark_cited([fid], now)
        assert n == 1, f"mark_cited should update 1 row, got {n}"
        got = db.get_fragment(fid)
        assert got.times_cited == 1 if hasattr(got, "times_cited") else True

        # After citation: label flips to 1 (the load-bearing signal).
        data1 = _fetch_training_rows(db._conn, "p1", cfg)
        assert data1.y[0] == 1, "cited fragment must become a positive example"
        r.note("uncited y=0 -> cited y=1; mark_cited updated 1 row")

        db.close()
        r.ok("Citation persists and v4's useful label learns from it")


def test_hard_negatives_feed_v4_label(r):
    """Candidates scored-but-not-surfaced become guaranteed label-0 rows, even
    if the fragment is cited elsewhere — the components are query-specific, so
    'rejected for THIS query' is the truth we train on. This is the signal that
    retrieved-only negatives can't give (it breaks frequency dominance)."""
    import json as _json
    from memory_system.schema import Database
    from memory_system.models import MemoryFragment, RetrievalEvent
    from memory_system.ids import new_id
    from memory_system.v4_reranker import _fetch_training_rows, RerankerConfig, COMPONENTS

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test_hardneg.db")
        db.connect()
        now = datetime.now(tz=timezone.utc).isoformat()

        surfaced, rejected = new_id(), new_id()
        for fid in (surfaced, rejected):
            db.upsert_fragment(MemoryFragment(
                id=fid, project_id="p1", scope="project", category="fact",
                content=f"frag {fid}", token_count=3, confidence=0.8,
                created_at=now, last_accessed=now,
                embedding_model="random", embedding_dim=128,
            ))
        # Both fragments are cited (popular) — yet `rejected` was a hard negative
        # for THIS query, so it must still train as a 0.
        db.mark_cited([surfaced, rejected], now)

        comps = {c: 0.5 for c in COMPONENTS}
        db.log_retrieval_event(RetrievalEvent(
            query_hash="qh", project_id="p1",
            fragment_ids_json=_json.dumps([surfaced]),
            crs_components_json=_json.dumps({surfaced: comps}),
            returned_at=now,
            rejected_components_json=_json.dumps({rejected: comps}),
        ))

        cfg = RerankerConfig()
        data = _fetch_training_rows(db._conn, "p1", cfg)
        assert len(data.y) == 2, f"expected 2 rows (1 surfaced + 1 rejected), got {len(data.y)}"
        # Map rows back: the surfaced+cited row is the positive; the hard negative is 0.
        labels = {int(yi): float(swi) for yi, swi in zip(data.y, data.sw)}
        assert 1 in labels, "surfaced+cited fragment should yield a positive (y=1)"
        assert 0 in labels, "rejected candidate must yield a hard negative (y=0) despite being cited"
        # The hard negative carries the full hard_negative_weight, not the weak re-touch weight.
        assert labels[0] == cfg.hard_negative_weight, (
            f"hard-negative weight should be {cfg.hard_negative_weight}, got {labels[0]}")
        r.note(f"surfaced->y=1, rejected->y=0 @ sw={cfg.hard_negative_weight} (forced despite citation)")

        db.close()
        r.ok("Hard negatives (scored-but-rejected) feed v4 as guaranteed label-0 rows")


def test_extract_json_object_contract(r):
    """_extract_json must honor its documented contract: always return a dict.
    Regression for the 2026-06-11 ingest crash 'list object has no attribute
    get' — a model emitted a bare JSON array, the old code returned the list,
    and the distill loop's data.get('summary') threw far from here (pipeline.py
    line outside the try/except, so it crashed the whole ingest instead of
    dropping one episode)."""
    from memory_system.llm import _extract_json

    # Plain object: returned as-is.
    assert _extract_json('{"summary": "ok", "confidence": 0.7}') == \
        {"summary": "ok", "confidence": 0.7}
    # Object buried in prose + code fence: still extracted.
    assert _extract_json('Sure!\n```json\n{"a": 1}\n```')["a"] == 1
    # Thinking-model wrapper stripped, then object extracted.
    assert _extract_json('<think>weighing it</think>{"b": 2}')["b"] == 2
    r.note("dict / fenced / think-wrapped objects all parse to a dict")

    # The bug shape: a bare array must NOT return a list (which crashes .get()).
    for bad in ('["new_facts", "lessons"]', '[1, 2, 3]', '[]', '"just a string"', '42'):
        try:
            out = _extract_json(bad)
            assert False, f"non-object {bad!r} must raise, returned {out!r} ({type(out).__name__})"
        except ValueError:
            pass
    r.note("bare arrays / scalars raise ValueError (caller drops the episode)")

    # Salvage the common `[{...}]` singleton-wrap slip into the object inside.
    assert _extract_json('[{"summary": "wrapped"}]') == {"summary": "wrapped"}
    r.note("single-object array [{...}] is unwrapped, not rejected")

    r.ok("_extract_json always yields a dict — no list ever reaches a caller's .get()")


def _dead_router():
    """A router that names a capable model but FAILS every distillation call —
    the 'configured cloud model, no key / Ollama down' outage on this box."""
    class DeadRouter:
        def model_for(self, role): return "claude-haiku-4-5-20251001"
        def call_json(self, role, prompt, system="", max_tokens=200):
            raise ConnectionError("simulated: distillation model unreachable")
    return DeadRouter()


def _live_router():
    """A router that distills successfully — stands in for the embedder/LLM
    having recovered, so the redrive can rebuild the lost Judgment."""
    class LiveRouter:
        def model_for(self, role): return "claude-haiku-4-5-20251001"
        def call_json(self, role, prompt, system="", max_tokens=200):
            return {
                "recall": "Recovered: fixed the Docker build by adding a .dockerignore",
                "intent": "fix the Docker build", "outcome": "build passes",
                "summary": "Added a .dockerignore to shrink the Docker build context.",
                "confidence": 0.8, "abstraction_level": 0.4,
                "key_decisions": [], "tools_used": [], "files_modified": [],
                "errors_encountered": [], "entities": [], "relations": [],
            }
    return LiveRouter()


def test_pipeline_reports_distill_failures(r):
    """The redrive's detection signal: when distillation throws for every
    episode, pipeline.ingest must report distill_failures > 0 alongside ZERO
    fragments — that pair is exactly how the ingest worker tells a genuine outage
    hole apart from a session that simply had nothing worth keeping."""
    from memory_system.schema import Database
    from memory_system.vector_index import VectorIndex
    from memory_system.config import CompressionConfig
    from memory_system.pipeline import ConsolidationPipeline, TranscriptMessage
    from memory_system.embedder import RandomEmbedder
    from memory_system.models import Session

    tmpdir = scratch_dir()
    db = Database(tmpdir / "redrive_signal.db"); db.connect()
    try:
        idx = VectorIndex(dim=128, persist_path=str(tmpdir / "rs.hnsw")); idx.init_fresh()
        emb = RandomEmbedder(dim=128)
        cfg = CompressionConfig(consolidation_threshold=3, max_episode_tokens=500)
        db.upsert_session(Session(id="sess_sig", project_id="p1", turn_count=3))
        pipe = ConsolidationPipeline(db, idx, cfg, embedder=emb, router=_dead_router())
        messages = [
            TranscriptMessage(role="user", content="Fix the Docker build, context too large"),
            TranscriptMessage(role="assistant", content="Missing a .dockerignore — creating one."),
            TranscriptMessage(role="user", content="Great, that fixed it!"),
        ]
        stats: dict = {}
        frags = pipe.ingest(messages, session_id="sess_sig", project_id="p1", stats_out=stats)
        assert not frags, f"a failed distill must produce 0 fragments, got {len(frags)}"
        assert stats.get("distill_failures", 0) >= 1, \
            f"distill_failures must be reported, got {stats!r}"
        r.note(f"0 fragments + distill_failures={stats['distill_failures']} → outage hole detected")
        r.ok("pipeline.ingest reports distill_failures so the worker can mark the session failed")
    finally:
        db.close()


def test_redrive_bookkeeping(r):
    """The session-level redrive ledger: set_distill_status / sessions_needing_redrive
    / count_redrive_pending, and the bounded-retry cap that stops a poison session
    from looping forever."""
    from memory_system.schema import Database
    from memory_system.models import Session

    db = Database(scratch_dir() / "redrive_book.db"); db.connect()
    try:
        db.upsert_session(Session(id="s_fail", project_id="p1", turn_count=9))
        # Healthy by default — nothing pending.
        assert db.count_redrive_pending() == 0
        # An outage marks it failed → it enters the redrive queue.
        db.set_distill_status("s_fail", "failed")
        assert db.count_redrive_pending() == 1
        pend = db.sessions_needing_redrive(max_attempts=5)
        assert [p["id"] for p in pend] == ["s_fail"], pend
        # Burn the retry cap — it ages out of the queue (no infinite loop).
        for _ in range(5):
            db.set_distill_status("s_fail", "failed", bump_attempt=True)
        assert db.count_redrive_pending(max_attempts=5) == 0, "capped session must leave the queue"
        r.note("failed→queued; 5 bumped attempts→aged out (bounded retries)")
        # Recovery flips it back to 'ok'.
        db.upsert_session(Session(id="s_recov", project_id="p1", turn_count=4))
        db.set_distill_status("s_recov", "failed")
        assert db.count_redrive_pending() == 1
        db.set_distill_status("s_recov", "ok")
        assert db.count_redrive_pending() == 0, "a recovered session must leave the queue"
        r.ok("redrive ledger: mark failed, bounded retries, recovery clears it")
    finally:
        db.close()


def test_redrive_backfill_scan(r):
    """The historical-recovery path: mark_zero_fragment_sessions_failed targets
    ONLY sessions that produced zero fragments and had real content — never one
    that already has fragments (no double-distill), never a trivially short one."""
    from memory_system.schema import Database
    from memory_system.models import Session, MemoryFragment
    from memory_system.ids import new_id
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()
    db = Database(scratch_dir() / "redrive_scan.db"); db.connect()
    try:
        # s_good: distilled fine — has a fragment → must stay 'ok'.
        db.upsert_session(Session(id="s_good", project_id="p1", turn_count=8))
        db.upsert_fragment(MemoryFragment(
            id=new_id(), project_id="p1", scope="project", category="episode",
            content="did a thing", token_count=3, confidence=0.8,
            source_type="distillation", source_session="s_good",
            created_at=now, last_accessed=now, embedding_model="random", embedding_dim=128))
        # s_hole: real content, zero fragments → the outage hole the scan recovers.
        db.upsert_session(Session(id="s_hole", project_id="p1", turn_count=8))
        # s_tiny: below the turn threshold → ignored (probably nothing to distill).
        db.upsert_session(Session(id="s_tiny", project_id="p1", turn_count=1))

        marked = db.mark_zero_fragment_sessions_failed(min_turns=3)
        assert marked == 1, f"only the content-bearing zero-fragment session should mark, got {marked}"
        pend_ids = {p["id"] for p in db.sessions_needing_redrive()}
        assert pend_ids == {"s_hole"}, f"scan must target only s_hole, got {pend_ids}"
        r.ok("backfill scan marks only real zero-fragment holes (spares distilled + trivial sessions)")
    finally:
        db.close()


def test_redrive_recovers_session_from_cold_storage(r):
    """End-to-end proof of the whole point: an outage drops a session's
    distillation (0 fragments), but because the raw transcript is safe in cold
    storage, re-distilling from cold storage once the LLM recovers rebuilds the
    lost memory — turning a permanent hole into a deferred one."""
    from memory_system.schema import Database
    from memory_system.vector_index import VectorIndex
    from memory_system.config import CompressionConfig
    from memory_system.pipeline import ConsolidationPipeline, TranscriptMessage
    from memory_system.embedder import RandomEmbedder
    from memory_system.models import Session
    from memory_system.cold_storage import append_session, read_session_from_path

    tmpdir = scratch_dir()
    cold_root = tmpdir / "cold"
    db = Database(tmpdir / "redrive_e2e.db"); db.connect()
    try:
        idx = VectorIndex(dim=128, persist_path=str(tmpdir / "e2e.hnsw")); idx.init_fresh()
        emb = RandomEmbedder(dim=128)
        cfg = CompressionConfig(consolidation_threshold=3, max_episode_tokens=500)
        segments = [
            {"role": "user", "content": "Fix the Docker build, context too large"},
            {"role": "assistant", "content": "Missing a .dockerignore — creating one."},
            {"role": "user", "content": "Great, that fixed it!"},
        ]
        # The durable Record is archived before ACK (as the daemon does).
        append_session(cold_root, "p1", "sess_e2e", segments)
        db.upsert_session(Session(id="sess_e2e", project_id="p1", turn_count=3))

        # 1) Outage: distillation fails → 0 fragments. The ingest worker marks it.
        msgs = [TranscriptMessage(role=s["role"], content=s["content"]) for s in segments]
        stats: dict = {}
        dead = ConsolidationPipeline(db, idx, cfg, embedder=emb, router=_dead_router())
        frags = dead.ingest(msgs, session_id="sess_e2e", project_id="p1", stats_out=stats)
        assert not frags and stats.get("distill_failures", 0) >= 1
        db.set_distill_status("sess_e2e", "failed")
        assert db.count_redrive_pending() == 1, "the lost session must be queued for redrive"

        # 2) Redrive: read the raw Record back from cold storage and re-distill
        #    with a recovered LLM (mirrors daemon._redrive_session). The cold file
        #    lives under cold_root/p1/<YYYY-MM>/sess_e2e.jsonl.zst.
        cold_file = next(cold_root.glob("p1/*/sess_e2e.jsonl.zst"))
        meta, segs = read_session_from_path(cold_file)
        replay = [TranscriptMessage(role=s.get("role", "user"), content=s.get("content", ""))
                  for s in segs if s.get("content")]
        assert replay, "cold storage must round-trip the raw transcript"
        live = ConsolidationPipeline(db, idx, cfg, embedder=emb, router=_live_router())
        recovered = live.ingest(replay, session_id="sess_e2e", project_id="p1")
        assert recovered, "redrive from cold storage must rebuild the lost memory"
        db.set_distill_status("sess_e2e", "ok")

        # 3) The hole is healed: no longer pending, and a real fragment now exists.
        assert db.count_redrive_pending() == 0
        stored = db.fetchall(
            "SELECT id FROM memory_fragments WHERE source_session='sess_e2e' AND is_deprecated=0")
        assert stored, "a distillation fragment must now exist for the recovered session"
        r.note(f"outage→0 frags→queued; cold replay→{len(recovered)} frag(s)→queue cleared")
        r.ok("Redrive rebuilds an outage-lost session from cold storage — a deferred hole, not a permanent one")
    finally:
        db.close()


def main() -> int:
    """Backwards-compatible entry point. There is now ONE source of truth for
    pass/fail — pytest — so `python -m memory_system.test_smoke` just delegates
    to it and returns its exit code."""
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    sys.exit(main())
