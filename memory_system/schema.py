"""
SQLite schema creation and low-level DB wrapper.
All queries are parameterized. No ORM — raw SQL for predictable performance.
"""

import sqlite3
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any
from .models import (
    MemoryFragment, Session, Entity, Triple, Correction,
    StructuralPattern, PendingPattern, Simulation, EpistemicEvent,
    RetrievalEvent, Event, Goal,
)


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memory_fragments (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    scope           TEXT NOT NULL CHECK(scope IN ('project','global','session')),
    category        TEXT NOT NULL CHECK(category IN ('fact','episode','preference','correction')),
    content         TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    abstraction_lvl REAL    DEFAULT 0.5,
    confidence      REAL    DEFAULT 0.8,
    source_type     TEXT,
    source_session  TEXT    REFERENCES sessions(id),
    created_at      TEXT    NOT NULL,
    last_accessed   TEXT    NOT NULL,
    access_count    INTEGER DEFAULT 0,
    is_pinned       INTEGER DEFAULT 0,
    is_deprecated   INTEGER DEFAULT 0,
    deprecated_by   TEXT    REFERENCES memory_fragments(id),
    graph_centrality REAL   DEFAULT 0.0,
    user_feedback   REAL    DEFAULT 0.0,
    embedding_model TEXT    NOT NULL DEFAULT '',
    embedding_dim   INTEGER NOT NULL DEFAULT 0,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_frag_project_scope
    ON memory_fragments(project_id, scope);
CREATE INDEX IF NOT EXISTS idx_frag_category
    ON memory_fragments(category);
CREATE INDEX IF NOT EXISTS idx_frag_crs_lookup
    ON memory_fragments(project_id, is_deprecated, last_accessed);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fragments_fts
    USING fts5(id UNINDEXED, content, project_id UNINDEXED,
               content='memory_fragments', content_rowid='rowid');

-- Triggers keep the external-content FTS index in sync automatically.
-- Without these, re-inserts on ON CONFLICT UPDATE accumulate stale postings.
CREATE TRIGGER IF NOT EXISTS frag_ai AFTER INSERT ON memory_fragments BEGIN
    INSERT INTO memory_fragments_fts(rowid, id, content, project_id)
    VALUES (new.rowid, new.id, new.content, new.project_id);
END;
CREATE TRIGGER IF NOT EXISTS frag_ad AFTER DELETE ON memory_fragments BEGIN
    INSERT INTO memory_fragments_fts(memory_fragments_fts, rowid, id, content, project_id)
    VALUES ('delete', old.rowid, old.id, old.content, old.project_id);
END;
CREATE TRIGGER IF NOT EXISTS frag_au AFTER UPDATE OF content ON memory_fragments BEGIN
    INSERT INTO memory_fragments_fts(memory_fragments_fts, rowid, id, content, project_id)
    VALUES ('delete', old.rowid, old.id, old.content, old.project_id);
    INSERT INTO memory_fragments_fts(rowid, id, content, project_id)
    VALUES (new.rowid, new.id, new.content, new.project_id);
END;

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    summary         TEXT,
    turn_count      INTEGER DEFAULT 0,
    model_id        TEXT,
    cost_input_tok  INTEGER DEFAULT 0,
    cost_output_tok INTEGER DEFAULT 0
);

-- §5.3 goals — the stated-intent table the intent-mirror reflects against.
-- goal_id columns already exist on memory_fragments + sessions; this is their target.
CREATE TABLE IF NOT EXISTS goals (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    statement   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed')),
    source      TEXT NOT NULL DEFAULT 'user' CHECK(source IN ('user','proposed')),
    created_at  TEXT NOT NULL,
    closed_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_goals_project_status ON goals(project_id, status);

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK(entity_type IN
                        ('file','function','class','package','config')),
    qualified_name  TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    properties_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_qualified
    ON entities(project_id, qualified_name);

CREATE TABLE IF NOT EXISTS triples (
    id              TEXT PRIMARY KEY,
    subject_id      TEXT NOT NULL REFERENCES entities(id),
    predicate       TEXT NOT NULL,
    object_id       TEXT NOT NULL REFERENCES entities(id),
    project_id      TEXT NOT NULL,
    source_fragment TEXT REFERENCES memory_fragments(id),
    confidence      REAL DEFAULT 0.8,
    created_at      TEXT NOT NULL,
    last_validated  TEXT
);
CREATE INDEX IF NOT EXISTS idx_triples_subject  ON triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_object   ON triples(object_id);
CREATE INDEX IF NOT EXISTS idx_triples_project  ON triples(project_id);

CREATE TABLE IF NOT EXISTS corrections (
    id              TEXT PRIMARY KEY,
    project_id      TEXT,
    original_fact   TEXT NOT NULL,
    corrected_fact  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    applied_to      TEXT
);

CREATE TABLE IF NOT EXISTS project_summaries (
    project_id      TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    fragment_count  INTEGER DEFAULT 0
);

-- §3.1 Pattern abstraction (Gap #2)
CREATE TABLE IF NOT EXISTS structural_patterns (
    id              TEXT PRIMARY KEY,
    signature_hash  TEXT UNIQUE NOT NULL,
    arity           INTEGER NOT NULL,
    abstract_form   TEXT NOT NULL DEFAULT '',
    failure_mode    TEXT,
    exemplars_json  TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL DEFAULT 0,
    occurrence_count INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL,
    last_seen       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_sig ON structural_patterns(signature_hash);

CREATE TABLE IF NOT EXISTS pending_structural_patterns (
    id              TEXT PRIMARY KEY,
    signature_hash  TEXT NOT NULL,
    arity           INTEGER NOT NULL,
    exemplars_json  TEXT NOT NULL,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_sig ON pending_structural_patterns(signature_hash);

-- §3.2 Counterfactual simulations (Gap #3)
CREATE TABLE IF NOT EXISTS simulations (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    seed_entity_id  TEXT,
    perturbation    TEXT NOT NULL,
    hypothesis      TEXT NOT NULL,
    predicted_facts_json TEXT NOT NULL,
    prior           REAL NOT NULL,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    outcome         TEXT,
    realization_evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_sim_unresolved ON simulations(outcome) WHERE outcome IS NULL;
CREATE INDEX IF NOT EXISTS idx_sim_project ON simulations(project_id);

-- §3.4 Epistemic events (Gap #1)
CREATE TABLE IF NOT EXISTS epistemic_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id   TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    delta_conf    REAL NOT NULL,
    evidence_frag TEXT,
    ts            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_epi_frag_ts ON epistemic_events(fragment_id, ts);

-- §5 Daemon observability
CREATE TABLE IF NOT EXISTS daemon_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    llm_calls   INTEGER DEFAULT 0,
    mutations   INTEGER DEFAULT 0,
    by_pass_json TEXT
);

-- §5.2 retrieval event log — becomes dataset for learned reranker
CREATE TABLE IF NOT EXISTS retrieval_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash          TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    fragment_ids_json   TEXT NOT NULL,
    crs_components_json TEXT,
    returned_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ret_evt_project ON retrieval_events(project_id, returned_at);

-- §5.5 cold storage sidecar index — enables O(1) search without full rehydration
CREATE TABLE IF NOT EXISTS cold_index (
    session_id  TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    summary     TEXT,
    frag_count  INTEGER DEFAULT 0,
    archived_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cold_idx_project ON cold_index(project_id, started_at);

-- §6.7 unified event log — single events table replacing epistemic_events + daemon_runs
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    ts          TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts);
"""

# ALTER TABLE migrations for new columns on memory_fragments (§2.1)
# Executed inside _apply_migrations(); duplicate-column errors are swallowed.
_COLUMN_MIGRATIONS = [
    # §2.1 epistemic tracking (original set)
    "ALTER TABLE memory_fragments ADD COLUMN epistemic_class    TEXT    DEFAULT 'observed'",
    "ALTER TABLE memory_fragments ADD COLUMN mutation_velocity  REAL    DEFAULT 0.0",
    "ALTER TABLE memory_fragments ADD COLUMN last_verified_at   TEXT",
    "ALTER TABLE memory_fragments ADD COLUMN contradiction_count INTEGER DEFAULT 0",
    "ALTER TABLE memory_fragments ADD COLUMN corroboration_count INTEGER DEFAULT 0",
    "ALTER TABLE memory_fragments ADD COLUMN realized_from_sim_id TEXT",
    # §5.1 belief revision
    "ALTER TABLE memory_fragments ADD COLUMN superseded_by      TEXT",
    # §5.3 goal alignment
    "ALTER TABLE memory_fragments ADD COLUMN goal_id            TEXT",
    # §5.6 producer provenance
    "ALTER TABLE memory_fragments ADD COLUMN producer_model     TEXT",
    "ALTER TABLE memory_fragments ADD COLUMN producer_version   TEXT",
    # §5.1 bitemporal triples
    "ALTER TABLE triples ADD COLUMN valid_from TEXT",
    "ALTER TABLE triples ADD COLUMN valid_to   TEXT",
    # §5.3 goal alignment on sessions
    "ALTER TABLE sessions ADD COLUMN goal_id TEXT",
    # §6.1 unified structural patterns — state column for pending|promoted
    "ALTER TABLE structural_patterns ADD COLUMN state TEXT DEFAULT 'promoted'",
]


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        # One connection, many threads (asyncio executor pool + the dashboard's
        # HTTP threads). check_same_thread=False silences sqlite's guardrail but
        # does NOT make concurrent use safe: two threads issuing statements on a
        # hand-managed BEGIN/COMMIT connection race into "cannot start a
        # transaction within a transaction" / "recursive use of cursors". This
        # reentrant lock serializes every use of the connection so access is
        # safe-by-construction. WAL + busy_timeout cover any separate connections
        # (e.g. mirror.py). The lock is shared with the v4 weight store, which
        # holds this same connection — see init_weight_store / CRSWeightStore.
        self._lock = threading.RLock()

    def connect(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._execute_script(DDL)
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        for stmt in _COLUMN_MIGRATIONS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # "duplicate column name" — already applied

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self):
        # Hold the lock for the WHOLE BEGIN..COMMIT so the transaction is both
        # atomic and isolated from concurrent reads/writes on the shared conn.
        # RLock is reentrant, so nested self.execute(...) calls below re-enter
        # without deadlock.
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _execute_script(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ── Fragment operations ─────────────────────────────────────────────────

    def upsert_fragment(self, f: MemoryFragment) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO memory_fragments
                    (id, project_id, scope, category, content, token_count,
                     abstraction_lvl, confidence, source_type, source_session,
                     created_at, last_accessed, access_count, is_pinned,
                     is_deprecated, deprecated_by, graph_centrality,
                     user_feedback, embedding_model, embedding_dim, metadata_json,
                     epistemic_class, mutation_velocity, last_verified_at,
                     contradiction_count, corroboration_count, realized_from_sim_id,
                     superseded_by, goal_id, producer_model, producer_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    content              = excluded.content,
                    token_count          = excluded.token_count,
                    abstraction_lvl      = excluded.abstraction_lvl,
                    confidence           = excluded.confidence,
                    last_accessed        = excluded.last_accessed,
                    access_count         = excluded.access_count,
                    is_pinned            = excluded.is_pinned,
                    is_deprecated        = excluded.is_deprecated,
                    deprecated_by        = excluded.deprecated_by,
                    graph_centrality     = excluded.graph_centrality,
                    user_feedback        = excluded.user_feedback,
                    metadata_json        = excluded.metadata_json,
                    epistemic_class      = excluded.epistemic_class,
                    mutation_velocity    = excluded.mutation_velocity,
                    last_verified_at     = excluded.last_verified_at,
                    contradiction_count  = excluded.contradiction_count,
                    corroboration_count  = excluded.corroboration_count,
                    realized_from_sim_id = excluded.realized_from_sim_id,
                    superseded_by        = excluded.superseded_by,
                    goal_id              = excluded.goal_id,
                    producer_model       = excluded.producer_model,
                    producer_version     = excluded.producer_version
            """, (
                f.id, f.project_id, f.scope, f.category, f.content, f.token_count,
                f.abstraction_lvl, f.confidence, f.source_type, f.source_session,
                f.created_at, f.last_accessed, f.access_count,
                int(f.is_pinned), int(f.is_deprecated), f.deprecated_by,
                f.graph_centrality, f.user_feedback,
                f.embedding_model, f.embedding_dim, f.metadata_json,
                f.epistemic_class, f.mutation_velocity, f.last_verified_at,
                f.contradiction_count, f.corroboration_count, f.realized_from_sim_id,
                f.superseded_by, f.goal_id, f.producer_model, f.producer_version,
            ))

    def get_fragments_batch(self, fragment_ids: list[str]) -> dict[str, MemoryFragment]:
        """Fetch multiple fragments in one query. Returns {id: fragment}."""
        if not fragment_ids:
            return {}
        placeholders = ",".join("?" * len(fragment_ids))
        rows = self.fetchall(
            f"SELECT * FROM memory_fragments WHERE id IN ({placeholders})",
            tuple(fragment_ids),
        )
        return {r["id"]: _row_to_fragment(r) for r in rows}

    def get_fragment(self, fragment_id: str) -> Optional[MemoryFragment]:
        row = self.fetchone(
            "SELECT * FROM memory_fragments WHERE id = ?", (fragment_id,)
        )
        return _row_to_fragment(row) if row else None

    def list_fragments(
        self,
        project_id: str,
        scope: Optional[str] = None,
        category: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> list[MemoryFragment]:
        clauses = ["project_id = ?"]
        params: list[Any] = [project_id]
        if scope:
            clauses.append("scope = ?"); params.append(scope)
        if category:
            clauses.append("category = ?"); params.append(category)
        if not include_deprecated:
            clauses.append("is_deprecated = 0")
        sql = f"SELECT * FROM memory_fragments WHERE {' AND '.join(clauses)}"
        return [_row_to_fragment(r) for r in self.fetchall(sql, tuple(params))]

    def list_fragment_ids(self, project_ids: list[str]) -> set[str]:
        """Return the set of non-deprecated fragment IDs for the given projects."""
        placeholders = ",".join("?" * len(project_ids))
        rows = self.fetchall(
            f"SELECT id FROM memory_fragments WHERE project_id IN ({placeholders}) AND is_deprecated=0",
            tuple(project_ids),
        )
        return {r["id"] for r in rows}

    def mark_deprecated(self, fragment_id: str, deprecated_by: Optional[str] = None) -> None:
        self.execute(
            "UPDATE memory_fragments SET is_deprecated=1, deprecated_by=? WHERE id=?",
            (deprecated_by, fragment_id),
        )

    def touch_fragment(self, fragment_id: str, accessed_at: str) -> None:
        self.execute("""
            UPDATE memory_fragments
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (accessed_at, fragment_id))

    def bm25_search(
        self, query: str, project_id: str, limit: int = 20
    ) -> list[tuple[str, float]]:
        rows = self.fetchall("""
            SELECT f.id, rank
            FROM memory_fragments_fts
            JOIN memory_fragments f ON memory_fragments_fts.id = f.id
            WHERE memory_fragments_fts MATCH ?
              AND f.project_id = ?
              AND f.is_deprecated = 0
            ORDER BY rank
            LIMIT ?
        """, (query, project_id, limit))
        # FTS5 rank is negative; invert so higher = better
        return [(r["id"], -r["rank"]) for r in rows]

    # ── Session operations ──────────────────────────────────────────────────

    def upsert_session(self, s: Session) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO sessions
                    (id, project_id, started_at, ended_at, summary,
                     turn_count, model_id, cost_input_tok, cost_output_tok, goal_id)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    ended_at        = excluded.ended_at,
                    summary         = COALESCE(excluded.summary, sessions.summary),
                    turn_count      = sessions.turn_count + excluded.turn_count,
                    cost_input_tok  = sessions.cost_input_tok  + excluded.cost_input_tok,
                    cost_output_tok = sessions.cost_output_tok + excluded.cost_output_tok,
                    goal_id         = COALESCE(excluded.goal_id, sessions.goal_id)
            """, (
                s.id, s.project_id, s.started_at, s.ended_at, s.summary,
                s.turn_count, s.model_id, s.cost_input_tok, s.cost_output_tok,
                s.goal_id,
            ))

    # ── Goal operations (§5.3 — intent mirror) ──────────────────────────────

    def insert_goal(self, g: "Goal") -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO goals
                    (id, project_id, statement, status, source, created_at, closed_at)
                VALUES (?,?,?,?,?,?,?)
            """, (g.id, g.project_id, g.statement, g.status, g.source,
                  g.created_at, g.closed_at))

    def list_goals(
        self,
        project_id: str,
        status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list["Goal"]:
        clauses = ["project_id = ?"]
        params: list[Any] = [project_id]
        if status:
            clauses.append("status = ?"); params.append(status)
        if source:
            clauses.append("source = ?"); params.append(source)
        rows = self.fetchall(
            f"SELECT * FROM goals WHERE {' AND '.join(clauses)} ORDER BY created_at",
            tuple(params),
        )
        return [_row_to_goal(r) for r in rows]

    def get_goal(self, goal_id: str) -> Optional["Goal"]:
        row = self.fetchone("SELECT * FROM goals WHERE id=?", (goal_id,))
        return _row_to_goal(row) if row else None

    def close_goal(
        self, goal_id: str, closed_at: str, project_id: Optional[str] = None
    ) -> bool:
        clauses = ["id = ?", "status = 'open'"]
        params: list[Any] = [goal_id]
        if project_id:
            clauses.append("project_id = ?"); params.append(project_id)
        cur = self.execute(
            f"UPDATE goals SET status='closed', closed_at=? WHERE {' AND '.join(clauses)}",
            (closed_at, *params),
        )
        return cur.rowcount > 0

    def dismiss_proposal(
        self, goal_id: str, closed_at: str, project_id: Optional[str] = None
    ) -> bool:
        """Retire a *proposed* goal only. Unlike close_goal, this refuses to touch
        a user-owned goal, so 'dismiss' can never silently close a real goal."""
        clauses = ["id = ?", "source = 'proposed'", "status = 'open'"]
        params: list[Any] = [goal_id]
        if project_id:
            clauses.append("project_id = ?"); params.append(project_id)
        cur = self.execute(
            f"UPDATE goals SET status='closed', closed_at=? WHERE {' AND '.join(clauses)}",
            (closed_at, *params),
        )
        return cur.rowcount > 0

    def confirm_goal(self, goal_id: str, project_id: Optional[str] = None) -> bool:
        """Promote a system-proposed goal to a user-owned one (the user's nod)."""
        clauses = ["id = ?", "source = 'proposed'", "status = 'open'"]
        params: list[Any] = [goal_id]
        if project_id:
            clauses.append("project_id = ?"); params.append(project_id)
        cur = self.execute(
            f"UPDATE goals SET source='user' WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        return cur.rowcount > 0

    # ── Entity operations ───────────────────────────────────────────────────

    def upsert_entity(self, e: Entity) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO entities
                    (id, project_id, entity_type, qualified_name,
                     last_seen, properties_json)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(project_id, qualified_name) DO UPDATE SET
                    last_seen       = excluded.last_seen,
                    properties_json = excluded.properties_json
            """, (
                e.id, e.project_id, e.entity_type,
                e.qualified_name, e.last_seen, e.properties_json,
            ))

    def get_entity_by_name(
        self, project_id: str, qualified_name: str
    ) -> Optional[Entity]:
        row = self.fetchone(
            "SELECT * FROM entities WHERE project_id=? AND qualified_name=?",
            (project_id, qualified_name),
        )
        return _row_to_entity(row) if row else None

    def fuzzy_match_entities(
        self, project_id: str, text: str, limit: int = 10
    ) -> list[Entity]:
        # Simple keyword match; replace with FTS or trigram for production
        rows = self.fetchall("""
            SELECT * FROM entities
            WHERE project_id = ? AND qualified_name LIKE ?
            LIMIT ?
        """, (project_id, f"%{text}%", limit))
        return [_row_to_entity(r) for r in rows]

    # ── Triple operations ───────────────────────────────────────────────────

    def upsert_triple(self, t: Triple) -> None:
        with self.transaction():
            self.execute("""
                INSERT OR IGNORE INTO triples
                    (id, subject_id, predicate, object_id, project_id,
                     source_fragment, confidence, created_at, last_validated,
                     valid_from, valid_to)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                t.id, t.subject_id, t.predicate, t.object_id, t.project_id,
                t.source_fragment, t.confidence, t.created_at, t.last_validated,
                t.valid_from, t.valid_to,
            ))

    def expire_triple(self, triple_id: str, valid_to: str) -> None:
        """Mark a triple as no longer valid (bitemporal invalidation)."""
        self.execute(
            "UPDATE triples SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
            (valid_to, triple_id),
        )

    def get_neighbors(
        self, entity_id: str, max_hops: int = 2
    ) -> list[str]:
        """BFS graph traversal; returns entity IDs within max_hops."""
        visited = {entity_id}
        frontier = {entity_id}
        for _ in range(max_hops):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self.fetchall(f"""
                SELECT subject_id, object_id FROM triples
                WHERE (subject_id IN ({placeholders})
                    OR object_id IN ({placeholders}))
                  AND confidence > 0.5
            """, (*frontier, *frontier))
            next_frontier = set()
            for r in rows:
                for nid in (r["subject_id"], r["object_id"]):
                    if nid not in visited:
                        visited.add(nid); next_frontier.add(nid)
            frontier = next_frontier
        visited.discard(entity_id)
        return list(visited)

    def fragments_linked_to_entity(self, entity_id: str) -> list[str]:
        rows = self.fetchall("""
            SELECT DISTINCT source_fragment FROM triples
            WHERE (subject_id=? OR object_id=?) AND source_fragment IS NOT NULL
        """, (entity_id, entity_id))
        return [r["source_fragment"] for r in rows]

    # ── Correction operations ───────────────────────────────────────────────

    def insert_correction(self, c: Correction) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO corrections
                    (id, project_id, original_fact, corrected_fact,
                     created_at, applied_to)
                VALUES (?,?,?,?,?,?)
            """, (
                c.id, c.project_id, c.original_fact,
                c.corrected_fact, c.created_at, c.applied_to,
            ))

    # ── Project summary operations ──────────────────────────────────────────

    def upsert_project_summary(
        self,
        project_id: str,
        content: str,
        generated_at: str,
        fragment_count: int = 0,
    ) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO project_summaries (project_id, content, generated_at, fragment_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    content        = excluded.content,
                    generated_at   = excluded.generated_at,
                    fragment_count = excluded.fragment_count
            """, (project_id, content, generated_at, fragment_count))

    def get_project_summary(self, project_id: str) -> Optional[str]:
        row = self.fetchone(
            "SELECT content FROM project_summaries WHERE project_id = ?",
            (project_id,),
        )
        return row["content"] if row else None

    def get_corrections(self, project_id: Optional[str] = None) -> list[Correction]:
        if project_id:
            rows = self.fetchall(
                "SELECT * FROM corrections WHERE project_id=? OR project_id IS NULL",
                (project_id,),
            )
        else:
            rows = self.fetchall("SELECT * FROM corrections WHERE project_id IS NULL")
        return [_row_to_correction(r) for r in rows]

    # ── Structural pattern operations (§3.1) ───────────────────────────────

    def get_structural_pattern_by_id(self, pattern_id: str) -> Optional[StructuralPattern]:
        row = self.fetchone(
            "SELECT * FROM structural_patterns WHERE id = ?", (pattern_id,)
        )
        return _row_to_pattern(row) if row else None

    def get_structural_pattern(self, signature_hash: str) -> Optional[StructuralPattern]:
        row = self.fetchone(
            "SELECT * FROM structural_patterns WHERE signature_hash = ?", (signature_hash,)
        )
        return _row_to_pattern(row) if row else None

    def upsert_structural_pattern(self, p: StructuralPattern) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO structural_patterns
                    (id, signature_hash, arity, abstract_form, failure_mode,
                     exemplars_json, embedding_dim, occurrence_count, created_at, last_seen)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(signature_hash) DO UPDATE SET
                    abstract_form    = excluded.abstract_form,
                    failure_mode     = excluded.failure_mode,
                    exemplars_json   = excluded.exemplars_json,
                    occurrence_count = excluded.occurrence_count,
                    last_seen        = excluded.last_seen
            """, (
                p.id, p.signature_hash, p.arity, p.abstract_form, p.failure_mode,
                p.exemplars_json, p.embedding_dim, p.occurrence_count,
                p.created_at, p.last_seen,
            ))

    def get_pending_pattern(self, signature_hash: str) -> Optional[PendingPattern]:
        row = self.fetchone(
            "SELECT * FROM pending_structural_patterns WHERE signature_hash = ?",
            (signature_hash,),
        )
        return _row_to_pending_pattern(row) if row else None

    def upsert_pending_pattern(self, p: PendingPattern) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO pending_structural_patterns
                    (id, signature_hash, arity, exemplars_json, first_seen, last_seen)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    exemplars_json = excluded.exemplars_json,
                    last_seen      = excluded.last_seen
            """, (p.id, p.signature_hash, p.arity, p.exemplars_json,
                  p.first_seen, p.last_seen))

    def get_pending_patterns_at_quorum(
        self, min_occurrences: int = 3, min_projects: int = 2
    ) -> list[PendingPattern]:
        rows = self.fetchall(
            "SELECT * FROM pending_structural_patterns"
        )
        result = []
        for row in rows:
            p = _row_to_pending_pattern(row)
            exemplars = json.loads(p.exemplars_json)
            if len(exemplars) < min_occurrences:
                continue
            projects = {e.get("project_id") for e in exemplars}
            if len(projects) < min_projects:
                continue
            result.append(p)
        return result

    def delete_pending_pattern(self, pattern_id: str) -> None:
        self.execute("DELETE FROM pending_structural_patterns WHERE id = ?", (pattern_id,))

    def list_unarticulated_patterns(self) -> list[StructuralPattern]:
        rows = self.fetchall(
            "SELECT * FROM structural_patterns WHERE abstract_form = '' OR abstract_form IS NULL"
        )
        return [_row_to_pattern(r) for r in rows]

    # ── Simulation operations (§3.2) ────────────────────────────────────────

    def insert_simulation(self, sim: Simulation) -> None:
        with self.transaction():
            self.execute("""
                INSERT OR IGNORE INTO simulations
                    (id, project_id, seed_entity_id, perturbation, hypothesis,
                     predicted_facts_json, prior, created_at,
                     resolved_at, outcome, realization_evidence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sim.id, sim.project_id, sim.seed_entity_id, sim.perturbation,
                sim.hypothesis, sim.predicted_facts_json, sim.prior, sim.created_at,
                sim.resolved_at, sim.outcome, sim.realization_evidence,
            ))

    def get_unresolved_simulations(self, project_id: str) -> list[Simulation]:
        rows = self.fetchall(
            "SELECT * FROM simulations WHERE project_id = ? AND outcome IS NULL",
            (project_id,),
        )
        return [_row_to_simulation(r) for r in rows]

    def resolve_simulation(
        self, sim_id: str, outcome: str, resolved_at: str,
        evidence: Optional[str] = None,
    ) -> None:
        self.execute("""
            UPDATE simulations
            SET outcome = ?, resolved_at = ?, realization_evidence = ?
            WHERE id = ?
        """, (outcome, resolved_at, evidence, sim_id))

    def expire_old_simulations(self, cutoff_iso: str) -> int:
        cursor = self.execute("""
            UPDATE simulations SET outcome = 'expired', resolved_at = ?
            WHERE outcome IS NULL AND created_at < ?
        """, (cutoff_iso, cutoff_iso))
        return cursor.rowcount

    def get_high_centrality_entities(self, project_id: str, limit: int = 5) -> list:
        return self.fetchall("""
            SELECT e.id, e.qualified_name, e.entity_type,
                   MAX(COALESCE(mf.graph_centrality, 0.0)) AS max_centrality
            FROM entities e
            LEFT JOIN triples t ON (t.subject_id = e.id OR t.object_id = e.id)
            LEFT JOIN memory_fragments mf ON mf.id = t.source_fragment
            WHERE e.project_id = ?
            GROUP BY e.id
            ORDER BY max_centrality DESC
            LIMIT ?
        """, (project_id, limit))

    def get_triples_for_entity(self, entity_id: str) -> list:
        return self.fetchall("""
            SELECT t.predicate,
                   es.entity_type AS subj_type, es.qualified_name AS subj_name,
                   eo.entity_type AS obj_type,  eo.qualified_name AS obj_name
            FROM triples t
            JOIN entities es ON t.subject_id = es.id
            JOIN entities eo ON t.object_id  = eo.id
            WHERE t.subject_id = ? OR t.object_id = ?
        """, (entity_id, entity_id))

    # ── Epistemic event operations (§3.4) ───────────────────────────────────

    def insert_epistemic_event(self, ev: EpistemicEvent) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO epistemic_events
                    (fragment_id, event_type, delta_conf, evidence_frag, ts)
                VALUES (?,?,?,?,?)
            """, (ev.fragment_id, ev.event_type, ev.delta_conf, ev.evidence_frag, ev.ts))

    def get_recent_epistemic_events(
        self, fragment_id: str, since_iso: str
    ) -> list[EpistemicEvent]:
        rows = self.fetchall("""
            SELECT * FROM epistemic_events
            WHERE fragment_id = ? AND ts >= ?
            ORDER BY ts DESC
        """, (fragment_id, since_iso))
        return [_row_to_epistemic_event(r) for r in rows]

    def count_contradiction_events(self, fragment_id: str, since_iso: str) -> int:
        row = self.fetchone("""
            SELECT COUNT(*) AS n FROM epistemic_events
            WHERE fragment_id = ? AND event_type = 'contradiction' AND ts >= ?
        """, (fragment_id, since_iso))
        return row["n"] if row else 0

    def get_mutation_velocities(self, project_id: str) -> list[float]:
        rows = self.fetchall("""
            SELECT mutation_velocity FROM memory_fragments
            WHERE project_id = ? AND is_deprecated = 0 AND mutation_velocity > 0
        """, (project_id,))
        return [r["mutation_velocity"] for r in rows]

    # ── Daemon observability (§5) ───────────────────────────────────────────

    def log_daemon_run(
        self, started_at: str, ended_at: str,
        llm_calls: int, mutations: int, by_pass: dict,
    ) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO daemon_runs
                    (started_at, ended_at, llm_calls, mutations, by_pass_json)
                VALUES (?,?,?,?,?)
            """, (started_at, ended_at, llm_calls, mutations, json.dumps(by_pass)))
            # Mirror to unified events table (§6.7)
            self.execute(
                "INSERT INTO events (kind, ts, payload_json) VALUES (?,?,?)",
                ("daemon_run", started_at, json.dumps({
                    "started_at": started_at, "ended_at": ended_at,
                    "llm_calls": llm_calls, "mutations": mutations, **by_pass,
                })),
            )

    # ── Retrieval event log (§5.2) ──────────────────────────────────────────

    def log_retrieval_event(self, ev: "RetrievalEvent") -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO retrieval_events
                    (query_hash, project_id, fragment_ids_json, crs_components_json, returned_at)
                VALUES (?,?,?,?,?)
            """, (ev.query_hash, ev.project_id, ev.fragment_ids_json,
                  ev.crs_components_json, ev.returned_at))

    # ── Cold sidecar index (§5.5) ───────────────────────────────────────────

    def upsert_cold_index(
        self,
        session_id: str,
        project_id: str,
        started_at: str,
        archived_at: str,
        summary: Optional[str] = None,
        frag_count: int = 0,
    ) -> None:
        with self.transaction():
            self.execute("""
                INSERT INTO cold_index
                    (session_id, project_id, started_at, summary, frag_count, archived_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary     = COALESCE(excluded.summary, cold_index.summary),
                    frag_count  = excluded.frag_count,
                    archived_at = excluded.archived_at
            """, (session_id, project_id, started_at, summary, frag_count, archived_at))

    def search_cold_index(
        self, project_id: str, since_iso: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        if since_iso:
            rows = self.fetchall("""
                SELECT * FROM cold_index WHERE project_id = ? AND started_at >= ?
                ORDER BY started_at DESC LIMIT ?
            """, (project_id, since_iso, limit))
        else:
            rows = self.fetchall("""
                SELECT * FROM cold_index WHERE project_id = ?
                ORDER BY started_at DESC LIMIT ?
            """, (project_id, limit))
        return [dict(r) for r in rows]

    # ── Unified events table (§6.7) ─────────────────────────────────────────

    def log_event(self, kind: str, ts: str, payload: dict) -> None:
        with self.transaction():
            self.execute(
                "INSERT INTO events (kind, ts, payload_json) VALUES (?,?,?)",
                (kind, ts, json.dumps(payload)),
            )

    def get_events(
        self, kind: Optional[str] = None, since_iso: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        if kind:
            clauses.append("kind = ?"); params.append(kind)
        if since_iso:
            clauses.append("ts >= ?"); params.append(since_iso)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.fetchall(
            f"SELECT * FROM events {where} ORDER BY ts DESC LIMIT ?", tuple(params)
        )
        return [{"id": r["id"], "kind": r["kind"], "ts": r["ts"],
                 **json.loads(r["payload_json"])} for r in rows]

    # ── Unified structural patterns (§6.1) ──────────────────────────────────

    def list_structural_patterns_by_state(self, state: str) -> list[StructuralPattern]:
        rows = self.fetchall(
            "SELECT * FROM structural_patterns WHERE COALESCE(state,'promoted') = ?", (state,)
        )
        return [_row_to_pattern(r) for r in rows]

    def promote_structural_pattern(self, pattern_id: str) -> None:
        self.execute(
            "UPDATE structural_patterns SET state = 'promoted' WHERE id = ?",
            (pattern_id,),
        )

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(self, project_id: Optional[str] = None) -> dict:
        if project_id:
            rows = self.fetchall("""
                SELECT scope, category, COUNT(*) as n
                FROM memory_fragments
                WHERE project_id=? AND is_deprecated=0
                GROUP BY scope, category
            """, (project_id,))
        else:
            rows = self.fetchall("""
                SELECT scope, category, COUNT(*) as n
                FROM memory_fragments
                WHERE is_deprecated=0
                GROUP BY scope, category
            """)
        return {f"{r['scope']}/{r['category']}": r["n"] for r in rows}

    def health(self, since_iso: str) -> dict:
        """Honest self-report for the status readout: how much is stored, how
        fresh it is, and whether new memories are actually arriving. `since_iso`
        is the ISO cutoff for the 'recently added' window (e.g. 24h ago)."""
        total = self.fetchone("SELECT COUNT(*) AS n FROM memory_fragments")["n"]
        active = self.fetchone(
            "SELECT COUNT(*) AS n FROM memory_fragments WHERE is_deprecated=0"
        )["n"]
        newest = self.fetchone(
            "SELECT MAX(created_at) AS t FROM memory_fragments"
        )["t"]
        recent = self.fetchone(
            "SELECT COUNT(*) AS n FROM memory_fragments WHERE created_at >= ?",
            (since_iso,),
        )["n"]
        return {
            "fragments_total": total,
            "fragments_active": active,
            "newest_created_at": newest,
            "added_recently": recent,
        }


# ── Row → dataclass helpers ─────────────────────────────────────────────────

def _row_to_fragment(r: sqlite3.Row) -> MemoryFragment:
    keys = r.keys()
    return MemoryFragment(
        id=r["id"], project_id=r["project_id"], scope=r["scope"],
        category=r["category"], content=r["content"],
        token_count=r["token_count"], abstraction_lvl=r["abstraction_lvl"],
        confidence=r["confidence"], source_type=r["source_type"],
        source_session=r["source_session"], created_at=r["created_at"],
        last_accessed=r["last_accessed"], access_count=r["access_count"],
        is_pinned=bool(r["is_pinned"]), is_deprecated=bool(r["is_deprecated"]),
        deprecated_by=r["deprecated_by"], graph_centrality=r["graph_centrality"],
        user_feedback=r["user_feedback"], embedding_model=r["embedding_model"],
        embedding_dim=r["embedding_dim"], metadata_json=r["metadata_json"],
        # §2.1 epistemic columns — default gracefully if migration hasn't run yet
        epistemic_class=r["epistemic_class"] if "epistemic_class" in keys else "observed",
        mutation_velocity=r["mutation_velocity"] if "mutation_velocity" in keys else 0.0,
        last_verified_at=r["last_verified_at"] if "last_verified_at" in keys else None,
        contradiction_count=r["contradiction_count"] if "contradiction_count" in keys else 0,
        corroboration_count=r["corroboration_count"] if "corroboration_count" in keys else 0,
        realized_from_sim_id=r["realized_from_sim_id"] if "realized_from_sim_id" in keys else None,
        # §5 new provenance / goal columns
        superseded_by=r["superseded_by"] if "superseded_by" in keys else None,
        goal_id=r["goal_id"] if "goal_id" in keys else None,
        producer_model=r["producer_model"] if "producer_model" in keys else None,
        producer_version=r["producer_version"] if "producer_version" in keys else None,
    )


def _row_to_goal(r: sqlite3.Row) -> Goal:
    return Goal(
        id=r["id"], project_id=r["project_id"], statement=r["statement"],
        status=r["status"], source=r["source"],
        created_at=r["created_at"], closed_at=r["closed_at"],
    )


def _row_to_entity(r: sqlite3.Row) -> Entity:
    return Entity(
        id=r["id"], project_id=r["project_id"], entity_type=r["entity_type"],
        qualified_name=r["qualified_name"], last_seen=r["last_seen"],
        properties_json=r["properties_json"],
    )


def _row_to_correction(r: sqlite3.Row) -> Correction:
    return Correction(
        id=r["id"], project_id=r["project_id"],
        original_fact=r["original_fact"], corrected_fact=r["corrected_fact"],
        created_at=r["created_at"], applied_to=r["applied_to"],
    )


def _row_to_pattern(r: sqlite3.Row) -> StructuralPattern:
    keys = r.keys()
    return StructuralPattern(
        id=r["id"], signature_hash=r["signature_hash"], arity=r["arity"],
        abstract_form=r["abstract_form"] or "" if "abstract_form" in keys else "",
        failure_mode=r["failure_mode"] if "failure_mode" in keys else None,
        exemplars_json=r["exemplars_json"],
        embedding_dim=r["embedding_dim"] if "embedding_dim" in keys else 0,
        occurrence_count=r["occurrence_count"] if "occurrence_count" in keys else 1,
        created_at=r["created_at"] if "created_at" in keys else r.get("first_seen", ""),
        last_seen=r["last_seen"],
    )


def _row_to_pending_pattern(r: sqlite3.Row) -> PendingPattern:
    return PendingPattern(
        id=r["id"], signature_hash=r["signature_hash"], arity=r["arity"],
        exemplars_json=r["exemplars_json"],
        first_seen=r["first_seen"], last_seen=r["last_seen"],
    )


def _row_to_simulation(r: sqlite3.Row) -> Simulation:
    return Simulation(
        id=r["id"], project_id=r["project_id"],
        seed_entity_id=r["seed_entity_id"],
        perturbation=r["perturbation"], hypothesis=r["hypothesis"],
        predicted_facts_json=r["predicted_facts_json"],
        prior=r["prior"], created_at=r["created_at"],
        resolved_at=r["resolved_at"], outcome=r["outcome"],
        realization_evidence=r["realization_evidence"],
    )


def _row_to_epistemic_event(r: sqlite3.Row) -> EpistemicEvent:
    return EpistemicEvent(
        id=r["id"], fragment_id=r["fragment_id"],
        event_type=r["event_type"], delta_conf=r["delta_conf"],
        evidence_frag=r["evidence_frag"], ts=r["ts"],
    )
