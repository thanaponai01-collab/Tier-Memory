"""
v4_reranker.py — The self-improving CRS reranker.

This is the one loop that makes Tier Memory *learn* instead of *accumulate*.

It reuses tables you already have (retrieval_events, corrections, fragments,
their access stats). It fits a logistic-regression weight vector over the six
CRS components against a "was this fragment actually useful?" label, then
damps those weights into the live scorer. Per-project weights, with a global
fallback for cold-start projects.

No new dependencies. Pure numpy. Runs in milliseconds on weeks of events.

Wire-in points (only two):
  1. CRS scoring reads weights from CRSWeightStore instead of hardcoded consts.
  2. Add `Pass 10 -> learn_crs_weights` to MemoryAuditor.audit().

------------------------------------------------------------------------------
DESIGN CONTRACTS (read these before touching anything)
------------------------------------------------------------------------------
- CRS is a PURE derived function of stored components. We never write final_crs
  back as state. We only ever change the *weights*, which live in one table.
- The epistemic_multiplier is NOT learned. It is a hand-set trust guardrail
  applied AFTER the learned linear combination. The model is not allowed to
  decide that `simulated=0.40` fragments are trustworthy just because a few of
  them got clicked.
- Weights are bounded (clamped non-negative, renormalized to sum to 1) so a bad
  training week cannot invert the meaning of a signal.
- Updates are DAMPED: w_new = (1-eta)*w_old + eta*w_fit. One week never wins.
- Cold start: a project with < MIN_EVENTS labeled rows borrows global weights.

Schema note: retrieval_events stores components as a JSON blob (crs_components_json)
keyed by fragment_id, and timestamps as ISO strings (returned_at). The fetch
functions unpack these in Python; everything downstream is unchanged.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# The six CRS components, in fixed order. This order is the contract between
# the scorer, the event log, and the weight vector. Do not reorder.
COMPONENTS = ("semantic", "recency", "frequency", "importance", "feedback", "confidence")
N_COMPONENTS = len(COMPONENTS)

# The v3 hardcoded weights become the *prior* / cold-start default.
DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.10, 0.15, 0.05, 0.15], dtype=np.float64)

GLOBAL_SCOPE = "__global__"


@dataclass
class RerankerConfig:
    # A project needs at least this many labeled retrieval rows before it gets
    # its own weights. Below this it uses the global vector (cold start).
    min_events_for_local: int = 200

    # Damping factor for the weight update. Low = conservative = stable.
    learning_rate: float = 0.20

    # How far back to pull retrieval events each audit cycle.
    lookback_days: float = 30.0

    # A retrieval is "useful" if the fragment is touched again within this many
    # seconds of being retrieved (the "re-touched soon" signal).
    retouch_window_seconds: float = 3 * 24 * 3600.0  # 3 days

    # Logistic regression: L2 strength + iterations. Tiny dataset -> regularize.
    l2: float = 1.0
    max_iter: int = 200
    lr_gd: float = 0.5  # gradient-descent step for the LR fit itself

    # Floor each weight at this before renormalizing, so no signal dies forever
    # (a dead signal can never come back, because its weight stops the gradient).
    weight_floor: float = 0.02


# ----------------------------------------------------------------------------
# Weight storage  (the only new persistent state v4 introduces)
# ----------------------------------------------------------------------------

CREATE_WEIGHTS_TABLE = """
CREATE TABLE IF NOT EXISTS crs_weights (
    scope        TEXT PRIMARY KEY,   -- project_id, or '__global__'
    weights_json TEXT NOT NULL,      -- JSON list[6], same order as COMPONENTS
    n_events     INTEGER NOT NULL,   -- training rows behind this fit
    updated_at   REAL NOT NULL,
    auc          REAL                -- last fit's train AUC, for the dashboard
);
"""


class CRSWeightStore:
    """Reads/writes learned weight vectors. The scorer reads from here.

    This store holds the SAME sqlite connection as the daemon's Database, used
    from multiple threads (scorer on the read path, auditor on the learn path).
    `lock` must be that Database's reentrant lock so weight-store access is
    serialized against all other use of the connection; without it, a get/put
    here can race a write elsewhere on the shared connection.
    """

    def __init__(self, conn: sqlite3.Connection, lock: Optional["threading.RLock"] = None):
        import threading
        self.conn = conn
        self._lock = lock or threading.RLock()
        with self._lock:
            self.conn.execute(CREATE_WEIGHTS_TABLE)
            self.conn.commit()
        self._cache: dict[str, np.ndarray] = {}

    def get_weights(self, project_id: str) -> np.ndarray:
        """
        Return the weight vector the scorer should use for this project.
        Falls back to global, then to the v3 default. Cached per process.
        """
        cached = self._cache.get(project_id)
        if cached is not None:
            return cached  # cache hit: no DB, no lock contention on the hot path

        with self._lock:
            cached = self._cache.get(project_id)  # re-check under lock
            if cached is not None:
                return cached

            row = self.conn.execute(
                "SELECT weights_json, n_events FROM crs_weights WHERE scope = ?",
                (project_id,),
            ).fetchone()

            if row and row[1] >= 0:  # local weights exist
                w = np.array(json.loads(row[0]), dtype=np.float64)
            else:
                grow = self.conn.execute(
                    "SELECT weights_json FROM crs_weights WHERE scope = ?",
                    (GLOBAL_SCOPE,),
                ).fetchone()
                w = np.array(json.loads(grow[0]), dtype=np.float64) if grow else DEFAULT_WEIGHTS.copy()

            self._cache[project_id] = w
            return w

    def put_weights(self, scope: str, w: np.ndarray, n_events: int, auc: float) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO crs_weights (scope, weights_json, n_events, updated_at, auc)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(scope) DO UPDATE SET
                     weights_json=excluded.weights_json,
                     n_events=excluded.n_events,
                     updated_at=excluded.updated_at,
                     auc=excluded.auc""",
                (scope, json.dumps(w.tolist()), n_events, time.time(), auc),
            )
            self.conn.commit()
            self._cache.pop(scope, None)  # invalidate


# ----------------------------------------------------------------------------
# The scoring function the read path now calls
# ----------------------------------------------------------------------------

def compute_crs(
    components: dict[str, float],
    epistemic_multiplier: float,
    weights: np.ndarray,
) -> float:
    """
    Pure derived CRS. Replaces the hardcoded formula in scoring.py.

    `components` is the dict the read path already computes per fragment.
    `weights` comes from CRSWeightStore.get_weights(project_id).
    The epistemic_multiplier is applied AFTER, unchanged — it is a guardrail,
    not a learned term.
    """
    x = np.array([components[c] for c in COMPONENTS], dtype=np.float64)
    base = float(np.dot(weights, x))
    return base * epistemic_multiplier


# ----------------------------------------------------------------------------
# Label assembly — turn logged events + outcomes into (X, y)
# ----------------------------------------------------------------------------

@dataclass
class TrainingData:
    X: np.ndarray              # (n, 6) component matrix
    y: np.ndarray              # (n,) 0/1 useful label
    project_id: str
    n_pos: int = field(init=False)

    def __post_init__(self):
        self.n_pos = int(self.y.sum())


def _fetch_training_rows(
    conn: sqlite3.Connection, project_id: str, cfg: RerankerConfig
) -> TrainingData:
    """
    Build the supervised dataset for one project.

    Positive label = the retrieved fragment was USEFUL, defined as ANY of:
      (a) re-touched within retouch_window after this retrieval, OR
      (b) pinned (is_pinned = 1), OR
      (c) positive user_feedback (> 0), OR
      (d) cited — actually used in an answer after being injected by the read
          reflex (times_cited > 0). This is the load-bearing signal: unlike
          re-touch (a click proxy), it means the fragment's content surfaced in
          real produced work.

    Negative = retrieved, surfaced into context, and none of the above.
    Deprecated-as-wrong rows are dropped (correctness != relevance signal).

    Schema adaptation: retrieval_events stores components as crs_components_json
    (a JSON dict keyed by fragment_id) and uses returned_at (ISO string) not
    created_at. This function unpacks the blobs in Python.
    """
    cutoff = time.time() - cfg.lookback_days * 86400.0
    cutoff_iso = _dt.datetime.fromtimestamp(
        cutoff, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S")

    events = conn.execute(
        """
        SELECT fragment_ids_json, crs_components_json, returned_at
        FROM retrieval_events
        WHERE project_id = ? AND returned_at >= ?
        """,
        (project_id, cutoff_iso),
    ).fetchall()

    # Expand JSON blobs → flat (fragment_id, components_dict, retrieved_at_unix)
    frag_entries: list[tuple[str, dict, float]] = []
    all_frag_ids: set[str] = set()

    for fids_json, comps_json, returned_at in events:
        try:
            fids = json.loads(fids_json) if fids_json else []
            comps = json.loads(comps_json) if comps_json else {}
            ra_ts = _dt.datetime.fromisoformat(returned_at).timestamp()
        except Exception:
            continue
        for fid in fids:
            if fid in comps:
                frag_entries.append((fid, comps[fid], ra_ts))
                all_frag_ids.add(fid)

    if not all_frag_ids:
        return TrainingData(X=np.empty((0, N_COMPONENTS)), y=np.empty((0,)), project_id=project_id)

    # Fetch outcome columns for all referenced fragments in one query
    placeholders = ",".join("?" * len(all_frag_ids))
    frag_rows = conn.execute(
        f"""
        SELECT id, last_accessed,
               COALESCE(is_pinned, 0)       AS is_pinned,
               COALESCE(user_feedback, 0.0) AS user_feedback,
               COALESCE(is_deprecated, 0)   AS is_deprecated,
               COALESCE(times_cited, 0)     AS times_cited
        FROM memory_fragments
        WHERE id IN ({placeholders})
        """,
        list(all_frag_ids),
    ).fetchall()
    frag_map = {r[0]: r for r in frag_rows}

    X_list: list[list[float]] = []
    y_list: list[int] = []

    for fid, c, ra_ts in frag_entries:
        if fid not in frag_map:
            continue
        _, last_accessed, is_pinned, user_feedback, is_deprecated, times_cited = frag_map[fid]

        if is_deprecated:
            continue

        try:
            la_ts = _dt.datetime.fromisoformat(last_accessed).timestamp() if last_accessed else None
        except Exception:
            la_ts = None

        retouched = la_ts is not None and 0 < (la_ts - ra_ts) <= cfg.retouch_window_seconds
        useful = retouched or bool(is_pinned) or (user_feedback > 0) or (times_cited > 0)

        try:
            row = [float(c.get(comp, 0.0)) for comp in COMPONENTS]
        except (TypeError, ValueError, AttributeError):
            continue

        X_list.append(row)
        y_list.append(1 if useful else 0)

    X = np.array(X_list, dtype=np.float64) if X_list else np.empty((0, N_COMPONENTS))
    y = np.array(y_list, dtype=np.float64) if y_list else np.empty((0,))
    return TrainingData(X=X, y=y, project_id=project_id)


# ----------------------------------------------------------------------------
# The fit — logistic regression by gradient descent (numpy, no sklearn)
# ----------------------------------------------------------------------------

def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0  # guard constant columns
    return (X - mu) / sd, mu, sd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logistic(X: np.ndarray, y: np.ndarray, cfg: RerankerConfig) -> np.ndarray:
    """
    Returns coefficient vector in ORIGINAL feature space (length 6, no bias —
    we map coefficients to relevance weights, and an intercept is just a global
    prior we don't carry into CRS). Standardizes internally for conditioning.
    """
    Xs, mu, sd = _standardize(X)
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(cfg.max_iter):
        p = _sigmoid(Xs @ w + b)
        err = p - y
        grad_w = (Xs.T @ err) / n + cfg.l2 * w / n
        grad_b = err.mean()
        w -= cfg.lr_gd * grad_w
        b -= cfg.lr_gd * grad_b
    # Map standardized coeffs back to original space.
    return w / sd


def _coeffs_to_weights(coeffs: np.ndarray, cfg: RerankerConfig) -> np.ndarray:
    """
    Convert raw LR coefficients into a valid CRS weight vector:
      - clamp negatives to 0 (a signal can't *hurt* relevance in CRS),
      - floor each at weight_floor so no signal becomes unrecoverable,
      - renormalize to sum to 1 (keeps CRS on its original ~[0,1] scale).
    """
    w = np.maximum(coeffs, 0.0)
    if w.sum() < 1e-8:
        w = DEFAULT_WEIGHTS.copy()  # degenerate fit -> fall back to prior
    w = np.maximum(w, cfg.weight_floor)
    return w / w.sum()


def _auc(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Rank-based AUC of the weight vector as a scorer. Cheap, no deps."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    scores = X @ w
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ----------------------------------------------------------------------------
# Pass 10 — the audit entry point
# ----------------------------------------------------------------------------

def learn_crs_weights(
    conn: sqlite3.Connection,
    project_id: str,
    store: CRSWeightStore,
    cfg: Optional[RerankerConfig] = None,
) -> dict:
    """
    One project's reranker update. Call this from MemoryAuditor as Pass 10,
    and once with project_id=GLOBAL_SCOPE per cycle to refresh the fallback
    (using all projects' events — see learn_global_weights below).

    Returns a small dict for the daemon_run / events log.
    """
    cfg = cfg or RerankerConfig()
    # _fetch_training_rows reads the shared connection directly; hold the store
    # lock (the Database lock) so these SELECTs don't race a concurrent write.
    with store._lock:
        data = _fetch_training_rows(conn, project_id, cfg)

    # Cold start / not enough signal -> leave project on global fallback.
    if len(data.y) < cfg.min_events_for_local or data.n_pos < 10:
        return {
            "scope": project_id, "status": "cold_start",
            "n_events": int(len(data.y)), "n_pos": int(data.n_pos),
        }

    coeffs = _fit_logistic(data.X, data.y, cfg)
    w_fit = _coeffs_to_weights(coeffs, cfg)

    # Damped update against whatever the project is currently using.
    w_old = store.get_weights(project_id)
    w_new = (1 - cfg.learning_rate) * w_old + cfg.learning_rate * w_fit
    w_new = w_new / w_new.sum()  # renormalize after the blend

    auc = _auc(data.X, data.y, w_new)
    store.put_weights(project_id, w_new, n_events=len(data.y), auc=auc)

    return {
        "scope": project_id, "status": "updated",
        "n_events": int(len(data.y)), "n_pos": int(data.n_pos),
        "auc": None if math.isnan(auc) else round(auc, 4),
        "weights": {c: round(float(v), 4) for c, v in zip(COMPONENTS, w_new)},
    }


def learn_global_weights(
    conn: sqlite3.Connection, store: CRSWeightStore, cfg: Optional[RerankerConfig] = None
) -> dict:
    """
    Refresh the global fallback from ALL projects' events pooled together.
    Run this once per audit cycle, before the per-project passes, so new
    projects inherit the best current global prior.
    """
    cfg = cfg or RerankerConfig()
    cutoff = time.time() - cfg.lookback_days * 86400.0
    cutoff_iso = _dt.datetime.fromtimestamp(
        cutoff, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S")

    # Hold the store lock (the Database lock) around the direct reads on the
    # shared connection so they don't race concurrent writes.
    with store._lock:
        events = conn.execute(
            """
            SELECT fragment_ids_json, crs_components_json, returned_at
            FROM retrieval_events
            WHERE returned_at >= ?
            """,
            (cutoff_iso,),
        ).fetchall()

    frag_entries: list[tuple[str, dict, float]] = []
    all_frag_ids: set[str] = set()

    for fids_json, comps_json, returned_at in events:
        try:
            fids = json.loads(fids_json) if fids_json else []
            comps = json.loads(comps_json) if comps_json else {}
            ra_ts = _dt.datetime.fromisoformat(returned_at).timestamp()
        except Exception:
            continue
        for fid in fids:
            if fid in comps:
                frag_entries.append((fid, comps[fid], ra_ts))
                all_frag_ids.add(fid)

    def _seed_global_if_missing():
        with store._lock:
            exists = store.conn.execute(
                "SELECT 1 FROM crs_weights WHERE scope=?", (GLOBAL_SCOPE,)
            ).fetchone()
        if exists is None:
            store.put_weights(GLOBAL_SCOPE, DEFAULT_WEIGHTS.copy(), 0, float("nan"))

    if not all_frag_ids:
        _seed_global_if_missing()
        return {"scope": GLOBAL_SCOPE, "status": "no_events", "n_events": 0}

    placeholders = ",".join("?" * len(all_frag_ids))
    with store._lock:
        frag_rows = conn.execute(
            f"""
            SELECT id, last_accessed,
                   COALESCE(is_pinned, 0)       AS is_pinned,
                   COALESCE(user_feedback, 0.0) AS user_feedback,
                   COALESCE(is_deprecated, 0)   AS is_deprecated,
                   COALESCE(times_cited, 0)     AS times_cited
            FROM memory_fragments
            WHERE id IN ({placeholders})
            """,
            list(all_frag_ids),
        ).fetchall()
    frag_map = {r[0]: r for r in frag_rows}

    X, y = [], []
    for fid, c, ra_ts in frag_entries:
        if fid not in frag_map:
            continue
        _, last_accessed, is_pinned, user_feedback, is_deprecated, times_cited = frag_map[fid]
        if is_deprecated:
            continue
        try:
            la_ts = _dt.datetime.fromisoformat(last_accessed).timestamp() if last_accessed else None
        except Exception:
            la_ts = None
        retouched = la_ts is not None and 0 < (la_ts - ra_ts) <= cfg.retouch_window_seconds
        useful = retouched or bool(is_pinned) or (user_feedback > 0) or (times_cited > 0)
        try:
            row = [float(c.get(comp, 0.0)) for comp in COMPONENTS]
        except (TypeError, ValueError, AttributeError):
            continue
        X.append(row)
        y.append(1 if useful else 0)

    X = np.array(X, dtype=np.float64) if X else np.empty((0, N_COMPONENTS))
    y = np.array(y, dtype=np.float64) if y else np.empty((0,))

    if len(y) < cfg.min_events_for_local or y.sum() < 10:
        _seed_global_if_missing()
        return {"scope": GLOBAL_SCOPE, "status": "insufficient_data", "n_events": int(len(y))}

    w_fit = _coeffs_to_weights(_fit_logistic(X, y, cfg), cfg)
    w_old = store.get_weights(GLOBAL_SCOPE)
    w_new = (1 - cfg.learning_rate) * w_old + cfg.learning_rate * w_fit
    w_new = w_new / w_new.sum()
    auc = _auc(X, y, w_new)
    store.put_weights(GLOBAL_SCOPE, w_new, len(y), auc)
    return {
        "scope": GLOBAL_SCOPE, "status": "updated", "n_events": int(len(y)),
        "auc": None if math.isnan(auc) else round(auc, 4),
        "weights": {c: round(float(v), 4) for c, v in zip(COMPONENTS, w_new)},
    }
