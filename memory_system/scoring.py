"""
Composite Relevance Score (CRS) — Section 2.4 of the architecture.

CRS drives both retrieval ranking and eviction tier placement:
  [0.60, 1.00] → Hot  (eligible for L1 prefix injection)
  [0.30, 0.59] → Warm (eligible for L2 semantic retrieval)
  [0.15, 0.29] → Cool (retained but not actively indexed)
  [0.00, 0.14] → Cold (moved to compressed cold storage)

§3.5 adds W_CONFIDENCE (0.15) and an EPISTEMIC_MULTIPLIER applied outside the
weighted sum, so simulated content can be retrieved but never out-ranks observed.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from .models import MemoryFragment, Score
from .embedder import cosine_similarity as _cosine_similarity  # canonical implementation

if TYPE_CHECKING:
    import sqlite3
    from .v4_reranker import CRSWeightStore

# Tuning constants
RECENCY_DECAY_LAMBDA = 0.003       # half-life ≈ 10 days
MAX_EXPECTED_ACCESS = 1_000        # cap for logarithmic frequency scaling

# v3 hardcoded weights — kept as fallback when weight_store is not initialised.
# v4 replaces these with learned per-project weights via init_weight_store().
W_SEMANTIC    = 0.30
W_RECENCY     = 0.25
W_FREQUENCY   = 0.10
W_IMPORTANCE  = 0.15
W_FEEDBACK    = 0.05
W_CONFIDENCE  = 0.15

def init_weight_store(conn: "sqlite3.Connection", lock=None) -> "CRSWeightStore":
    """Construct the per-project learned-weight store at daemon startup.

    Returns the store so the caller can thread it explicitly into
    fused_retrieval / the auditor. There is intentionally no module-level
    singleton: scoring is a pure function of (fragment, query, weights), so the
    weight store is always an explicit argument. This keeps composite_relevance_score
    verifiable in isolation and free of hidden process-global state.

    Pass the Database's reentrant lock so the weight store's reads/writes on the
    shared connection are serialized against all other DB access.
    """
    from .v4_reranker import CRSWeightStore
    return CRSWeightStore(conn, lock)

# §3.5 — epistemic class multiplier (applied outside weighted sum)
EPISTEMIC_MULTIPLIER: dict[str, float] = {
    "observed":           1.00,
    "reflected":          0.95,
    "consolidated":       0.90,
    "analogical":         0.80,
    "simulated_realized": 0.85,
    "simulated":          0.40,  # heavy penalty; surfaces only when nothing else does
}

# Eviction thresholds
HOT_THRESHOLD  = 0.60
WARM_THRESHOLD = 0.15


def composite_relevance_score(
    fragment: MemoryFragment,
    query_embedding: Optional[list[float]] = None,
    semantic_override: Optional[float] = None,
    weight_store: Optional["CRSWeightStore"] = None,
) -> Score:
    """
    Returns a Score [0.0, 1.0].  Pinned fragments always score 1.0.
    Score implements __float__ and comparison operators so existing code
    treating the return value as a float continues to work unchanged.

    Pass query_embedding for retrieval-time scoring;
    omit it for maintenance/eviction scoring (uses historical avg similarity).

    semantic_override: the query↔fragment cosine the vector lane already
    computed. Fragments loaded from the DB never carry their embedding (vectors
    live in the HNSW index, not a column), so without this the semantic signal
    silently collapses to the 0.5 neutral fallback at retrieval time. Pass the
    vector-search similarity here so the largest CRS weight scores real meaning.

    weight_store: per-project learned CRS weights (v4). When None, the hardcoded
    v3 weights below are used. Always an explicit argument — no module global —
    so this function's output depends only on what it is handed.
    """
    if fragment.is_pinned:
        return Score(value=1.0, components={
            "semantic": 1.0, "recency": 1.0, "frequency": 1.0,
            "importance": 1.0, "feedback": 1.0, "confidence": 1.0,
        }, multiplier=1.0)

    recency    = _recency(fragment.last_accessed)
    frequency  = _frequency(fragment.access_count)
    importance = _clamp(fragment.graph_centrality, 0.0, 1.0)
    feedback   = _clamp((fragment.user_feedback + 1.0) / 2.0, 0.0, 1.0)
    confidence = _clamp(fragment.confidence, 0.0, 1.0)

    if semantic_override is not None:
        semantic_sim = semantic_override
    elif query_embedding is not None and fragment.embedding:
        semantic_sim = _cosine_similarity(fragment.embedding, query_embedding)
    else:
        # No query context: use a neutral middle value so eviction is driven
        # by recency + frequency rather than an absent semantic signal.
        semantic_sim = 0.5

    components = {
        "semantic":   semantic_sim,
        "recency":    recency,
        "frequency":  frequency,
        "importance": importance,
        "feedback":   feedback,
        "confidence": confidence,
    }

    multiplier = EPISTEMIC_MULTIPLIER.get(
        getattr(fragment, "epistemic_class", "observed"), 1.00
    )

    if weight_store is not None:
        from .v4_reranker import compute_crs
        weights = weight_store.get_weights(fragment.project_id)
        crs_value = compute_crs(components, multiplier, weights)
    else:
        base_crs = (
            W_SEMANTIC   * semantic_sim +
            W_RECENCY    * recency      +
            W_FREQUENCY  * frequency    +
            W_IMPORTANCE * importance   +
            W_FEEDBACK   * feedback     +
            W_CONFIDENCE * confidence
        )
        crs_value = base_crs * multiplier

    return Score(
        value=_clamp(crs_value, 0.0, 1.0),
        components=components,
        multiplier=multiplier,
    )


def tier(crs: float) -> str:
    """Map a CRS value to its storage tier name."""
    if crs >= HOT_THRESHOLD:
        return "hot"
    if crs >= WARM_THRESHOLD:
        return "warm"
    return "cold"


# ── Private helpers ─────────────────────────────────────────────────────────

def _recency(last_accessed_iso: str) -> float:
    """Exponential decay from last_accessed timestamp."""
    try:
        ts = datetime.fromisoformat(last_accessed_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours_since = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 3600.0
    except (ValueError, TypeError):
        hours_since = 24.0  # default to 1-day-old if parse fails
    return math.exp(-RECENCY_DECAY_LAMBDA * hours_since)


def _frequency(access_count: int) -> float:
    """Logarithmic scaling; caps at 1.0 regardless of access_count."""
    return math.log1p(access_count) / math.log1p(MAX_EXPECTED_ACCESS)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
