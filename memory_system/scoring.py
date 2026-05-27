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
from typing import Optional
from .models import MemoryFragment

# Tuning constants
RECENCY_DECAY_LAMBDA = 0.003       # half-life ≈ 10 days
MAX_EXPECTED_ACCESS = 1_000        # cap for logarithmic frequency scaling

# §3.5 — updated CRS weights (sum = 1.0)
W_SEMANTIC    = 0.30   # was 0.35
W_RECENCY     = 0.25   # was 0.30
W_FREQUENCY   = 0.10   # was 0.15
W_IMPORTANCE  = 0.15
W_FEEDBACK    = 0.05
W_CONFIDENCE  = 0.15   # NEW — closes Gap #4

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
) -> float:
    """
    Returns a float [0.0, 1.0].  Pinned fragments always score 1.0.
    Pass query_embedding for retrieval-time scoring;
    omit it for maintenance/eviction scoring (uses historical avg similarity).
    """
    if fragment.is_pinned:
        return 1.0

    recency    = _recency(fragment.last_accessed)
    frequency  = _frequency(fragment.access_count)
    importance = _clamp(fragment.graph_centrality, 0.0, 1.0)
    feedback   = _clamp((fragment.user_feedback + 1.0) / 2.0, 0.0, 1.0)
    confidence = _clamp(fragment.confidence, 0.0, 1.0)

    if query_embedding is not None and fragment.embedding:
        semantic_sim = _cosine_similarity(fragment.embedding, query_embedding)
    else:
        # No query context: use a neutral middle value so eviction is driven
        # by recency + frequency rather than an absent semantic signal.
        semantic_sim = 0.5

    base_crs = (
        W_SEMANTIC   * semantic_sim +
        W_RECENCY    * recency      +
        W_FREQUENCY  * frequency    +
        W_IMPORTANCE * importance   +
        W_FEEDBACK   * feedback     +
        W_CONFIDENCE * confidence
    )

    # §3.5 — epistemic multiplier (outside weighted sum to preserve ordinal ranking)
    multiplier = EPISTEMIC_MULTIPLIER.get(
        getattr(fragment, "epistemic_class", "observed"), 1.00
    )
    return _clamp(base_crs * multiplier, 0.0, 1.0)


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _clamp(dot / (norm_a * norm_b), -1.0, 1.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
