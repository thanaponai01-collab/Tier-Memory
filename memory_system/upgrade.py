"""
upgrade.py — the flywheel's "a smarter model is available" detector + safe-mix gate.

The flywheel promise: when you start running a smarter model, your stored memory
should be able to level up to match it (re-distill / re-judge with the better
model). Two pieces live here:

  1. DETECT — compare the model your memory was *built with* (the producer_model
     stamped on each fragment) against the model you're *configured to run now*.
     If the current model outranks what built the store, an upgrade is available.

  2. SAFE MIX — never auto-run an expensive reprocess. detect_upgrade() is
     read-only; it returns *what* would be reprocessed and *how much* it would
     cost, so the CLI can show the plan and require an explicit human "yes"
     before turning the crank (see cli.py `mem upgrade`).

Model ranking is an explicit, ordered list (low capability -> high). Matching is
prefix-based so a dated id like "claude-haiku-4-5-20251001" still resolves. A
NULL producer (pre-provenance fragments) ranks below every real model, so the
1789 fragments written before stamping existed are correctly seen as "behind."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema import Database

# Capability order, weakest first. Add new models here as they ship.
# Matching is by prefix after normalization, so undated stems work.
_RANKED_MODELS: list[str] = [
    "ollama/qwen3:8b",
    "ollama/qwen3",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
]

_RANK_NULL = -1      # no recorded producer (pre-provenance) — behind everything real
_RANK_UNKNOWN = -1   # a model we don't recognize — treat conservatively as behind


def _normalize(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def model_rank(name: Optional[str]) -> int:
    """Capability rank for a model id. Higher beats lower.

    NULL/empty -> _RANK_NULL. Unrecognized -> _RANK_UNKNOWN. Known -> its index
    in _RANKED_MODELS. Matching is prefix-based against the longest entries first
    so 'claude-opus-4-8-20260115' resolves to 'claude-opus-4-8'.
    """
    norm = _normalize(name)
    if not norm:
        return _RANK_NULL
    # Longest entries first so 'claude-opus-4-8' wins over a shorter prefix.
    for entry in sorted(_RANKED_MODELS, key=len, reverse=True):
        if norm.startswith(entry) or entry.startswith(norm):
            return _RANKED_MODELS.index(entry)
    return _RANK_UNKNOWN


@dataclass
class UpgradeStatus:
    current_model: str                 # what new memory would be produced with now
    current_rank: int
    stored_model: Optional[str]        # the WEAKEST producer found in the store
    stored_rank: int
    upgrade_available: bool
    fragments_total: int               # active (non-deprecated) fragments
    fragments_behind: int              # active fragments produced by a weaker/NULL model
    low_confidence_facts: int          # facts < 0.70 a resynthesis pass would target
    cold_sessions: int                 # sessions a cold replay could re-judge (capped)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "current_model": self.current_model,
            "current_rank": self.current_rank,
            "stored_model": self.stored_model,
            "stored_rank": self.stored_rank,
            "upgrade_available": self.upgrade_available,
            "fragments_total": self.fragments_total,
            "fragments_behind": self.fragments_behind,
            "low_confidence_facts": self.low_confidence_facts,
            "cold_sessions": self.cold_sessions,
            "note": self.note,
        }


def detect_upgrade(
    db: Database,
    current_model: str,
    cold_sessions_limit: int = 500,
    project_id: Optional[str] = None,
) -> UpgradeStatus:
    """Read-only. Compare the current production model against what built the store.

    `current_model` is the model the pipeline would stamp on new fragments today
    (i.e. LLMRouter.model_for('cheap')). Never mutates the database.
    """
    where = "project_id = ?" if project_id else "1=1"
    params: tuple = (project_id,) if project_id else ()
    current_rank = model_rank(current_model)

    # Producer distribution over ACTIVE fragments.
    rows = db.fetchall(
        f"""SELECT producer_model AS m, COUNT(*) AS n
            FROM memory_fragments
            WHERE is_deprecated = 0 AND {where}
            GROUP BY producer_model""",
        params,
    )
    fragments_total = sum(r["n"] for r in rows)

    # "Behind" = produced by a model that ranks below the current one (NULL counts).
    fragments_behind = sum(
        r["n"] for r in rows if model_rank(r["m"]) < current_rank
    )

    # The weakest producer actually present — what a re-judgment would lift from.
    stored_model: Optional[str] = None
    stored_rank = current_rank
    if rows:
        weakest = min(rows, key=lambda r: model_rank(r["m"]))
        stored_model = weakest["m"]
        stored_rank = model_rank(weakest["m"])

    low_conf = db.fetchone(
        f"""SELECT COUNT(*) AS n FROM memory_fragments
            WHERE is_deprecated = 0 AND category = 'fact'
              AND confidence < 0.70 AND {where}""",
        params,
    )
    low_confidence_facts = low_conf["n"] if low_conf else 0

    sess = db.fetchone(
        f"SELECT COUNT(*) AS n FROM sessions WHERE {where}", params
    )
    cold_sessions = min(sess["n"] if sess else 0, cold_sessions_limit)

    upgrade_available = fragments_behind > 0 and current_rank > stored_rank

    if upgrade_available:
        note = (
            f"{fragments_behind} of {fragments_total} active fragments were built "
            f"by a weaker model ({stored_model or 'unknown/none'}); re-judging with "
            f"{current_model} would level them up."
        )
    elif current_rank == _RANK_UNKNOWN:
        note = (
            f"Current model {current_model!r} is not in the known capability "
            f"ranking - can't tell if it's an upgrade. Add it to _RANKED_MODELS."
        )
    else:
        note = f"Memory is already current for {current_model}; nothing to gain."

    return UpgradeStatus(
        current_model=current_model,
        current_rank=current_rank,
        stored_model=stored_model,
        stored_rank=stored_rank,
        upgrade_available=upgrade_available,
        fragments_total=fragments_total,
        fragments_behind=fragments_behind,
        low_confidence_facts=low_confidence_facts,
        cold_sessions=cold_sessions,
        note=note,
    )
