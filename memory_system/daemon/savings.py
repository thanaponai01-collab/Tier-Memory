"""
Token / cost savings accounting for the daemon.

Two sources of "savings", kept strictly separate so a projection is never
reported as a measurement:

  measured  — prompt-cache reads are input tokens the API genuinely did not
              re-charge for. When sessions carry real token counts (forwarded by
              the ingest hook), this is an observed, defensible number.
  estimated — with no token data we can only *project* savings from the volume
              of compressed memory we hold, using an explicit assumed ratio.
              This is a projection, labelled `basis="estimated"`, and the ratio
              it assumes travels with the result so it can never masquerade as
              a measured figure.

The North Star is honest, lab-grade metrics; a fabricated headline number
undermines exactly that, so the `basis` flag is part of the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema import Database

# One stored memory token is assumed to stand in for ~this many raw transcript
# tokens. Used ONLY for the estimated branch, and surfaced in the payload.
ASSUMED_COMPRESSION_RATIO = 8.0

# Blended $/Mtok used to translate saved tokens into a headline dollar figure.
BLENDED_COST_PER_MTOK_USD = 5.40


def compute_savings(db: "Database") -> dict:
    """Return savings analytics for every session, with an explicit `basis`.

    Keys (stable for the CLI + dashboard):
      total_sessions, total_turns, total_input_tokens, total_output_tokens
      cache_read_tok, cache_hit_rate
      tokens_saved, cost_saved_usd
      basis ("measured" | "estimated"), assumed_compression_ratio (None if measured)
      sessions (raw rows, newest first)
    """
    sessions = [
        dict(r) for r in db.fetchall(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        )
    ]
    total_sessions = len(sessions)
    total_turns = sum(s.get("turn_count", 0) or 0 for s in sessions)
    total_input_tokens = sum(s.get("cost_input_tok", 0) or 0 for s in sessions)
    total_output_tokens = sum(s.get("cost_output_tok", 0) or 0 for s in sessions)

    cache = db.cache_stats()
    cache_read = cache["cache_read_tok"]

    if total_input_tokens > 0 or cache_read > 0:
        # Measured: prompt-cache reads are tokens we genuinely did not re-pay for.
        basis = "measured"
        assumed_ratio = None
        tokens_saved = cache_read
    else:
        # Estimated: no real token data yet — project from compressed volume.
        basis = "estimated"
        assumed_ratio = ASSUMED_COMPRESSION_RATIO
        row = db.fetchone(
            "SELECT COALESCE(SUM(token_count), 0) AS n "
            "FROM memory_fragments WHERE is_deprecated=0"
        )
        fragment_tokens = int(row["n"]) if row else 0
        tokens_saved = int(fragment_tokens * (assumed_ratio - 1.0))

    cost_saved_usd = round((tokens_saved / 1_000_000) * BLENDED_COST_PER_MTOK_USD, 2)

    return {
        "total_sessions": total_sessions,
        "total_turns": total_turns,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cache_read_tok": cache_read,
        "cache_hit_rate": cache["cache_hit_rate"],
        "tokens_saved": tokens_saved,
        "cost_saved_usd": cost_saved_usd,
        "basis": basis,
        "assumed_compression_ratio": assumed_ratio,
        "sessions": sessions,
    }
