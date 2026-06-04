"""
Token / cost accounting for the daemon — an honest ledger, not a headline.

Three quantities are reported and kept strictly separate, because conflating
them is exactly how a dashboard ends up lying:

  harness cache savings  — prompt-cache reads are input tokens billed at the
              reduced cache rate (Anthropic charges cache reads at ~0.1x of the
              base input rate, so ~0.9x is genuinely saved). This is REAL and
              MEASURED when sessions carry token counts forwarded by the ingest
              hook — but it is produced by the Claude Code *harness's* prompt
              caching, which happens with or without this memory system. So it
              is reported as harness-attributable, NOT as value the memory
              system created. The cache hit rate is the one lever memory does
              influence (a stable memory prefix keeps the cache warm).

  memory injection cost  — the read reflex (UserPromptSubmit hook) injects a
              `<recalled_memory>` block before the model answers. Those tokens
              are the only flow this system *directly* adds to the bill. It is a
              COST, and it is fully memory-attributable. Sourced from
              retrieval_events, so it is measured, not assumed.

  net                    — harness cache savings minus memory injection cost.
              The honest "what did running this system net you, in raw tokens,
              inside Claude Code" figure. (The system's real payoff — not
              re-explaining context across sessions — is a counterfactual we
              can't measure from logs, so we don't pretend to.)

  estimated  — fallback only: with no token data at all we project from the
              volume of compressed memory we hold, using an explicit assumed
              ratio that travels with the result (basis="estimated").

The North Star is honest, lab-grade metrics; a fabricated or mis-attributed
headline undermines exactly that, so `basis` and every assumed rate are part of
the contract and surfaced in the payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema import Database

# One stored memory token is assumed to stand in for ~this many raw transcript
# tokens. Used ONLY for the estimated branch, and surfaced in the payload.
ASSUMED_COMPRESSION_RATIO = 8.0

# Blended $/Mtok used to translate tokens into dollars. A single documented rate
# (model_id is unknown on most sessions, so per-model pricing isn't credible).
BLENDED_COST_PER_MTOK_USD = 5.40

# Anthropic bills a prompt-cache read at ~0.1x the base input rate, so the saving
# versus paying full price is ~0.9x. Applying this is why the headline is lower
# (and truer) than "cache_read_tok x full rate".
CACHE_READ_DISCOUNT = 0.9


def compute_savings(db: "Database") -> dict:
    """Return an honest, separated savings ledger with an explicit `basis`.

    Keys (stable for the CLI + dashboard):
      total_sessions, total_turns, total_input_tokens, total_output_tokens
      cache_read_tok, cache_hit_rate
      basis ("measured" | "estimated"), assumed_compression_ratio (None if measured)
      blended_cost_per_mtok, cache_read_discount

      Harness-attributable (measured branch):
        cache_savings_usd        — cache_read_tok * discount * rate
      Memory-attributable (always computed):
        injection_events, injected_tokens, injection_cost_usd
        net_usd                  — cache_savings_usd - injection_cost_usd

      Back-compat aliases (so older callers don't break):
        tokens_saved   = cache_read_tok (measured) | projected tokens (estimated)
        cost_saved_usd = cache_savings_usd (measured) | projected cost (estimated)

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

    # Memory's true, directly-attributable cost: read-reflex injected tokens.
    inj = db.injected_token_cost()
    injected_tokens = inj["injected_tokens"]
    injection_cost_usd = round(
        (injected_tokens / 1_000_000) * BLENDED_COST_PER_MTOK_USD, 2
    )

    if total_input_tokens > 0 or cache_read > 0:
        # Measured: prompt-cache reads, billed at the reduced cache rate.
        basis = "measured"
        assumed_ratio = None
        cache_savings_usd = round(
            (cache_read / 1_000_000) * BLENDED_COST_PER_MTOK_USD * CACHE_READ_DISCOUNT,
            2,
        )
        tokens_saved = cache_read
        cost_saved_usd = cache_savings_usd
        net_usd = round(cache_savings_usd - injection_cost_usd, 2)
    else:
        # Estimated: no real token data yet — project from compressed volume.
        basis = "estimated"
        assumed_ratio = ASSUMED_COMPRESSION_RATIO
        row = db.fetchone(
            "SELECT COALESCE(SUM(token_count), 0) AS n "
            "FROM memory_fragments WHERE is_deprecated=0"
        )
        fragment_tokens = int(row["n"]) if row else 0
        projected = int(fragment_tokens * (assumed_ratio - 1.0))
        cache_savings_usd = round(
            (projected / 1_000_000) * BLENDED_COST_PER_MTOK_USD, 2
        )
        tokens_saved = projected
        cost_saved_usd = cache_savings_usd
        net_usd = round(cache_savings_usd - injection_cost_usd, 2)

    return {
        "total_sessions": total_sessions,
        "total_turns": total_turns,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cache_read_tok": cache_read,
        "cache_hit_rate": cache["cache_hit_rate"],
        "basis": basis,
        "assumed_compression_ratio": assumed_ratio,
        "blended_cost_per_mtok": BLENDED_COST_PER_MTOK_USD,
        "cache_read_discount": CACHE_READ_DISCOUNT,
        # Harness-attributable
        "cache_savings_usd": cache_savings_usd,
        # Memory-attributable
        "injection_events": inj["events"],
        "injected_tokens": injected_tokens,
        "injection_cost_usd": injection_cost_usd,
        "net_usd": net_usd,
        # Back-compat aliases
        "tokens_saved": tokens_saved,
        "cost_saved_usd": cost_saved_usd,
        "sessions": sessions,
    }
