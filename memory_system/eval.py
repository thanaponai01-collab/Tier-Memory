"""
memory_system.eval — the tasting spoon.

A tiny, offline, token-free retrieval-quality harness. You write down a handful
of (query -> the fragment that SHOULD win) pairs once; `mem eval` then replays
them through the REAL read path (the daemon's fused_retrieval, CRS re-rank,
semantic gate and token-budget knapsack — exactly what the agent sees) and
reports a single set of numbers you can watch before and after any change:

    recall@k     — for how many queries did a correct fragment land in the top k
    MRR          — mean reciprocal rank of the first correct hit (1.0 = always #1)

This is the instrument that makes the ~20 magic numbers in the system safe to
touch: change a threshold, re-run `mem eval`, see if the number moved the right
way. It calls retrieval in READ-ONLY mode so replaying the eval set never
touches access counts or pollutes the retrieval-event log that v4 trains on.

The eval set is plain JSON the user can edit by hand:

    {
      "cases": [
        {
          "query": "what database path does the system really use",
          "expect": "02 Memory Storage"            // substring, case-insensitive
        },
        {
          "query": "who is the user",
          "expect": ["taster", "doesn't code"],     // any-of substrings
          "project": "proj_f6f16b"                   // optional per-case override
        },
        {
          "query": "the durability invariant",
          "expect_id": "frag_abc123"                 // or pin an exact fragment id
        }
      ]
    }

A case matches when a returned fragment's id is in `expect_id` OR any `expect`
substring appears in its content. `expect` / `expect_id` may each be a string or
a list of strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_EVAL_FILENAME = "eval_set.json"
DEFAULT_K_VALUES = (1, 3, 5, 10)

# Written by `mem eval --init` so a first-time user has something to edit
# instead of a blank page. Intentionally generic — the real value comes from
# the user replacing these with queries that matter to them.
STARTER_TEMPLATE = {
    "_comment": (
        "Each case is a query plus what a correct answer must contain. "
        "'expect' is a case-insensitive substring (or list of substrings, any of "
        "which counts as a hit) that should appear in a retrieved fragment. "
        "'expect_id' pins an exact fragment id instead. 'project' optionally "
        "overrides the project scope for that one case. Edit freely, then run "
        "'mem eval'."
    ),
    "cases": [
        {"query": "what am I really trying to build", "expect": "flywheel"},
        {"query": "how should you work with me", "expect": "taster"},
        {"query": "where is the memory database stored", "expect": "Memory Storage"},
    ],
}


# ── Eval set loading ──────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    query: str
    expect: list[str] = field(default_factory=list)        # substrings (any-of)
    expect_id: list[str] = field(default_factory=list)     # exact fragment ids
    project: Optional[str] = None                          # per-case scope override

    def is_specified(self) -> bool:
        return bool(self.expect or self.expect_id)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def parse_cases(raw: Any) -> list[EvalCase]:
    """Turn the loaded JSON into EvalCase objects. Raises ValueError on a
    malformed file so the user gets a clear message rather than a silent
    half-empty eval."""
    if isinstance(raw, dict):
        items = raw.get("cases")
    elif isinstance(raw, list):
        items = raw
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("eval set must be a JSON object with a 'cases' list "
                         "(or a top-level list of cases)")

    cases: list[EvalCase] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"case #{i + 1} is not an object")
        query = str(item.get("query", "")).strip()
        if not query:
            raise ValueError(f"case #{i + 1} is missing a 'query'")
        case = EvalCase(
            query=query,
            expect=_as_str_list(item.get("expect")),
            expect_id=_as_str_list(item.get("expect_id")),
            project=item.get("project"),
        )
        if not case.is_specified():
            raise ValueError(
                f"case #{i + 1} ({query!r}) has neither 'expect' nor 'expect_id' — "
                "there's nothing to check it against"
            )
        cases.append(case)
    if not cases:
        raise ValueError("eval set contains no cases")
    return cases


def load_eval_set(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_cases(raw)


def write_starter_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(STARTER_TEMPLATE, indent=2), encoding="utf-8")


# ── Scoring (pure functions — no I/O, fully unit-testable) ─────────────────────

def first_match_rank(fragments: list[dict], case: EvalCase) -> Optional[int]:
    """Return the 1-based rank of the first fragment in `fragments` that
    satisfies `case`, or None if none do. `fragments` is the CRS-ordered list
    the read path returned, so the index here IS the CRS rank."""
    expect_lower = [s.lower() for s in case.expect]
    expect_ids = set(case.expect_id)
    for rank, frag in enumerate(fragments, start=1):
        fid = str(frag.get("id", ""))
        if fid in expect_ids:
            return rank
        content = str(frag.get("content", "")).lower()
        if any(sub in content for sub in expect_lower):
            return rank
    return None


@dataclass
class CaseResult:
    case: EvalCase
    rank: Optional[int]            # 1-based rank of first hit, or None
    returned: int                  # how many fragments the read path returned

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


@dataclass
class EvalReport:
    results: list[CaseResult]
    k_values: tuple[int, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def matched(self) -> int:
        return sum(1 for r in self.results if r.rank is not None)

    def recall_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit_at(k)) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "matched": self.matched,
            "mrr": round(self.mrr, 4),
            "recall": {f"@{k}": round(self.recall_at(k), 4) for k in self.k_values},
            "cases": [
                {
                    "query": r.case.query,
                    "rank": r.rank,
                    "returned": r.returned,
                    "reciprocal_rank": round(r.reciprocal_rank, 4),
                }
                for r in self.results
            ],
        }


def score(case_outputs: list[tuple[EvalCase, list[dict]]],
          k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> EvalReport:
    """Given (case, returned_fragments) pairs, compute the report. Separated
    from retrieval so it can be tested without a daemon."""
    results = [
        CaseResult(case=case, rank=first_match_rank(frags, case), returned=len(frags))
        for case, frags in case_outputs
    ]
    return EvalReport(results=results, k_values=tuple(k_values))
