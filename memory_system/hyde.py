"""
HyDE — Hypothetical Document Embeddings (query expansion for retrieval).

The problem this solves: a natural question ("what am I really trying to build")
embeds far from the memory that answers it ("the conversation clarified the
flywheel north star"), because the question and the stored note are written in
different language. Neither vector nor keyword search bridges that gap, so the
right memory never reaches the candidate set — no amount of re-ranking can
recover it (see project_retrieval_phrasing_gap).

HyDE bridges it: before searching, ask a cheap model to sketch a brief plausible
ANSWER to the question, then embed THAT for the vector lane. The hypothetical
answer is written in the vocabulary the real memory uses, so it lands in the
right neighbourhood. The original question is still used verbatim for the BM25
keyword lane — HyDE only steers the meaning search.

Design choices that keep this safe on the hot read path:
  • cheap role + short output (this fires before every answer via the read reflex)
  • best-effort: any failure (model offline, timeout) returns None and the caller
    falls back to embedding the raw query — retrieval never breaks
  • small LRU cache keyed by query text so repeated questions cost nothing
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .llm import LLMRouter

_SYSTEM = (
    "You are recalling from a single user's personal project notes. Given their "
    "question, write a brief, plausible answer the way such a note would be "
    "written — concrete, declarative, using the specific nouns and project "
    "vocabulary the note itself would use. 1-2 sentences. No preamble, no "
    "hedging, no 'I don't know'. If unsure, still write the most likely note."
)

# Module-level cache shared across calls. Bounded; oldest evicted first.
_CACHE_MAX = 512
_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: str) -> Optional[str]:
    with _cache_lock:
        val = _cache.get(key)
        if val is not None:
            _cache.move_to_end(key)
        return val


def _cache_put(key: str, val: str) -> None:
    with _cache_lock:
        _cache[key] = val
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def hypothetical_document(
    query: str,
    router: "LLMRouter",
    role: str = "cheap",
    max_tokens: int = 120,
) -> Optional[str]:
    """Return a short hypothetical answer to `query` to embed in place of the raw
    question, or None if generation fails or yields nothing usable.

    The returned text is intended to be PREPENDED to the original query before
    embedding (so the real question's words still contribute), not to replace it
    wholesale — the caller decides how to combine.
    """
    q = (query or "").strip()
    if not q:
        return None
    key = f"{role}\x00{q}"   # role changes the document, so it's part of the key
    cached = _cache_get(key)
    if cached is not None:
        return cached or None  # cached "" means "tried, gave nothing"
    doc = ""
    try:
        raw = router.call(role, q, system=_SYSTEM, max_tokens=max_tokens)
        doc = _clean(raw)
    except Exception:
        # Best-effort: never let expansion break retrieval. Cache the empty
        # result briefly so a hard-down model isn't hammered once per query.
        doc = ""
    _cache_put(key, doc)
    return doc or None


def _clean(text: str) -> str:
    import re
    # Strip <think>...</think> (qwen3 etc.) and surrounding whitespace.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    return text
