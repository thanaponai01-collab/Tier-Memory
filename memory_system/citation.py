"""
memory_system.citation — the outcome-loop "was this memory actually used?" detector.

The read reflex injects fragments into a turn; this decides which of them the
model's ANSWER actually drew on. It's a lexical-grounding proxy (the honest kind
RAG eval uses): a fragment counts as cited when the answer contains a meaningful
slice of the fragment's distinctive content that the USER did not already supply
in the prompt. Subtracting the prompt is the crucial part — otherwise a fragment
would get credit for words the user themselves typed.

This is deliberately stricter and more load-bearing than v4's existing re-touch
proxy ("the fragment got retrieved again soon after"): here the fragment's own
content has to show up in the produced answer, novel to the question.

Pure functions, no I/O — so the Stop hook and the test suite share one source of
truth and it's unit-testable without a daemon or a transcript.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

# Common words carry no attribution signal — drop them so overlap reflects
# distinctive content, not shared grammar. Kept deliberately small; the len>=4
# floor already removes most function words.
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
    "was", "were", "are", "will", "would", "should", "could", "what", "when",
    "where", "which", "who", "why", "how", "into", "than", "then", "them",
    "they", "their", "there", "these", "those", "your", "you", "our", "about",
    "not", "but", "all", "any", "can", "use", "used", "using", "via", "per",
    "out", "off", "its", "it's", "also", "more", "most", "some", "such", "only",
    "just", "like", "over", "under", "after", "before", "because", "been",
    "being", "does", "done", "doing", "make", "made", "need", "needs", "want",
})

# Defaults tuned to be conservative: better to miss a real citation than to
# credit a fragment that merely shares a topic with the answer.
MIN_OVERLAP = 0.15   # fraction of the fragment's distinctive tokens echoed
MIN_NOVEL = 4        # absolute floor on novel shared tokens (anti-coincidence)


def content_tokens(text: str, min_len: int = 4) -> set[str]:
    """Distinctive lowercased word tokens: length >= min_len, no stopwords."""
    if not text:
        return set()
    return {
        t for t in _WORD.findall(text.lower())
        if len(t) >= min_len and t not in _STOPWORDS
    }


def detect_cited(
    injected: dict[str, str],
    prompt: str,
    response: str,
    min_overlap: float = MIN_OVERLAP,
    min_novel: int = MIN_NOVEL,
) -> list[str]:
    """
    injected : {fragment_id -> fragment content} that the read reflex surfaced.
    prompt   : the user's ORIGINAL prompt (NOT the injected block — pass the
               clean prompt so the model can't get credit for the user's words).
    response : the assistant's answer for that turn.

    Returns the ids of fragments whose distinctive content surfaced in the
    answer, novel to the prompt, above both thresholds.
    """
    resp_tok = content_tokens(response)
    if not resp_tok:
        return []
    prompt_tok = content_tokens(prompt)

    cited: list[str] = []
    for fid, content in injected.items():
        frag_tok = content_tokens(content)
        if not frag_tok:
            continue
        # Fragment content that made it into the answer but wasn't in the question.
        novel = (frag_tok & resp_tok) - prompt_tok
        if len(novel) >= min_novel and (len(novel) / len(frag_tok)) >= min_overlap:
            cited.append(fid)
    return cited
