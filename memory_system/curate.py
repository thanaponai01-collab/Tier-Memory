"""
curate — absorb hand-written curated notes into the searchable fragment store.

The problem (see project_retrieval_phrasing_gap): the crisp answers to the most
important questions — what the user is building, who they are, what the rules are
— live in hand-curated markdown notes, but the searchable store only holds
diffuse conversational summaries. So "what am I really trying to build" can't find
the flywheel note even though it exists. This module turns those notes into
high-confidence fragments written in the user's own words, so identity/goal
questions match directly.

Pure functions here (no daemon, no I/O beyond reading the given files) so they're
unit-testable; the CLI command does the daemon round-trip via the existing import
op (which upserts + embeds + indexes inside the daemon, the only process that may
touch the live vector index).

Fragments are marked category="fact", source_type="user_explicit", confidence=1.0
and given a stable id (cur_<hash>) so re-running is idempotent — already-imported
notes are skipped by the import op. They are NOT pinned: pinning would force them
to the top of EVERY query and crowd out conversational memory. High confidence +
importance lets them win the queries they actually match while the semantic gate
keeps them out of unrelated ones.

Alias fragments: a note can declare natural question phrasings in its frontmatter:

    questions:
      - what am I really trying to build
      - what is the main thing I am building

Each question produces an additional fragment whose content leads with
"Q: <question>" followed by the note content. This bridges the query↔note
phrasing gap without any LLM call (deterministic, no latency on the hot path).
Alias ids are stable: cur_<sha1(name:q0)>, cur_<sha1(name:q1)>, etc.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

# Notes that are indexes/scaffolding, not facts — never absorb these.
_SKIP_NAMES = {"MEMORY.md", "README.md"}

_CATEGORY = "fact"
_CONFIDENCE = 1.0
_IMPORTANCE = 0.8


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---`-delimited YAML-ish frontmatter block from the body.
    Returns (fields, body). Handles scalar values (key: value) and simple
    YAML lists (key:\\n  - item). No YAML dependency; never throws on odd notes."""
    fields: dict = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return fields, text.strip()
    block, body = m.group(1), m.group(2)
    current_list_key: Optional[str] = None
    for line in block.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if v:
                fields[k] = v
                current_list_key = None
            else:
                fields[k] = []
                current_list_key = k
        elif line.startswith("  - ") and current_list_key is not None:
            item = line[4:].strip().strip("\"'")
            if isinstance(fields.get(current_list_key), list):
                fields[current_list_key].append(item)
        else:
            current_list_key = None
    return fields, body.strip()


def _make_fragment(fid: str, project_id: str, content: str, note_name: str) -> dict:
    return {
        "id": fid,
        "project_id": project_id,
        "scope": "project",
        "category": _CATEGORY,
        "content": content,
        "token_count": max(1, len(content) // 4),
        "confidence": _CONFIDENCE,
        "graph_centrality": _IMPORTANCE,
        "source_type": "user_explicit",
        "is_pinned": False,
        "metadata_json": f'{{"curated": true, "note": "{note_name}"}}',
    }


def note_to_fragments(
    path: Path,
    project_id: str,
    text: Optional[str] = None,
) -> list[dict]:
    """Turn one curated note into fragment dicts: one main + one per alias question.
    Returns [] if there is no usable content."""
    raw = text if text is not None else path.read_text(encoding="utf-8")
    fields, body = _strip_frontmatter(raw)
    desc = fields.get("description", "").strip()
    name = fields.get("name", path.stem).strip()
    questions = fields.get("questions", [])
    if isinstance(questions, str):
        questions = [q.strip() for q in questions.split(",") if q.strip()]

    parts = [p for p in (desc, body) if p]
    content = "\n\n".join(parts).strip()
    if not content:
        return []

    fid = "cur_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    frags = [_make_fragment(fid, project_id, content, path.name)]

    # Alias fragments use description only (not body) so they:
    # (a) stay well under Ollama's embed length limit (~7K chars),
    # (b) score high in BM25 (short doc → high TF for exact question terms),
    # (c) don't pollute ranking for unrelated queries (minimal overlap with body).
    alias_body = desc or content[:300]
    for i, q in enumerate(questions):
        q = q.strip()
        if not q:
            continue
        alias_content = f"Q: {q}\n\n{alias_body}"
        alias_id = "cur_" + hashlib.sha1(f"{name}:q{i}".encode("utf-8")).hexdigest()[:16]
        frags.append(_make_fragment(alias_id, project_id, alias_content, path.name))

    return frags


def note_to_fragment(
    path: Path,
    project_id: str,
    text: Optional[str] = None,
) -> Optional[dict]:
    """Return the main fragment for a note, or None. Compat wrapper for tests/callers
    that don't need alias fragments."""
    frags = note_to_fragments(path, project_id, text)
    return frags[0] if frags else None


def build_fragments_from_dir(notes_dir: Path, project_id: str) -> list[dict]:
    """Read every curated note in a directory into fragment dicts (main + aliases)."""
    frags: list[dict] = []
    for path in sorted(notes_dir.glob("*.md")):
        if path.name in _SKIP_NAMES:
            continue
        try:
            note_frags = note_to_fragments(path, project_id)
        except OSError:
            continue
        frags.extend(note_frags)
    return frags
