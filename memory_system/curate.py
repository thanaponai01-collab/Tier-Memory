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

Fragments are marked category="profile", source_type="user_explicit",
confidence=1.0 and given a stable id (cur_<hash>) so re-running is idempotent —
already-imported notes are skipped by the import op. They are NOT pinned: pinning
would force them to the top of EVERY query and crowd out conversational memory.
High confidence + importance lets them win the queries they actually match while
the semantic gate keeps them out of unrelated ones.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

# Notes that are indexes/scaffolding, not facts — never absorb these.
_SKIP_NAMES = {"MEMORY.md", "README.md"}

# 'fact' is the only valid category for declarative notes (the schema enforces
# category IN fact|episode|preference|correction). Curated fragments stay
# identifiable by their cur_<hash> id and the "curated": true metadata flag, not
# by a custom category.
_CATEGORY = "fact"
_CONFIDENCE = 1.0               # user-authored = authoritative
_IMPORTANCE = 0.8              # graph_centrality → boosts the CRS importance term


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---`-delimited YAML-ish frontmatter block from the body.
    Returns (fields, body). Parsing is deliberately minimal (key: value) so this
    has no YAML dependency and never throws on odd notes."""
    fields: dict[str, str] = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return fields, text.strip()
    block, body = m.group(1), m.group(2)
    for line in block.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    return fields, body.strip()


def note_to_fragment(
    path: Path,
    project_id: str,
    text: Optional[str] = None,
) -> Optional[dict]:
    """Turn one curated markdown note into a fragment dict, or None if it has no
    usable content. The fragment content leads with the note's one-line
    description (the crispest phrasing) followed by the body, so the embedding
    captures both the summary and the detail in the user's own words."""
    raw = text if text is not None else path.read_text(encoding="utf-8")
    fields, body = _strip_frontmatter(raw)
    desc = fields.get("description", "").strip()
    name = fields.get("name", path.stem).strip()
    # Lead with the description (best phrasing), then the body.
    parts = [p for p in (desc, body) if p]
    content = "\n\n".join(parts).strip()
    if not content:
        return None
    fid = "cur_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
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
        "metadata_json": f'{{"curated": true, "note": "{path.name}"}}',
    }


def build_fragments_from_dir(notes_dir: Path, project_id: str) -> list[dict]:
    """Read every curated note in a directory into fragment dicts."""
    frags: list[dict] = []
    for path in sorted(notes_dir.glob("*.md")):
        if path.name in _SKIP_NAMES:
            continue
        try:
            frag = note_to_fragment(path, project_id)
        except OSError:
            continue
        if frag:
            frags.append(frag)
    return frags
