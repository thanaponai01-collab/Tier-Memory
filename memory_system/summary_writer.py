"""
memory_system.summary_writer — LLM-synthesized project summary for the L1 layer.

Generates and persists a project-scoped summary that includes:
  • Synthesized description of the project's purpose and stack
  • Pinned facts (user-pinned fragments)
  • Recent corrections
  • Highest-confidence facts learned from sessions

The result is stored in project_summaries table and served as L1 project_memory
on subsequent retrieves, rather than recomputing on every query.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .llm import call_model
from .schema import Database

log = logging.getLogger("memoryd.summary")

_SUMMARY_SYSTEM = (
    "You are a technical writer synthesizing project knowledge for a software engineering assistant. "
    "Be concise, factual, and specific. Write in present tense."
)

_MAX_FACTS = 10
_MAX_EPISODES = 5
_MAX_CORRECTIONS = 5


def generate_project_summary(
    db: Database,
    project_id: str,
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[str]:
    """
    Synthesize a project summary from stored fragments and corrections.
    Persists the result to project_summaries. Returns the content string,
    or None if there is not enough material to summarize.
    """
    # Collect source material
    pinned = db.list_fragments(project_id, include_deprecated=False)
    pinned = [f for f in pinned if f.is_pinned]

    facts = db.list_fragments(project_id, category="fact", include_deprecated=False)
    facts.sort(key=lambda f: f.confidence, reverse=True)
    facts = facts[:_MAX_FACTS]

    episodes = db.list_fragments(project_id, category="episode", include_deprecated=False)
    episodes.sort(key=lambda e: e.last_accessed, reverse=True)
    episodes = episodes[:_MAX_EPISODES]

    corrections = db.get_corrections(project_id)
    corrections = corrections[:_MAX_CORRECTIONS]

    total_fragments = len(pinned) + len(facts) + len(episodes) + len(corrections)
    if total_fragments == 0:
        return None

    # Build context for the LLM
    sections: list[str] = []

    if pinned:
        items = "\n".join(f"• {f.content}" for f in pinned)
        sections.append(f"PINNED FACTS:\n{items}")

    if facts:
        items = "\n".join(
            f"• [{f.confidence:.2f}] {f.content}" for f in facts
        )
        sections.append(f"KNOWN FACTS (confidence score):\n{items}")

    if episodes:
        items = "\n".join(
            f"• {e.content[:200]}" for e in episodes
        )
        sections.append(f"RECENT EPISODES:\n{items}")

    if corrections:
        items = "\n".join(
            f"• was: {c.original_fact!r} → now: {c.corrected_fact!r}"
            for c in corrections
        )
        sections.append(f"CORRECTIONS:\n{items}")

    context = "\n\n".join(sections)

    prompt = f"""Below is accumulated knowledge about a software project.
Synthesize it into a concise project memory block (max 400 words) for a software engineering assistant.

Include:
1. A 2-3 sentence description of the project (if determinable from the facts)
2. PINNED FACTS section (verbatim list, if any exist)
3. KEY FACTS section (most important learnings, deduped)
4. CORRECTIONS section (active corrections, if any)

Omit any section that has no content. Write only what is known — do not invent.

SOURCE MATERIAL:
{context}"""

    try:
        summary = call_model(prompt, system=_SUMMARY_SYSTEM, model=model, max_tokens=600)
        summary = summary.strip()
        if not summary:
            return None
    except Exception as e:
        log.warning("project summary LLM call failed: %s", e)
        return None

    # Persist
    now = datetime.now(tz=timezone.utc).isoformat()
    db.upsert_project_summary(
        project_id=project_id,
        content=summary,
        generated_at=now,
        fragment_count=total_fragments,
    )
    log.info(
        "project summary regenerated: project=%s fragments=%d chars=%d",
        project_id, total_fragments, len(summary),
    )
    return summary
