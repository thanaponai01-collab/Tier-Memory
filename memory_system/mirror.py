"""
memory_system.mirror — the intent mirror.

Reflects the gap between what the user SAID they were trying to do (goals)
and what they ACTUALLY did (recent observed behavior — episodes/facts the
pipeline already distilled from their sessions).

On-demand only: nothing here runs on a timer. The user picks the mirror up
when they want to look. All the value is in that one moment, so the reflection
is synthesized by an LLM into plain language, not raw rows.

Reads the DB directly (the goals table is brand-new, so there is no daemon
coordination to do). The reflection is synthesized through the same LLMRouter
the rest of the system uses, so it honours whatever model the user has
configured (local Ollama by default) rather than hard-coding a provider.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .config import load_config
from .llm import LLMRouter
from .models import Goal
from .schema import Database


# Reflection runs through the 'medium' role of the user's configured LLMRouter.
# A reflective synthesis wants more than the cheap extraction model, and 'medium'
# maps to the user's local Ollama chat model by default — available and free.
_MIRROR_ROLE = "medium"
_OBSERVATION_LIMIT = 40   # most-recent behaviour fragments to weigh

_MIRROR_SYSTEM = (
    "You are a clear-eyed mirror. You compare what a person SAID they were "
    "trying to do against what they ACTUALLY did, and you reflect the gap back "
    "honestly. You never flatter and you never invent progress. You speak in "
    "plain language with no jargon. The person is the authority on their own "
    "intent: when their actions diverge from a stated goal you point it out as "
    "an observation and a question, never as a correction."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_goal_id() -> str:
    return "g_" + uuid.uuid4().hex[:8]


def _open_db() -> Database:
    cfg = load_config()
    db = Database(cfg.storage.db_path)
    db.connect()
    # The daemon may hold a write lock; wait briefly rather than erroring.
    db.execute("PRAGMA busy_timeout=5000")
    return db


# ── Goal management ──────────────────────────────────────────────────────────

def add_goal(project_id: str, statement: str) -> Goal:
    db = _open_db()
    try:
        goal = Goal(
            id=new_goal_id(),
            project_id=project_id,
            statement=statement.strip(),
            created_at=_now(),
        )
        db.insert_goal(goal)
        return goal
    finally:
        db.close()


def list_goals(project_id: str, status: str | None = "open") -> list[Goal]:
    db = _open_db()
    try:
        return db.list_goals(project_id, status=status)
    finally:
        db.close()


def close_goal(project_id: str, goal_id: str) -> bool:
    db = _open_db()
    try:
        return db.close_goal(goal_id, _now())
    finally:
        db.close()


# ── The reflection ───────────────────────────────────────────────────────────

def _gather_observations(db: Database, project_id: str, limit: int) -> list[dict]:
    rows = db.fetchall("""
        SELECT category, created_at, content
        FROM memory_fragments
        WHERE project_id = ?
          AND is_deprecated = 0
          AND category IN ('episode', 'fact')
        ORDER BY created_at DESC
        LIMIT ?
    """, (project_id, limit))
    return [dict(r) for r in rows]


def _build_prompt(goals: list[Goal], observations: list[dict]) -> str:
    goal_lines = "\n".join(
        f"{i}. {g.statement}" for i, g in enumerate(goals, 1)
    )
    obs_lines = "\n".join(
        f"- ({o['category']}) {o['content'].strip()}" for o in observations
    )
    return f"""STATED GOALS — what they said they're trying to do:
{goal_lines}

WHAT THEY ACTUALLY DID RECENTLY — observed from their work, most recent first:
{obs_lines}

For each goal, in 1-3 plain sentences, tell them:
- where their recent behaviour is genuinely moving TOWARD the goal (cite what they actually did), and
- where it has quietly DRIFTED away, or where there is no sign of it at all.

Rules:
- Be specific and grounded only in the observations above. Do not invent progress.
- If a goal has no matching activity, say so plainly — that silence is the signal.
- No jargon, no flattery.

Then end with exactly ONE question — the single most important gap for them to
decide on next. Make it a question that helps them choose, not one that nags."""


def reflect(project_id: str) -> str:
    """Return a plain-language reflection of goals vs. recent behaviour.

    Returns a friendly message (not an error) when there is nothing to reflect.
    """
    cfg = load_config()
    db = Database(cfg.storage.db_path)
    db.connect()
    db.execute("PRAGMA busy_timeout=5000")
    try:
        goals = db.list_goals(project_id, status="open")
        if not goals:
            return (
                "You haven't told the mirror what you're trying to do yet.\n"
                "Add a goal first, e.g.:\n"
                '  mem goal add "I want the memory system to know my world"\n'
                "Then come back and run  mem mirror."
            )
        observations = _gather_observations(db, project_id, _OBSERVATION_LIMIT)
        if not observations:
            return (
                "I have your goals, but no recent activity to hold them against "
                "in this project yet. Do some work, then look in the mirror."
            )
        prompt = _build_prompt(goals, observations)
    finally:
        db.close()

    # LLM call happens outside the DB lock, through the user's configured router.
    router = LLMRouter(cfg.llm_roles)
    return router.call(_MIRROR_ROLE, prompt, system=_MIRROR_SYSTEM, max_tokens=1200)
