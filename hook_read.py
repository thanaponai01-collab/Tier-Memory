#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook — the READ REFLEX.

Mirror image of hook_ingest.py (the Stop hook / write reflex). Where the Stop
hook reflexively *writes* every finished turn into memoryd, this hook reflexively
*reads*: before the model answers, it retrieves the memory most relevant to the
user's prompt and injects it as context — so memory shows up without anyone
having to call a tool or ask for it.

Contract (verified on this machine): a UserPromptSubmit hook that prints to
stdout and exits 0 has that stdout injected into the model's context for the
turn. So we print a compact <recalled_memory> block and exit 0.

Discipline (same as the write reflex):
  - Always exits 0. Never blocks Claude Code.
  - Silent when there's nothing useful (daemon down, no prompt, no good hits).
  - Token-conscious: a modest budget + a hard cap on injected fragments, in
    keeping with the token-saving North Star.

NOT read-only: this is genuine usage. Retrieval touches fragments and logs a
retrieval event on purpose — that persisted (query -> injected fragment_ids)
linkage is what step 3 (closing the outcome loop) will read to learn which
injected memories actually got used.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make memory_system importable without a package install
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# Keep it light: a small token budget and a hard cap on how many fragments we
# inject, so the reflex never bloats the prompt. Retrieval's own min_crs gate
# (default 0.30) already drops weak matches before these limits apply.
_MAX_TOKENS = 1500
_MAX_FRAGMENTS = 6
_MIN_PROMPT_LEN = 12
_CLIENT_TIMEOUT = 5.0  # never hang the prompt waiting on memory

# Intent injection: the user's open goals are the one piece of memory that should
# STEER the turn, not just inform it — so the agent starts every prompt knowing
# what the user is actually trying to do, not only what's topically similar.
# A cheap direct DB read (no LLM), capped and truncated to stay token-conscious.
# (The full goals-vs-behaviour drift reflection lives in `mem mirror`; it needs
# an LLM call and is far too slow to run on every prompt.)
_MAX_GOALS = 5
_GOAL_STATEMENT_CAP = 240  # chars per goal statement

# Hand-off file: what we injected this turn, so the Stop hook (hook_ingest) can
# check which fragments the answer actually used and close the outcome loop.
_HANDOFF_FILE = Path.home() / ".agent" / "read_reflex_state.json"
_HANDOFF_CONTENT_CAP = 600   # chars of content kept per fragment (enough tokens)
_HANDOFF_MAX_SESSIONS = 100  # bound the file

# Conversational acks that aren't real queries — injecting memory for these is
# pure noise. Mirrors the skip set the other UserPromptSubmit hook uses.
_SKIP_WORDS = {
    "thanks", "thank you", "ok", "okay", "k", "good", "fine", "done", "yes",
    "no", "sure", "nice", "yep", "nope", "great", "cool", "hi", "hello",
    "continue", "go", "go on", "do it", "skip", "stop", "next",
}


# Temporal lane: prompts that ask "what did I work on" are answered by walking
# sessions in date order, not by vector similarity — semantic search is poor at
# "last week" (it matches the words "last"/"week", not recent activity). When the
# prompt is temporal we ALSO inject a date-ordered digest of recent sessions, the
# same answer `mem recent` and the dashboard give. Kept narrow on purpose: a real
# "what was I doing" question, not any prompt that happens to mention a day.
_TEMPORAL_RE = re.compile(
    r"\b("
    r"what (did|have|was|were) (i|we|you).{0,30}\b(work|working|do|doing|build|building|ship|fix|fixing|last|recent|up to)"
    r"|what('?s| was| have i been| did i get) .{0,20}\b(done|up to|working on|built)"
    r"|(remind me|recap|catch me up|where (did|were) we|where was i)"
    r"|what.{0,20}\b(last (week|night|session|time)|yesterday|recently|this week|past (few|couple) (days|weeks))"
    r")\b",
    re.IGNORECASE,
)
_MAX_RECENT_SESSIONS = 5
_RECENT_LINE_CAP = 160  # chars per session digest line


def _is_temporal_query(prompt: str) -> bool:
    return bool(_TEMPORAL_RE.search(prompt))


def _recent_digest(client, project_id: str, prompt: str) -> list[str]:
    """Date-ordered 'what did I work on' digest for a temporal prompt. Reuses the
    CLI's free-form time grammar so the reflex, `mem recent`, and the dashboard
    answer identically. Scoped to this project (the reflex is always in a project
    cwd). Best-effort: any failure returns nothing and the turn is unaffected."""
    try:
        from memory_system.cli import parse_time_window
        since_iso, limit, _label = parse_time_window(prompt)
    except Exception:
        since_iso, limit = None, _MAX_RECENT_SESSIONS
    limit = min(limit or _MAX_RECENT_SESSIONS, _MAX_RECENT_SESSIONS)
    try:
        resp = client.recent(since_iso=since_iso, limit=limit, project_id=project_id)
    except Exception:
        return []
    lines: list[str] = []
    for s in resp.get("sessions", [])[:_MAX_RECENT_SESSIONS]:
        when = (s.get("started_at") or "")[:10]
        summary = (s.get("summary") or "").strip()
        if summary:
            text = " ".join(summary.split())
        else:
            highlights = s.get("highlights") or []
            text = " ".join(highlights[0].split()) if highlights else ""
        if not text:
            continue
        if len(text) > _RECENT_LINE_CAP:
            text = text[:_RECENT_LINE_CAP - 1].rstrip() + "…"
        lines.append(f"- {when}: {text}")
    return lines


def _should_skip(prompt: str) -> bool:
    p = prompt.strip()
    if len(p) < _MIN_PROMPT_LEN:
        return True
    # Slash commands and shell escapes aren't semantic queries.
    if p.startswith("/") or p.startswith("!"):
        return True
    if p.lower().rstrip("!.,? ") in _SKIP_WORDS:
        return True
    return False


def _write_handoff(session_id: str, prompt: str, fragments: list[dict]) -> None:
    """Record what we injected so the Stop hook can later detect which of these
    fragments the model's answer actually used (the outcome-loop substrate).
    Best-effort: a failure here must never affect the injection itself."""
    if not session_id:
        return
    try:
        try:
            state = json.loads(_HANDOFF_FILE.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}
        state[session_id] = {
            "prompt": prompt[:2000],
            "ts": datetime.now(timezone.utc).isoformat(),
            # {fragment_id -> truncated content} — enough distinctive tokens for
            # the Stop hook's overlap check without bloating the file.
            "fragments": {
                f.get("id", ""): str(f.get("content", ""))[:_HANDOFF_CONTENT_CAP]
                for f in fragments if f.get("id")
            },
        }
        # Bound growth: keep the most recently written sessions.
        if len(state) > _HANDOFF_MAX_SESSIONS:
            for k in list(state.keys())[:-_HANDOFF_MAX_SESSIONS]:
                del state[k]
        _HANDOFF_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HANDOFF_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _open_goals(project_id: str) -> list[str]:
    """The user's open, self-declared goals for this project — what they SAID
    they're trying to do. Best-effort: a failure here must never affect the
    fragment injection or the turn. No LLM, just a direct read of the goals
    table (the same one `mem mirror` reflects against)."""
    try:
        from memory_system.mirror import list_goals
        goals = list_goals(project_id, status="open")  # source='user' only
    except Exception:
        return []
    statements: list[str] = []
    for g in goals[:_MAX_GOALS]:
        s = " ".join(str(getattr(g, "statement", "")).split())
        if not s:
            continue
        if len(s) > _GOAL_STATEMENT_CAP:
            s = s[:_GOAL_STATEMENT_CAP].rstrip() + "…"
        statements.append(s)
    return statements


def _format_block(goals: list[str], fragments: list[dict], recent: list[str]) -> str:
    """Render the injected context block. Intent first (the open goals that
    should steer the turn), then — for a temporal prompt — a date-ordered digest
    of recent work, then the topically-relevant memory. Compact, clearly
    labelled as auto-surfaced and optional, with a fragment id tag for
    traceability. Lives inside <recalled_memory> so the Stop hook strips the
    whole thing before ingest — goals are never re-ingested as new memory."""
    lines = ["<recalled_memory source=\"memory-read-reflex\">"]
    if goals:
        lines.append(
            "What you've said you're trying to do (your open goals — keep these in view):"
        )
        for g in goals:
            lines.append(f"- {g}")
    if recent:
        if goals:
            lines.append("")
        lines.append(
            "What you actually worked on recently (newest first — answer 'what did I do' from this):"
        )
        lines.extend(recent)
    if fragments:
        if goals or recent:
            lines.append("")  # blank line between sections
        lines.append(
            "Relevant memory auto-surfaced for this prompt (use if helpful, ignore if not):"
        )
        for f in fragments:
            crs = f.get("crs", 0.0)
            fid = f.get("id", "?")
            content = " ".join(str(f.get("content", "")).split())  # collapse whitespace
            lines.append(f"- [{crs:.2f} | {fid}] {content}")
    lines.append("</recalled_memory>")
    return "\n".join(lines)


def main() -> None:
    # Recursion/noise guard: skip memory injection for the daemon's own
    # `claude -p` budget-fallback helper sessions (see hook_ingest.py).
    if os.environ.get("TIER_MEMORY_NO_INGEST"):
        sys.exit(0)

    # Force UTF-8 stdout on Windows so injected memory can't crash on cp1252.
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 1. Parse hook payload
    try:
        hook_data = json.loads(sys.stdin.read().strip())
    except Exception:
        sys.exit(0)

    prompt = (hook_data.get("prompt") or "").strip()
    cwd = hook_data.get("cwd") or os.getcwd()
    session_id = hook_data.get("session_id", "")

    if not prompt or _should_skip(prompt):
        sys.exit(0)

    # 2. Resolve project_id from cwd (same logic as the write reflex)
    try:
        from memory_system.project import resolve_project_id
        project_id = resolve_project_id(Path(cwd))
    except Exception:
        sys.exit(0)

    # 3. Retrieve — needs the daemon (vector/graph/BM25). If it's down or errors,
    #    we don't bail: goals are a direct DB read and can still inject below.
    resp: dict = {}
    recent_lines: list[str] = []
    temporal = _is_temporal_query(prompt)
    try:
        from memory_system.daemon import get_client, is_running
        if is_running():
            with get_client(timeout=_CLIENT_TIMEOUT) as client:
                resp = client.retrieve(
                    project_id=project_id,
                    query_text=prompt,
                    max_tokens=_MAX_TOKENS,
                    scopes=["project", "global"],
                    # read_only stays False on purpose: this is real usage and the
                    # retrieval-event log is the substrate for the outcome loop.
                )
                # Temporal lane: 'what did I work on' is a date-walk, not a vector
                # search — inject the recent-sessions digest alongside the topical
                # hits so the answer comes from real history, not similarity.
                if temporal:
                    recent_lines = _recent_digest(client, project_id, prompt)
    except Exception:
        resp = {}

    fragments = resp.get("fragments", [])[:_MAX_FRAGMENTS]

    # 4. Fetch the user's open goals — the intent that should steer the turn.
    #    These surface even when no fragment is topically relevant, so the agent
    #    always knows what the user is working toward.
    goals = _open_goals(project_id)

    if not fragments and not goals and not recent_lines:
        sys.exit(0)  # nothing worth injecting — stay quiet

    # 5. Hand off the fragments we're injecting so the Stop hook can close the
    #    citation loop. (Goals and the recent digest aren't fragments and don't
    #    participate in it.)
    if fragments:
        _write_handoff(session_id, prompt, fragments)

    # 6. Inject. stdout on a UserPromptSubmit hook is added to the model context.
    print(_format_block(goals, fragments, recent_lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
