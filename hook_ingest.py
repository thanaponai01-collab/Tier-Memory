#!/usr/bin/env python3
"""
Claude Code Stop hook — ingest the completed agent turn into memoryd.

Fires after every agent turn. Tracks the last-processed line per session
so only new lines are sent to the daemon on each call.

Always exits 0. Never blocks Claude Code.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make memory_system importable without a package install
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

_STATE_FILE = Path.home() / ".agent" / "hook_state.json"
_MAX_NEW_LINES = 500          # safety cap per invocation
_CORRECTION_WORDS = ("no ", "wrong", "incorrect", "that's not", "not right")

# The read reflex (hook_read.py) injects a <recalled_memory> block into the
# turn's context. If it lands in the transcript we must strip it before
# ingesting — otherwise memory re-ingests its own injected echoes every turn,
# flooding the store with what it already knows.
_INJECTED_BLOCK = re.compile(
    r"<recalled_memory\b.*?</recalled_memory>", re.DOTALL | re.IGNORECASE
)

# Claude Code wraps slash-command invocations and harness notices into the
# transcript as ordinary "user" turns. Left in, the distiller faithfully
# summarizes them into junk headlines — e.g. "Use the /next command to
# navigate" (from command tags) — that then dominate temporal recall. None of
# it is the user's actual words, so strip these tag-blocks before ingest.
_SCAFFOLD_BLOCKS = re.compile(
    r"<(command-name|command-message|command-args|system-reminder"
    r"|local-command-stdout|local-command-stderr)\b.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# A slash-command skill expands into a user turn that begins with this marker
# followed by the skill's whole markdown body — pure instructions, never the
# user's content. (This is why "Chief Engineer skill routes requests based on
# artifact state…" leaked verbatim into 4 different session headlines.) Drop
# the entire turn when present. Forward-only: existing fragments are unaffected.
_SKILL_BODY_MARKER = "Base directory for this skill:"


def _scrub_scaffolding(text: str) -> str:
    """Strip Claude Code harness scaffolding from a transcript turn so the
    distiller never summarizes it into a junk headline. Returns "" when the
    whole turn is scaffolding (the caller then skips it)."""
    text = _INJECTED_BLOCK.sub("", text)
    text = _SCAFFOLD_BLOCKS.sub("", text)
    if _SKILL_BODY_MARKER in text:
        return ""
    return text.strip()

# Outcome loop: what the read reflex injected this session, so we can detect
# which fragments the answer actually used and report them as citations.
_READ_REFLEX_HANDOFF = Path.home() / ".agent" / "read_reflex_state.json"

# Grow the eval oracle from real usage: every real citation is a free
# (query -> fragment-the-answer-used) ground-truth pair. We append one JSON line
# per citation here; `mem eval grow` later mines them into eval cases. Bounded so
# it can't grow without limit. (Path mirrors memory_system.eval.CITATION_LOG.)
_CITATION_LOG = Path.home() / ".agent" / "citation_examples.jsonl"
_CITATION_LOG_MAX_LINES = 500


def _log_citation_example(query: str, cited_ids: list[str]) -> None:
    """Append a (query -> cited fragment ids) example for `mem eval grow`.
    Best-effort: a failure here must never affect the turn or the citation."""
    if not query.strip() or not cited_ids:
        return
    try:
        rec = json.dumps({
            "query": query[:2000],
            "expect_id": cited_ids,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        existing: list[str] = []
        if _CITATION_LOG.exists():
            existing = _CITATION_LOG.read_text(encoding="utf-8").splitlines()
        existing.append(rec)
        if len(existing) > _CITATION_LOG_MAX_LINES:
            existing = existing[-_CITATION_LOG_MAX_LINES:]
        _CITATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        _CITATION_LOG.write_text("\n".join(existing) + "\n", encoding="utf-8")
    except Exception:
        pass


# ── State tracking (per session, last processed line index) ──────────────────

def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


# ── Transcript parsing ────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _parse_new_lines(
    transcript_path: str,
    start_line: int,
) -> tuple[list[dict], int, dict]:
    """
    Read lines from start_line onwards, parse into transcript_segments.
    Returns (segments, new_line_count, usage) where usage sums the real API
    token counts across the new assistant turns. Forwarding these is what lets
    the savings metric report *measured* numbers instead of a projection.
    """
    p = Path(transcript_path)
    if not p.exists():
        return [], start_line, {}

    try:
        all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return [], start_line, {}

    total = len(all_lines)
    if start_line >= total:
        return [], total, {}  # nothing new

    new_lines = all_lines[start_line : start_line + _MAX_NEW_LINES]
    now = datetime.now(timezone.utc).isoformat()
    segments: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_creation_tokens": 0}

    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue

        # Claude Code JSONL is either {"type": "user", "message": {...}} or {"role": "user", "content": ...}
        role = entry.get("role") or entry.get("type") or ""
        msg = entry.get("message") or entry
        if not isinstance(msg, dict):
            continue

        content = msg.get("content")
        if content is None:
            continue

        # Strip injected memory echoes AND harness scaffolding (command tags,
        # system reminders, skill bodies) so we store the real turn, not the
        # plumbing — otherwise the distiller summarizes the plumbing.
        text = _scrub_scaffolding(_extract_text(content))
        if not text:
            continue

        timestamp = entry.get("timestamp") or msg.get("timestamp") or now
        lower_role = role.lower()

        if "user" in lower_role:
            lower_text = text.lower()
            segments.append({
                "role": "user",
                "content": text,
                "timestamp": timestamp,
                "is_correction": any(w in lower_text for w in _CORRECTION_WORDS),
                "is_explicit_remember": "remember" in lower_text,
                "tool_call_chain_len": 0,
            })
        elif "assistant" in lower_role:
            tool_count = (
                sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
                if isinstance(content, list)
                else 0
            )
            segments.append({
                "role": "assistant",
                "content": text,
                "timestamp": timestamp,
                "is_correction": False,
                "is_explicit_remember": False,
                "tool_call_chain_len": tool_count,
            })
            # Real API token accounting — the Anthropic usage block on the
            # assistant message. This is the path the savings metric measures.
            u = msg.get("usage")
            if isinstance(u, dict):
                usage["input_tokens"] += int(u.get("input_tokens", 0) or 0)
                usage["output_tokens"] += int(u.get("output_tokens", 0) or 0)
                usage["cache_read_tokens"] += int(u.get("cache_read_input_tokens", 0) or 0)
                usage["cache_creation_tokens"] += int(u.get("cache_creation_input_tokens", 0) or 0)

    new_total = min(start_line + len(new_lines), total)
    return segments, new_total, usage


# ── Outcome loop: detect & report citations ────────────────────────────────────

def _report_citations(client, session_id: str, segments: list[dict]) -> None:
    """Close the read→use loop: of the fragments the read reflex injected for
    this session's last prompt, mark the ones the assistant's answer actually
    drew on. Best-effort and self-consuming — never blocks the turn."""
    if not session_id:
        return
    try:
        state = json.loads(_READ_REFLEX_HANDOFF.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return
    except Exception:
        return

    entry = state.pop(session_id, None)  # consume regardless of outcome
    # Persist the consumption so a stale entry can't be matched to a later turn.
    try:
        _READ_REFLEX_HANDOFF.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass

    if not entry:
        return
    injected = entry.get("fragments") or {}
    if not injected:
        return

    assistant_text = " ".join(
        s.get("content", "") for s in segments if s.get("role") == "assistant"
    )
    if not assistant_text.strip():
        return

    try:
        from memory_system.citation import detect_cited
        cited = detect_cited(injected, entry.get("prompt", ""), assistant_text)
    except Exception:
        return

    if cited:
        try:
            client.cite(cited)
        except Exception:
            pass
        # Free ground truth for the eval oracle: this query genuinely used these
        # fragments, so the read path should surface them for it. (See
        # `mem eval grow`.) Best-effort; never affects the citation above.
        _log_citation_example(entry.get("prompt", ""), cited)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 0. Recursion guard. When the memory daemon's budget fallback runs LLM work
    #    through `claude -p` (subscription model, because the paid API is out of
    #    credit), that helper session would otherwise be ingested here → distilled
    #    → spawn another `claude -p` → loop. The daemon sets TIER_MEMORY_NO_INGEST
    #    on those child processes; skip them entirely (also keeps internal helper
    #    sessions out of the store).
    if os.environ.get("TIER_MEMORY_NO_INGEST"):
        sys.exit(0)

    # 1. Parse hook payload
    try:
        hook_data = json.loads(sys.stdin.read().strip())
    except Exception:
        sys.exit(0)

    session_id = hook_data.get("session_id", "")
    transcript_path = hook_data.get("transcript_path", "")
    cwd = hook_data.get("cwd") or os.getcwd()
    model = hook_data.get("model", "") or None

    if not session_id or not transcript_path:
        sys.exit(0)

    # 2. Load per-session state (last processed line)
    state = _load_state()
    start_line = state.get(session_id, 0)

    # 3. Parse only new lines from the transcript
    segments, new_line_count, usage = _parse_new_lines(transcript_path, start_line)
    if not segments:
        sys.exit(0)

    # 4. Resolve project_id from cwd
    try:
        from memory_system.project import resolve_project_id
        project_id = resolve_project_id(Path(cwd))
    except Exception:
        project_id = f"proj_{Path(cwd).name[:20]}"

    # 5. Send to daemon (skip silently if not running)
    try:
        from memory_system.daemon import get_client, is_running
        if not is_running():
            sys.exit(0)
        with get_client(timeout=5.0) as client:
            client.ingest(
                project_id=project_id,
                session_id=session_id,
                transcript_segments=segments,
                model_id=model,
                priority="background",
                input_tokens=usage.get("input_tokens") or None,
                output_tokens=usage.get("output_tokens") or None,
                cache_read_tokens=usage.get("cache_read_tokens") or None,
                cache_creation_tokens=usage.get("cache_creation_tokens") or None,
            )
            # Close the outcome loop: mark injected fragments the answer used.
            _report_citations(client, session_id, segments)
    except Exception:
        sys.exit(0)

    # 6. Persist updated line position
    state[session_id] = new_line_count
    # Evict old sessions — keep at most 100 to prevent unbounded growth
    if len(state) > 100:
        oldest = sorted(state.keys())[:-100]
        for k in oldest:
            del state[k]
    _save_state(state)

    sys.exit(0)


if __name__ == "__main__":
    main()
