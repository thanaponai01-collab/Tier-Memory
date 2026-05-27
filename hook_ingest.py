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
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make memory_system importable without a package install
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

_STATE_FILE = Path.home() / ".agent" / "hook_state.json"
_MAX_NEW_LINES = 500          # safety cap per invocation
_CORRECTION_WORDS = ("no ", "wrong", "incorrect", "that's not", "not right")


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
) -> tuple[list[dict], int]:
    """
    Read lines from start_line onwards, parse into transcript_segments.
    Returns (segments, new_line_count).
    """
    p = Path(transcript_path)
    if not p.exists():
        return [], start_line

    try:
        all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return [], start_line

    total = len(all_lines)
    if start_line >= total:
        return [], total  # nothing new

    new_lines = all_lines[start_line : start_line + _MAX_NEW_LINES]
    now = datetime.now(timezone.utc).isoformat()
    segments: list[dict] = []

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

        text = _extract_text(content).strip()
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

    new_total = min(start_line + len(new_lines), total)
    return segments, new_total


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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
    segments, new_line_count = _parse_new_lines(transcript_path, start_line)
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
            )
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
