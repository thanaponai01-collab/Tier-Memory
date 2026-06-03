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
import sys
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

# Conversational acks that aren't real queries — injecting memory for these is
# pure noise. Mirrors the skip set the other UserPromptSubmit hook uses.
_SKIP_WORDS = {
    "thanks", "thank you", "ok", "okay", "k", "good", "fine", "done", "yes",
    "no", "sure", "nice", "yep", "nope", "great", "cool", "hi", "hello",
    "continue", "go", "go on", "do it", "skip", "stop", "next",
}


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


def _format_block(fragments: list[dict]) -> str:
    """Render the injected context block. Compact, clearly labelled as
    auto-surfaced and optional, with a fragment id tag for traceability."""
    lines = [
        "<recalled_memory source=\"memory-read-reflex\">",
        "Relevant memory auto-surfaced for this prompt (use if helpful, ignore if not):",
    ]
    for f in fragments:
        crs = f.get("crs", 0.0)
        fid = f.get("id", "?")
        content = " ".join(str(f.get("content", "")).split())  # collapse whitespace
        lines.append(f"- [{crs:.2f} | {fid}] {content}")
    lines.append("</recalled_memory>")
    return "\n".join(lines)


def main() -> None:
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

    if not prompt or _should_skip(prompt):
        sys.exit(0)

    # 2. Resolve project_id from cwd (same logic as the write reflex)
    try:
        from memory_system.project import resolve_project_id
        project_id = resolve_project_id(Path(cwd))
    except Exception:
        sys.exit(0)

    # 3. Retrieve — skip silently if the daemon isn't up
    try:
        from memory_system.daemon import get_client, is_running
        if not is_running():
            sys.exit(0)
        with get_client(timeout=_CLIENT_TIMEOUT) as client:
            resp = client.retrieve(
                project_id=project_id,
                query_text=prompt,
                max_tokens=_MAX_TOKENS,
                scopes=["project", "global"],
                # read_only stays False on purpose: this is real usage and the
                # retrieval-event log is the substrate for the outcome loop.
            )
    except Exception:
        sys.exit(0)

    fragments = resp.get("fragments", [])[:_MAX_FRAGMENTS]
    if not fragments:
        sys.exit(0)  # nothing worth injecting — stay quiet

    # 4. Inject. stdout on a UserPromptSubmit hook is added to the model context.
    print(_format_block(fragments))
    sys.exit(0)


if __name__ == "__main__":
    main()
