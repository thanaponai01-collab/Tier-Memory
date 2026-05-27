"""
memory_system.mcp_server — MCP server exposing memory to Claude Code.

Tools:
  memory_retrieve(query)  — fetch relevant memories before answering
  memory_save(content)    — persist a fact/decision/lesson to memory
  memory_stats()          — show daemon status

Registered via .claude/settings.json; Claude Code connects automatically.
The daemon is auto-started if not already running.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from memory_system.daemon import MemoryClientError, get_client, is_running
from memory_system.daemon.server import DEFAULT_PORT
from memory_system.project import new_session_id, resolve_project_id

mcp = FastMCP("memory")


def _ensure_daemon() -> None:
    if is_running():
        return
    cmd = [sys.executable, "-m", "memory_system.daemon.server", "--port", str(DEFAULT_PORT)]
    if platform.system() == "Windows":
        subprocess.Popen(
            cmd,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if is_running():
            return
        time.sleep(0.25)
    raise RuntimeError("memoryd did not start in time")


def _resolve(project_id: str) -> str:
    return project_id or resolve_project_id(Path.cwd())


@mcp.tool()
def memory_retrieve(query: str, project_id: str = "") -> str:
    """
    Retrieve memory fragments relevant to a query.

    Call this at the start of a conversation or when you need context about
    past work, decisions, or facts from previous sessions.

    Args:
        query: what you want to recall (natural language)
        project_id: project scope (default: derived from current directory)
    """
    _ensure_daemon()
    pid = _resolve(project_id)
    try:
        with get_client() as c:
            resp = c.retrieve(pid, query_text=query, max_tokens=2000)
    except MemoryClientError as e:
        return f"[memory error: {e}]"

    fragments = resp.get("fragments", [])
    summary = resp.get("project_summary")

    if not fragments and not summary:
        return "[memory: nothing relevant found]"

    parts: list[str] = []
    if summary:
        parts.append(f"Project context:\n{summary}")
    if fragments:
        parts.append(f"\nRelevant memories ({len(fragments)} found):")
        for i, f in enumerate(fragments, 1):
            crs = f.get("crs", 0)
            content = f.get("content", "")
            parts.append(f"  [{i}] (relevance={crs:.2f}) {content}")

    return "\n".join(parts)


@mcp.tool()
def memory_save(
    content: str,
    project_id: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> str:
    """
    Save something important to memory for future sessions.

    Use this to persist facts, decisions, corrections, or lessons learned
    during the current conversation so they are available in future sessions.

    Args:
        content: the text to remember (a fact, decision, or lesson)
        project_id: project scope (default: derived from current directory)
        input_tokens: input tokens used in this conversation turn (for savings tracking)
        output_tokens: output tokens used in this conversation turn (for savings tracking)
    """
    _ensure_daemon()
    pid = _resolve(project_id)
    session_id = new_session_id()
    segments = [
        {"role": "user", "content": "Remember this:", "is_explicit_remember": True},
        {"role": "assistant", "content": content},
    ]
    try:
        with get_client(timeout=30.0) as c:
            resp = c.ingest(
                project_id=pid,
                session_id=session_id,
                transcript_segments=segments,
                priority="immediate",
                input_tokens=input_tokens or None,
                output_tokens=output_tokens or None,
            )
    except MemoryClientError as e:
        return f"[memory error: {e}]"

    episodes = resp.get("episodes_created", 0)
    facts = resp.get("facts_created", 0)
    ep_label = "episode" if episodes == 1 else "episodes"
    return f"[memory] saved | {episodes} {ep_label} | {facts} new facts"


@mcp.tool()
def memory_stats(project_id: str = "") -> str:
    """
    Show memory daemon status and fragment counts.

    Args:
        project_id: filter stats to a project (default: all projects)
    """
    _ensure_daemon()
    pid = _resolve(project_id) if project_id else None
    try:
        with get_client() as c:
            resp = c.stats(project_id=pid)
    except MemoryClientError as e:
        return f"[memory error: {e}]"

    stats = resp.get("stats", {})
    active = sum(stats.values()) if stats else 0
    idx = resp.get("vector_index_size", "?")
    queue = resp.get("ingest_queue_depth", 0)

    lines = [
        f"fragments: {active} active",
        f"vector index: {idx} entries",
        f"ingest queue: {queue} pending",
    ]
    if stats:
        lines.append("breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
