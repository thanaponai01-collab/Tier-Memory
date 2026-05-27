"""
memory_system.tools — tool schema and executor for the agent runner.

Three tools are exposed to the model:
  read_file    — read a file relative to CWD
  write_file   — create or overwrite a file relative to CWD
  run_command  — run a shell command in CWD, 30 s timeout

All file paths are validated to stay within CWD (no traversal).
Tool errors are returned as strings starting with "ERROR: " — the model sees
them and can decide how to respond rather than receiving an exception.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Frozen between versions for cache stability.
TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file relative to the current working directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root (CWD).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (create or overwrite) a file relative to the current working directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root (CWD).",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the current working directory. "
            "Captures stdout and stderr. 30 second timeout. "
            "Does not receive stdin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
            },
            "required": ["command"],
        },
    },
]

_MAX_OUTPUT = 8_000


def _safe_path(path_str: str, cwd: Path) -> Path | None:
    """Return the resolved absolute path if it is within cwd, else None."""
    try:
        resolved = (cwd / path_str).resolve()
        resolved.relative_to(cwd.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def execute(tool_name: str, tool_input: dict, cwd: Path) -> str:
    """
    Dispatch a tool call and return the result as a plain string.
    Errors are returned as strings starting with "ERROR: ".
    """
    if tool_name == "read_file":
        path_str = tool_input.get("path", "")
        path = _safe_path(path_str, cwd)
        if path is None:
            return f"ERROR: path '{path_str}' would escape the project directory"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return f"ERROR: file not found: {path_str}"
        except OSError as e:
            return f"ERROR: {e}"
        if len(content) > _MAX_OUTPUT:
            content = content[:_MAX_OUTPUT] + "\n... [truncated]"
        return content

    if tool_name == "write_file":
        path_str = tool_input.get("path", "")
        path = _safe_path(path_str, cwd)
        if path is None:
            return f"ERROR: path '{path_str}' would escape the project directory"
        content = tool_input.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return "ok"
        except OSError as e:
            return f"ERROR: {e}"

    if tool_name == "run_command":
        command = tool_input.get("command", "")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 30 seconds"
        except OSError as e:
            return f"ERROR: {e}"
        if not output:
            return "(no output)"
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n... [truncated]"
        return output

    return f"ERROR: unknown tool '{tool_name}'"
