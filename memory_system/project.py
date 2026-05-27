"""
memory_system.project — project ID resolution, session IDs, user profile, and
git dirty-file helpers.
"""

from __future__ import annotations

import hashlib
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROFILE_PATH = Path.home() / ".agent" / "user_profile.md"


def resolve_project_id(cwd: Path) -> str:
    """
    Return a deterministic project ID like 'proj_a7f3b2' for the given directory.
    Prefers the git remote URL; falls back to the absolute CWD path.
    """
    raw: str | None = None
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
    except Exception:
        pass

    if not raw:
        raw = str(cwd.resolve())

    digest = hashlib.sha256(raw.encode()).hexdigest()[:6]
    return f"proj_{digest}"


def new_session_id() -> str:
    """Return a unique session ID like 'sess_20260525_143000_a3f1'."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"sess_{ts}_{suffix}"


def load_global_profile() -> str | None:
    """Return the global user profile text from ~/.agent/user_profile.md, or None."""
    try:
        text = PROFILE_PATH.read_text(encoding="utf-8").strip()
        return text or None
    except FileNotFoundError:
        return None


def save_global_profile(text: str) -> None:
    """Write the global user profile to ~/.agent/user_profile.md."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(text, encoding="utf-8")


def git_dirty_files(cwd: Path, max_bytes: int = 16_000) -> str | None:
    """
    Return a unified diff of all changes vs HEAD (staged + unstaged).
    Returns None if git is unavailable, the repo is clean, or cwd is not a repo.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        diff = result.stdout
        if len(diff) > max_bytes:
            diff = diff[:max_bytes] + "\n... [diff truncated]"
        return diff
    except Exception:
        return None
