"""
memory_system.obsidian_bridge.watcher

Watches vault/notes/ and vault/corrections/ for human-written Markdown files
and feeds them into the memory system via the daemon.

- vault/notes/*.md       → ingest as user_explicit fragment (priority: immediate)
- vault/corrections/*.md → insert into corrections table via /api/obsidian/correct

Anti-loop: this watcher only watches notes/ and corrections/. The exporter writes
to memories/, entities/, patterns/ — no overlap, no feedback cycle.

After processing, `memory_synced_at` and `memory_sync_hash` are written back
into the file's frontmatter so the note is not re-ingested on the next pass.

Usage:
    watcher = VaultWatcher(vault_path, project_id, host="127.0.0.1", port=8765)
    watcher.run(interval=5)  # poll every 5 seconds
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..daemon.state import DEFAULT_PORT, read_state
from ..project import new_session_id


# ── Frontmatter helpers ───────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()
    return fm, body


def _write_frontmatter(path: Path, extra: dict) -> None:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            existing_fm = text[3:end].strip()
            body_after = text[end + 4:]
            new_lines = []
            for k, v in extra.items():
                if isinstance(v, bool):
                    val = "true" if v else "false"
                else:
                    val = str(v)
                new_lines.append(f"{k}: {val}")
            new_fm = existing_fm + "\n" + "\n".join(new_lines)
            path.write_text(f"---\n{new_fm}\n---{body_after}", encoding="utf-8")
            return
    new_lines = "\n".join(
        f"{k}: {'true' if v is True else 'false' if v is False else v}"
        for k, v in extra.items()
    )
    path.write_text(f"---\n{new_lines}\n---\n\n{text}", encoding="utf-8")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── TCP ingest helper ─────────────────────────────────────────────────────────

def _tcp_ingest(host: str, port: int, payload: dict, timeout: float = 10.0) -> None:
    """Send ingest payload; expects immediate queued-ack (background priority)."""
    msg = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(msg)
        f = sock.makefile("rb")
        line = f.readline()
        if line:
            resp = json.loads(line)
            if resp.get("status") == "error":
                raise RuntimeError(f"daemon error: {resp.get('message')}")


# ── Watcher ───────────────────────────────────────────────────────────────────

class VaultWatcher:
    def __init__(
        self,
        vault_path: str | Path,
        project_id: str,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ):
        self.vault = Path(vault_path)
        self.project_id = project_id
        self.host = host
        self.port = port
        self._web_port = port + 1

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, interval: float = 5.0) -> None:
        print(f"[obsidian-watch] vault: {self.vault}")
        print(f"[obsidian-watch] daemon: {self.host}:{self.port}")
        print(f"[obsidian-watch] watching notes/ and corrections/ (poll every {interval}s)")
        print("[obsidian-watch] Ctrl-C to stop")
        try:
            while True:
                self.watch_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[obsidian-watch] stopped.")

    def watch_once(self) -> tuple[int, int]:
        notes_processed = 0
        corr_processed  = 0

        notes_dir = self.vault / "notes"
        corr_dir  = self.vault / "corrections"
        notes_dir.mkdir(exist_ok=True)
        corr_dir.mkdir(exist_ok=True)

        for path in sorted(notes_dir.glob("*.md")):
            if self._process_note(path):
                notes_processed += 1

        for path in sorted(corr_dir.glob("*.md")):
            if self._process_correction(path):
                corr_processed += 1

        return notes_processed, corr_processed

    # ── Note processing ────────────────────────────────────────────────────────

    def _process_note(self, path: Path) -> bool:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False

        fm, body = _parse_frontmatter(text)
        body = body.strip()
        if not body:
            return False

        current_hash = _content_hash(body)
        if fm.get("memory_sync_hash") == current_hash:
            return False  # already ingested, content unchanged

        project_id = fm.get("project_id") or self.project_id
        session_id = new_session_id()
        now = datetime.now(timezone.utc).isoformat()

        payload = {
            "op": "ingest",
            "project_id": project_id,
            "session_id": session_id,
            "transcript_segments": [
                {
                    "role": "user",
                    "content": f"[obsidian-note: {path.name}] {body}",
                    "timestamp": now,
                    "is_correction": False,
                    "is_error_fix": False,
                    "is_explicit_remember": True,
                    "tool_call_chain_len": 0,
                }
            ],
            "priority": "background",
        }

        try:
            _tcp_ingest(self.host, self.port, payload)
            _write_frontmatter(path, {
                "memory_synced_at": now[:19].replace("T", " "),
                "memory_sync_hash": current_hash,
            })
            print(f"[obsidian-watch] note ingested: {path.name}")
            return True
        except Exception as e:
            print(f"[obsidian-watch] note failed ({path.name}): {e}")
            return False

    # ── Correction processing ──────────────────────────────────────────────────

    def _process_correction(self, path: Path) -> bool:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False

        fm, body = _parse_frontmatter(text)
        original = fm.get("original_fact", "").strip()
        corrected = fm.get("corrected_fact", "").strip()

        if not original or not corrected:
            return False

        sync_hash = _content_hash(original + corrected)
        if fm.get("memory_sync_hash") == sync_hash:
            return False  # already applied

        project_id = fm.get("project_id") or self.project_id
        now = datetime.now(timezone.utc).isoformat()

        payload = {
            "original_fact": original,
            "corrected_fact": corrected,
            "project_id": project_id,
        }

        try:
            _post_json(
                f"http://{self.host}:{self._web_port}/api/obsidian/correct",
                payload,
            )
            _write_frontmatter(path, {
                "memory_synced_at": now[:19].replace("T", " "),
                "memory_sync_hash": sync_hash,
            })
            print(f"[obsidian-watch] correction applied: {path.name}")
            return True
        except urllib.error.URLError as e:
            print(f"[obsidian-watch] correction failed ({path.name}): {e}")
            return False
        except Exception as e:
            print(f"[obsidian-watch] correction failed ({path.name}): {e}")
            return False
