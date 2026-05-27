"""
memory_system.cold_storage — raw session transcript archival.

Each session is written as a .jsonl.zst file:
  <cold_root>/<project_id>/<YYYY-MM>/<session_id>.jsonl.zst

Line 1:  {"type": "session_meta", ...}
Line 2+: {"type": "segment", "role": ..., "content": ..., ...}

Reads back with:
  read_session(cold_root, project_id, session_id) -> (meta, segments)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import zstandard as zstd


def append_session(
    cold_root: Path | str,
    project_id: str,
    session_id: str,
    segments: list[dict],
    *,
    model_id: Optional[str] = None,
    summary: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Path:
    """
    Write a session transcript to cold storage. Returns the path written.
    Safe to call multiple times for the same session — overwrites.
    """
    cold_root = Path(cold_root)
    now = datetime.now(tz=timezone.utc)
    month_dir = cold_root / project_id / now.strftime("%Y-%m")
    month_dir.mkdir(parents=True, exist_ok=True)

    dest = month_dir / f"{session_id}.jsonl.zst"

    meta = {
        "type": "session_meta",
        "project_id": project_id,
        "session_id": session_id,
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "summary": summary,
        "archived_at": now.isoformat(),
    }

    cctx = zstd.ZstdCompressor(level=3)
    with dest.open("wb") as fh:
        with cctx.stream_writer(fh, closefd=False) as writer:
            writer.write((json.dumps(meta) + "\n").encode())
            for seg in segments:
                line = {"type": "segment", **seg}
                writer.write((json.dumps(line) + "\n").encode())

    return dest


def read_session_from_path(path: Path) -> tuple[dict, list[dict]]:
    """
    Read a session directly from a .jsonl.zst file path.
    Returns (meta_dict, list_of_segment_dicts).
    """
    dctx = zstd.ZstdDecompressor()
    meta: dict = {}
    segments: list[dict] = []

    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            for raw_line in reader.read().splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "session_meta":
                    meta = obj
                elif obj.get("type") == "segment":
                    segments.append(obj)

    return meta, segments


def read_session(
    cold_root: Path | str,
    project_id: str,
    session_id: str,
) -> tuple[dict, list[dict]]:
    """
    Read a session back from cold storage.
    Returns (meta_dict, list_of_segment_dicts).
    Raises FileNotFoundError if the session file doesn't exist.
    """
    cold_root = Path(cold_root)
    matches = list(cold_root.glob(f"{project_id}/*/{session_id}.jsonl.zst"))
    if not matches:
        raise FileNotFoundError(
            f"Cold storage: session {session_id!r} not found under {cold_root / project_id}"
        )

    dctx = zstd.ZstdDecompressor()
    meta: dict = {}
    segments: list[dict] = []

    with matches[0].open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            for raw_line in reader.read().splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "session_meta":
                    meta = obj
                elif obj.get("type") == "segment":
                    segments.append(obj)

    return meta, segments
