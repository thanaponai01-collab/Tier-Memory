"""
Lightweight daemon state-file helpers.
Imported by both the client and CLI — must NOT import server.py or any heavy deps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_PORT = 7463
STATE_FILE   = Path.home() / ".agent" / "memoryd.state"


def read_state() -> Optional[dict]:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def write_state(host: str, port: int) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"host": host, "port": port, "pid": os.getpid()})
    )


def clear_state() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
