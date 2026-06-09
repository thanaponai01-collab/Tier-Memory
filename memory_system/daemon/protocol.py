"""
Wire protocol for the memory daemon.

Transport: line-delimited JSON over TCP loopback (Windows-compatible;
no Unix sockets required).  Each message is one UTF-8 JSON line.

Request:  {"op": "<operation>", ...params}
Response: {"status": "ok"|"error", ...payload}
          or {"status": "queued", "estimated_completion_ms": N}  for async ops
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Any, Optional


# ── Supported operations ─────────────────────────────────────────────────────

OP_RETRIEVE = "retrieve"
OP_INGEST   = "ingest"
OP_STATS    = "stats"
OP_SEARCH   = "search"
OP_PIN      = "pin"
OP_FORGET   = "forget"
OP_PING     = "ping"
OP_AUDIT    = "audit"
OP_REINDEX  = "reindex"
OP_EXPORT   = "export"
OP_IMPORT   = "import"
OP_FEEDBACK = "feedback"
OP_SAVINGS  = "savings"
OP_UPGRADE  = "upgrade"
OP_CITE     = "cite"
OP_RECENT   = "recent"
OP_REINDEX_STATUS = "reindex_status"


# ── Wire helpers ─────────────────────────────────────────────────────────────

def encode(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def decode(line: bytes) -> dict:
    return json.loads(line.decode().strip())


def ok(**kwargs) -> dict:
    return {"status": "ok", **kwargs}


def error(msg: str) -> dict:
    return {"status": "error", "message": msg}


def queued(estimated_ms: int = 4500) -> dict:
    return {"status": "queued", "estimated_completion_ms": estimated_ms}


# ── Request schemas (validation only — no strict types to keep the protocol
#    forward-compatible) ─────────────────────────────────────────────────────

def validate_retrieve(req: dict) -> Optional[str]:
    for field in ("project_id", "query_text"):
        if field not in req:
            return f"missing field: {field}"
    if not isinstance(req.get("query_embedding", []), list):
        return "query_embedding must be a list of floats"
    return None


def validate_ingest(req: dict) -> Optional[str]:
    for field in ("project_id", "session_id", "transcript_segments"):
        if field not in req:
            return f"missing field: {field}"
    if not isinstance(req["transcript_segments"], list):
        return "transcript_segments must be a list"
    return None


def validate_pin(req: dict) -> Optional[str]:
    if "fragment_id" not in req:
        return "missing field: fragment_id"
    return None


def validate_forget(req: dict) -> Optional[str]:
    if "fragment_id" not in req:
        return "missing field: fragment_id"
    return None
