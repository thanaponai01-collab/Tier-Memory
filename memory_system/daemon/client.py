"""
Synchronous TCP client for communicating with the memory daemon.

Usage in a CLI session:
    client = MemoryClient.from_state()   # connects to running daemon
    result = client.retrieve(
        project_id="my-app",
        query_text="fix the docker build",
        query_embedding=[...],           # pre-computed; omit to let daemon embed
    )
    client.ingest(project_id="my-app", session_id="sess_123", transcript_segments=[...])
    client.close()

All methods raise MemoryClientError on daemon errors or connection failure.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Optional

from . import protocol as P
from .protocol import OP_FEEDBACK
from .state import DEFAULT_PORT, read_state


class MemoryClientError(Exception):
    pass


class MemoryClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        timeout: float = 60.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._file = None

    # ── Connection ────────────────────────────────────────────────────────────

    @classmethod
    def from_state(cls, timeout: float = 10.0) -> "MemoryClient":
        """Connect to the daemon described in the state file."""
        state = read_state()
        if not state:
            raise MemoryClientError(
                "No running memoryd found. Start it with:\n"
                "  python -m memory_system.daemon.server"
            )
        client = cls(host=state["host"], port=state["port"], timeout=timeout)
        client.connect()
        return client

    def connect(self) -> None:
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._file = self._sock.makefile("rb")
        except OSError as e:
            raise MemoryClientError(f"Cannot connect to memoryd at {self.host}:{self.port}: {e}")

    def close(self) -> None:
        try:
            if self._file:
                self._file.close()
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        finally:
            self._sock = None
            self._file = None

    def __enter__(self) -> "MemoryClient":
        if not self._sock:
            self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Core operations ───────────────────────────────────────────────────────

    def ping(self) -> dict:
        return self._call({"op": P.OP_PING})

    def retrieve(
        self,
        project_id: str,
        query_text: str,
        query_embedding: Optional[list[float]] = None,
        max_tokens: Optional[int] = None,
        scopes: Optional[list[str]] = None,
        min_crs: Optional[float] = None,
        read_only: bool = False,
    ) -> dict:
        """
        Retrieve memory fragments relevant to query_text.

        Returns dict with keys:
          fragments: list of {id, scope, content, crs, token_count, ...}
          project_summary: str | None
          token_budget_used: int

        read_only=True replays the read path without side-effects (no fragment
        touch, no retrieval-event log) — used by measurement probes like
        `mem eval` so they don't contaminate access counts or the v4 signal.
        """
        req: dict[str, Any] = {
            "op": P.OP_RETRIEVE,
            "project_id": project_id,
            "query_text": query_text,
        }
        if query_embedding is not None:
            req["query_embedding"] = query_embedding
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if scopes is not None:
            req["scopes"] = scopes
        if min_crs is not None:
            req["min_crs"] = min_crs
        if read_only:
            req["read_only"] = True
        return self._call(req)

    def ingest(
        self,
        project_id: str,
        session_id: str,
        transcript_segments: list[dict],
        model_id: Optional[str] = None,
        priority: str = "background",
        session_summary: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
    ) -> dict:
        """
        Queue a session transcript for background compression and storage.

        transcript_segments: list of dicts with keys:
          role: "user"|"assistant"|"tool"
          content: str
          timestamp: str (ISO 8601, optional)
          is_correction: bool (optional)
          is_error_fix: bool (optional)
          is_explicit_remember: bool (optional)
          tool_call_chain_len: int (optional)

        Returns immediately with {"status": "queued"} unless priority="immediate".
        """
        req: dict[str, Any] = {
            "op": P.OP_INGEST,
            "project_id": project_id,
            "session_id": session_id,
            "transcript_segments": transcript_segments,
            "priority": priority,
        }
        if model_id:
            req["model_id"] = model_id
        if session_summary is not None:
            req["session_summary"] = session_summary
        if input_tokens is not None:
            req["input_tokens"] = input_tokens
        if output_tokens is not None:
            req["output_tokens"] = output_tokens
        if cache_read_tokens is not None:
            req["cache_read_tokens"] = cache_read_tokens
        if cache_creation_tokens is not None:
            req["cache_creation_tokens"] = cache_creation_tokens
        return self._call(req)

    def stats(self, project_id: Optional[str] = None) -> dict:
        req: dict[str, Any] = {"op": P.OP_STATS}
        if project_id:
            req["project_id"] = project_id
        return self._call(req)

    def search(self, query: str, project_id: Optional[str] = None) -> dict:
        req: dict[str, Any] = {"op": P.OP_SEARCH, "query": query}
        if project_id:
            req["project_id"] = project_id
        return self._call(req)

    def pin(self, fragment_id: str, pinned: bool = True) -> dict:
        return self._call({"op": P.OP_PIN, "fragment_id": fragment_id, "pinned": pinned})

    def forget(self, fragment_id: str) -> dict:
        return self._call({"op": P.OP_FORGET, "fragment_id": fragment_id})

    def feedback(self, fragment_id: str, value: float) -> dict:
        """
        Set user feedback on a fragment.
        value: +1.0 (helpful), -1.0 (unhelpful), 0.0 (reset).
        Clamped to [-1.0, 1.0].
        """
        return self._call({"op": OP_FEEDBACK, "fragment_id": fragment_id, "value": value})

    def cite(self, fragment_ids: list[str]) -> dict:
        """
        Record that these fragments were actually used in the answer that
        followed their injection — the outcome-loop citation signal. Bumps
        times_cited + last_cited_at, which v4 reads as a "useful" label.
        """
        return self._call({"op": P.OP_CITE, "fragment_ids": fragment_ids})

    def audit(self, project_id: Optional[str] = None) -> dict:
        """Trigger an immediate memory audit. Returns audit statistics."""
        req: dict[str, Any] = {"op": P.OP_AUDIT}
        if project_id:
            req["project_id"] = project_id
        return self._call(req)

    def reindex(
        self,
        project_id: Optional[str] = None,
        resynthesize: bool = False,
        reprocess_cold: bool = False,
        background: bool = False,
    ) -> dict:
        """
        Re-embed all fragments using the current embedding model.
        Use after upgrading the embedding model or to repair a corrupted index.
        Pass project_id=None to reindex the entire database.

        background=True returns immediately with {"status": "started"}; poll
        reindex_status() for progress and the final scorecard. background=False
        blocks until complete (may exceed the socket timeout on large stores).
        """
        req: dict[str, Any] = {
            "op": P.OP_REINDEX,
            "resynthesize": resynthesize,
            "reprocess_cold": reprocess_cold,
        }
        if project_id:
            req["project_id"] = project_id
        if background:
            req["background"] = True
            return self._call(req)
        old_timeout = self.timeout
        self._sock.settimeout(300.0)
        try:
            return self._call(req)
        finally:
            self._sock.settimeout(old_timeout)

    def reindex_status(self) -> dict:
        """Poll the state of a background reindex: status (idle/running/done/error),
        current phase + progress counts, and the final scorecard report when done."""
        return self._call({"op": P.OP_REINDEX_STATUS})

    def export(self, project_id: str) -> dict:
        """
        Export all active (non-deprecated) fragments for a project.
        Returns a dict with keys: exported_at, project_id, fragment_count, fragments.
        """
        return self._call({"op": P.OP_EXPORT, "project_id": project_id})

    def savings(self) -> dict:
        """
        Retrieve token savings and analytics statistics.
        """
        return self._call({"op": P.OP_SAVINGS})

    def upgrade_status(self, project_id: Optional[str] = None) -> dict:
        """
        Read-only check: is a smarter model configured than the one that built
        the stored memory, and what would a reprocess cost? Does NOT reprocess.
        """
        req: dict[str, Any] = {"op": P.OP_UPGRADE}
        if project_id:
            req["project_id"] = project_id
        return self._call(req)

    def import_memory(
        self,
        fragments: list[dict],
        project_id: Optional[str] = None,
    ) -> dict:
        """
        Import fragments from a previously exported backup.
        Fragments whose IDs already exist in the store are skipped.
        Pass project_id to override the project_id on every imported fragment
        (useful for copying memory from one project to another).
        Blocks until complete.
        """
        req: dict[str, Any] = {"op": P.OP_IMPORT, "fragments": fragments}
        if project_id:
            req["project_id"] = project_id
        old_timeout = self.timeout
        self._sock.settimeout(300.0)
        try:
            return self._call(req)
        finally:
            self._sock.settimeout(old_timeout)

    # ── Transport ─────────────────────────────────────────────────────────────

    def _call(self, req: dict) -> dict:
        if not self._sock:
            raise MemoryClientError("Not connected. Call connect() first.")
        try:
            self._sock.sendall(P.encode(req))
            line = self._file.readline()
            if not line:
                raise MemoryClientError("Daemon closed connection unexpectedly.")
            resp = P.decode(line)
        except OSError as e:
            raise MemoryClientError(f"Connection error: {e}")
        if resp.get("status") == "error":
            raise MemoryClientError(f"Daemon error: {resp.get('message', resp)}")
        return resp
