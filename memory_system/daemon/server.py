"""
Memory daemon server — asyncio TCP, single process, background ingest queue.

Architecture (Section 3.2):
  - Read handler: fast synchronous path (vector + graph + BM25 + ranking)
  - Write handler: enqueues immediately, ACKs, processes in background
  - Background worker: asyncio task running the 4-stage compression pipeline
  - Shared embedding cache across all connected sessions

Startup:
    python -m memory_system.daemon.server [--port N] [--config PATH]

Programmatic:
    from memory_system.daemon.server import MemoryDaemon
    daemon = MemoryDaemon(cfg)
    asyncio.run(daemon.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from ..auditor import MemoryAuditor
from ..reindex import ModelUpgradeReindexJob
from ..cold_storage import append_session as _cold_append
from ..config import MemoryConfig, load_config
from ..summary_writer import generate_project_summary
from ..embedder import CachedEmbedder, OllamaEmbedder, OpenAIEmbedder, RandomEmbedder
from ..ids import new_id
from ..models import Session
from ..pipeline import ConsolidationPipeline, TranscriptMessage as _TM
from ..retrieval import fused_retrieval
from ..schema import Database
from ..scoring import init_weight_store
from ..vector_index import VectorIndex
from . import protocol as P
from .protocol import (
    OP_AUDIT, OP_EXPORT, OP_FEEDBACK, OP_FORGET, OP_IMPORT, OP_INGEST, OP_PIN,
    OP_PING, OP_REINDEX, OP_RETRIEVE, OP_SEARCH, OP_STATS, OP_SAVINGS,
)
from .state import DEFAULT_PORT, STATE_FILE, read_state, write_state, clear_state

log = logging.getLogger("memoryd")


class _BadRequest(Exception):
    """Raised by dispatch helpers to signal HTTP 400."""


# ── HTTP Server Request Handler ───────────────────────────────────────────────

class DaemonHTTPHandler(BaseHTTPRequestHandler):
    daemon: Optional[MemoryDaemon] = None
    loop: Optional[asyncio.AbstractEventLoop] = None

    def log_message(self, format, *args):
        log.info(f"HTTP: {self.address_string()} - {format % args}")

    def send_json(self, status_code: int, data: Any):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── Dispatch helpers ─────────────────────────────────────────────────────

    def _run(self, coro) -> Any:
        """Run a daemon coroutine from the HTTP thread and return the result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _api_response(self, fn, *args, **kwargs):
        """Call fn(*args, **kwargs), send JSON 200 on success, 500 on error."""
        try:
            result = fn(*args, **kwargs)
            self.send_json(200, result)
        except _BadRequest as e:
            self.send_json(400, {"status": "error", "message": str(e)})
        except Exception as e:
            self.send_json(500, {"status": "error", "message": str(e)})

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path in ("/", "/index.html"):
            self.serve_static_file("index.html", "text/html; charset=utf-8")
            return

        if not path.startswith("/api/"):
            self.send_response(404)
            self.end_headers()
            return

        self._api_response(self._dispatch_get, path)

    def _dispatch_get(self, path: str) -> dict:
        db = self.daemon._db
        if path == "/api/status":
            return self._run(self.daemon._handle_ping({}))
        if path == "/api/stats":
            return self._run(self.daemon._handle_stats({}))
        if path == "/api/fragments":
            rows = db.fetchall("SELECT * FROM memory_fragments ORDER BY created_at DESC")
            return {"status": "ok", "fragments": [dict(r) for r in rows]}
        if path == "/api/graph":
            return {
                "status": "ok",
                "nodes": [dict(r) for r in db.fetchall("SELECT * FROM entities")],
                "links": [dict(r) for r in db.fetchall("SELECT * FROM triples")],
            }
        if path == "/api/self-improvement":
            return self._run(self.daemon._handle_self_improvement({}))
        if path == "/api/savings":
            res = self.daemon._get_savings_data()
            return {
                "status": "ok",
                "summary": {k: res[k] for k in (
                    "total_sessions", "total_turns", "total_input_tokens",
                    "total_output_tokens", "tokens_saved", "cost_saved_usd",
                )},
                "sessions": res["sessions"],
            }
        raise _BadRequest(f"unknown endpoint: {path}")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        req_data: dict = {}
        if post_data:
            try:
                req_data = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                self.send_json(400, {"status": "error", "message": f"Invalid JSON: {e}"})
                return

        if not path.startswith("/api/"):
            self.send_response(404)
            self.end_headers()
            return

        self._api_response(self._dispatch_post, path, req_data)

    def _dispatch_post(self, path: str, d: dict) -> dict:
        def req(key: str, msg: str | None = None) -> Any:
            val = d.get(key, "")
            if not val:
                raise _BadRequest(msg or f"missing {key}")
            return val

        if path == "/api/feedback":
            return self._run(self.daemon._handle_feedback({
                "fragment_id": req("fragment_id", "Missing fragment_id"),
                "value": d.get("value", 0.0),
            }))
        if path == "/api/pin":
            return self._run(self.daemon._handle_pin({
                "fragment_id": req("fragment_id", "Missing fragment_id"),
                "pinned": d.get("pinned", False),
            }))
        if path == "/api/forget":
            return self._run(self.daemon._handle_forget({
                "fragment_id": req("fragment_id", "Missing fragment_id"),
            }))
        if path == "/api/audit":
            return self._run(self.daemon._handle_audit({"project_id": d.get("project_id")}))
        if path == "/api/reindex":
            return self._run(self.daemon._handle_reindex({
                "project_id":     d.get("project_id"),
                "resynthesize":   d.get("resynthesize", False),
                "reprocess_cold": d.get("reprocess_cold", False),
            }))
        if path == "/api/obsidian/correct":
            return self._run(self.daemon._handle_obsidian_correct({
                "original_fact":  req("original_fact", "missing original_fact or corrected_fact").strip(),
                "corrected_fact": req("corrected_fact", "missing original_fact or corrected_fact").strip(),
                "project_id":     d.get("project_id"),
            }))
        if path == "/api/obsidian/note":
            return self._run(self.daemon._handle_obsidian_note({
                "content":     req("content", "missing content or project_id").strip(),
                "project_id":  req("project_id", "missing content or project_id"),
                "source_name": d.get("source_name", "obsidian-note"),
            }))
        if path == "/api/obsidian/export":
            return self._run(self.daemon._handle_obsidian_export({
                "vault_path": req("vault_path", "missing vault_path").strip(),
                "project_id": d.get("project_id"),
            }))
        raise _BadRequest(f"unknown endpoint: {path}")

    def serve_static_file(self, filename: str, content_type: str):
        web_dir = Path(__file__).parent / "web"
        file_path = web_dir / filename
        if not file_path.exists() or not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            log.error(f"Failed to serve static file {filename}: {e}")


class MemoryDaemon:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="memoryd-worker"
        )
        self._ingest_queue: asyncio.Queue = asyncio.Queue(
            maxsize=cfg.daemon.write_queue_size
        )
        self._db: Optional[Database] = None
        self._idx: Optional[VectorIndex] = None
        self._embedder: Optional[CachedEmbedder] = None
        self._pipeline: Optional[ConsolidationPipeline] = None
        self._auditor: Optional[MemoryAuditor] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False
        self._dirty_projects: set[str] = set()
        self._web_server: Optional[ThreadingHTTPServer] = None
        self._web_thread: Optional[threading.Thread] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def run(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self._init_storage()
        self._running = True

        log.info("memoryd starting on %s:%d", host, port)
        self._server = await asyncio.start_server(
            self._handle_connection, host, port
        )

        self._start_web_server(host, port + 1)

        _write_state(host, port)
        worker_task   = asyncio.create_task(self._background_worker())
        audit_task    = asyncio.create_task(self._audit_scheduler())
        summary_task  = asyncio.create_task(self._summary_scheduler())

        async with self._server:
            try:
                await self._server.serve_forever()
            except asyncio.CancelledError:
                pass
            finally:
                self._running = False
                for task in (worker_task, audit_task, summary_task):
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                self._shutdown()

    def stop(self) -> None:
        if self._server:
            self._server.close()

    def _init_storage(self) -> None:
        db_path = Path(self.cfg.storage.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = Database(db_path)
        self._db.connect()

        index_path = db_path.parent / "vector.hnsw"
        self._idx = VectorIndex(
            dim=self.cfg.embedding.dimensions,
            persist_path=index_path,
        )
        if not self._idx.load():
            self._idx.init_fresh()

        self._embedder = _build_embedder(self.cfg)
        from ..llm import LLMRouter
        router = LLMRouter(self.cfg.llm_roles)
        self._pipeline = ConsolidationPipeline(
            self._db, self._idx, self.cfg.compression, self._embedder, router=router
        )
        # v4 reranker: install the learned-weight store once, on the live
        # connection. init_weight_store sets the module-level store the scorer
        # reads (read path) AND returns it to hand to the auditor's Pass 10
        # (learn path) — one call wires both ends of the loop.
        weight_store = init_weight_store(self._db._conn, self._db._lock)
        self._auditor = MemoryAuditor(
            self._db, self.cfg.self_improvement, self._embedder,
            eviction_cfg=self.cfg.eviction,
            storage_cfg=self.cfg.storage,
            weight_store=weight_store,
            router=router,
        )

    def _shutdown(self) -> None:
        log.info("memoryd shutting down")
        if self._idx:
            try:
                self._idx.save()
            except Exception as e:
                log.warning("vector index save failed: %s", e)
        self._stop_web_server()
        if self._db:
            self._db.close()
        _clear_state()
        self._executor.shutdown(wait=False)

    def _start_web_server(self, host: str, port: int) -> None:
        DaemonHTTPHandler.daemon = self
        DaemonHTTPHandler.loop = asyncio.get_running_loop()
        try:
            self._web_server = ThreadingHTTPServer((host, port), DaemonHTTPHandler)
            self._web_thread = threading.Thread(
                target=self._web_server.serve_forever,
                name="memoryd-web",
                daemon=True
            )
            self._web_thread.start()
            log.info("Web Dashboard server started on http://%s:%d", host, port)
        except Exception as e:
            log.error("Failed to start Web Dashboard server: %s", e)

    def _stop_web_server(self) -> None:
        if self._web_server:
            log.info("Stopping Web Dashboard server")
            self._web_server.shutdown()
            self._web_server.server_close()
            self._web_server = None

    # ── Connection handler ───────────────────────────────────────────────────

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        log.debug("client connected: %s", peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    req = P.decode(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    resp = P.error(f"bad request: {e}")
                    writer.write(P.encode(resp))
                    await writer.drain()
                    continue

                resp = await self._dispatch(req)
                writer.write(P.encode(resp))
                await writer.drain()
        except ConnectionResetError:
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            log.debug("client disconnected: %s", peer)

    # ── Request dispatcher ───────────────────────────────────────────────────

    async def _dispatch(self, req: dict) -> dict:
        op = req.get("op", "")
        handlers = {
            OP_PING:     self._handle_ping,
            OP_RETRIEVE: self._handle_retrieve,
            OP_INGEST:   self._handle_ingest,
            OP_STATS:    self._handle_stats,
            OP_SEARCH:   self._handle_search,
            OP_PIN:      self._handle_pin,
            OP_FORGET:   self._handle_forget,
            OP_AUDIT:    self._handle_audit,
            OP_REINDEX:  self._handle_reindex,
            OP_EXPORT:   self._handle_export,
            OP_IMPORT:   self._handle_import,
            OP_FEEDBACK: self._handle_feedback,
            OP_SAVINGS:  self._handle_savings,
        }
        handler = handlers.get(op)
        if not handler:
            return P.error(f"unknown op: {op!r}")
        try:
            return await handler(req)
        except Exception as e:
            log.exception("handler error for op=%r", op)
            return P.error(str(e))

    # ── Read handlers (fast path) ────────────────────────────────────────────

    async def _handle_ping(self, req: dict) -> dict:
        return P.ok(message="pong", pid=os.getpid())

    async def _handle_retrieve(self, req: dict) -> dict:
        err = P.validate_retrieve(req)
        if err:
            return P.error(err)

        project_id    = req["project_id"]
        query_text    = req["query_text"]
        query_emb     = req.get("query_embedding") or []
        token_budget  = req.get("max_tokens", self.cfg.retrieval.default_token_budget)
        scopes        = req.get("scopes", ["project", "global"])
        min_crs       = req.get("min_crs", self.cfg.retrieval.min_crs)

        # If client didn't pre-compute embedding, do it here
        if not query_emb and query_text:
            loop = asyncio.get_running_loop()
            query_emb = await loop.run_in_executor(
                self._executor, self._embedder.embed, query_text
            )

        # Run retrieval in thread pool (DB + HNSW are blocking)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            fused_retrieval,
            self._db,
            self._idx,
            query_emb,
            query_text,
            project_id,
            token_budget,
            self.cfg.retrieval,
            "global" in scopes,
        )

        return P.ok(
            fragments=[
                {
                    "id":             f.id,
                    "scope":          f.scope,
                    "content":        f.content,
                    "crs":            round(f.crs, 4),
                    "source_episode": f.source_session,
                    "token_count":    f.token_count,
                    "confidence":     round(f.confidence, 4),
                    "created_at":     f.created_at[:10],
                }
                for f in result.fragments
            ],
            project_summary=result.project_summary,
            global_profile_hash=result.global_profile_hash,
            token_budget_used=result.token_budget_used,
        )

    async def _handle_stats(self, req: dict) -> dict:
        project_id = req.get("project_id")
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            self._executor, self._db.stats, project_id
        )
        return P.ok(
            stats=stats,
            vector_index_size=self._idx.size,
            ingest_queue_depth=self._ingest_queue.qsize(),
        )

    async def _handle_search(self, req: dict) -> dict:
        query   = req.get("query", "")
        project = req.get("project_id")
        if not query:
            return P.error("missing query")

        loop = asyncio.get_running_loop()
        emb = await loop.run_in_executor(
            self._executor, self._embedder.embed, query
        )
        result = await loop.run_in_executor(
            self._executor,
            fused_retrieval,
            self._db, self._idx, emb, query,
            project or "__global__",
            self.cfg.retrieval.default_token_budget,
            self.cfg.retrieval,
            project is None,
        )
        return P.ok(
            fragments=[{"id": f.id, "content": f.content, "crs": round(f.crs, 4)}
                       for f in result.fragments]
        )

    async def _handle_pin(self, req: dict) -> dict:
        err = P.validate_pin(req)
        if err:
            return P.error(err)
        fid = req["fragment_id"]
        pinned = req.get("pinned", True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._db.execute,
            "UPDATE memory_fragments SET is_pinned=? WHERE id=?",
            (int(pinned), fid),
        )
        return P.ok(fragment_id=fid, pinned=pinned)

    async def _handle_forget(self, req: dict) -> dict:
        err = P.validate_forget(req)
        if err:
            return P.error(err)
        fid = req["fragment_id"]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._db.mark_deprecated,
            fid,
            None,   # user-initiated forget; NULL satisfies the FK constraint
        )
        await loop.run_in_executor(self._executor, self._idx.remove, fid)
        return P.ok(fragment_id=fid, forgotten=True)

    async def _handle_feedback(self, req: dict) -> dict:
        fid   = req.get("fragment_id")
        if not fid:
            return P.error("missing field: fragment_id")
        value = max(-1.0, min(1.0, float(req.get("value", 0.0))))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._db.execute,
            "UPDATE memory_fragments SET user_feedback=? WHERE id=?",
            (value, fid),
        )
        return P.ok(fragment_id=fid, user_feedback=value)

    async def _handle_audit(self, req: dict) -> dict:
        project_id = req.get("project_id")
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            self._executor, self._auditor.audit, project_id
        )
        return P.ok(
            contradictions_found=report.contradictions_found,
            fragments_decayed=report.fragments_decayed,
            orphans_pruned=report.orphans_pruned,
            triples_removed=report.triples_removed,
            centrality_updated=report.centrality_updated,
            evicted=report.evicted,
            patterns_articulated=report.patterns_articulated,
            simulations_run=report.simulations_run,
            crystallized=report.crystallized,
            meta_insights=report.meta_insights,
            llm_calls=report.llm_calls,
        )

    async def _handle_reindex(self, req: dict) -> dict:
        project_id     = req.get("project_id")
        resynthesize   = bool(req.get("resynthesize", False))
        reprocess_cold = bool(req.get("reprocess_cold", False))
        loop = asyncio.get_running_loop()
        job = ModelUpgradeReindexJob(
            db=self._db,
            vector_index=self._idx,
            new_embedder=self._embedder,
            cfg=self.cfg.self_improvement,
            resynthesize_facts=resynthesize,
            reprocess_cold=reprocess_cold,
            pipeline=self._pipeline,
            cold_storage_path=self.cfg.storage.cold_storage_path,
        )
        report = await loop.run_in_executor(self._executor, job.execute, project_id)
        return P.ok(
            fragments_reindexed=report.fragments_reindexed,
            facts_resynthesized=report.facts_resynthesized,
            cold_sessions_reprocessed=report.cold_sessions_reprocessed,
            errors=report.errors,
        )

    async def _handle_export(self, req: dict) -> dict:
        project_id = req.get("project_id")
        if not project_id:
            return P.error("missing field: project_id")
        loop = asyncio.get_running_loop()
        fragments = await loop.run_in_executor(
            self._executor,
            self._db.list_fragments,
            project_id, None, None, False,
        )
        now = datetime.now(tz=timezone.utc).isoformat()
        return P.ok(
            exported_at=now,
            project_id=project_id,
            fragment_count=len(fragments),
            fragments=[
                {
                    "id":             f.id,
                    "project_id":     f.project_id,
                    "scope":          f.scope,
                    "category":       f.category,
                    "content":        f.content,
                    "token_count":    f.token_count,
                    "abstraction_lvl": f.abstraction_lvl,
                    "confidence":     f.confidence,
                    "source_type":    f.source_type,
                    "source_session": f.source_session,
                    "created_at":     f.created_at,
                    "last_accessed":  f.last_accessed,
                    "access_count":   f.access_count,
                    "is_pinned":      bool(f.is_pinned),
                    "graph_centrality": f.graph_centrality,
                    "user_feedback":  f.user_feedback,
                    "metadata_json":  f.metadata_json,
                }
                for f in fragments
            ],
        )

    async def _handle_self_improvement(self, req: dict) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_self_improvement_data)

    def _get_self_improvement_data(self) -> dict:
        db = self._db
        runs = db.fetchall("SELECT * FROM daemon_runs ORDER BY started_at DESC LIMIT 10")
        insights = db.fetchall(
            "SELECT content, confidence, created_at FROM memory_fragments "
            "WHERE source_type='meta_learning' AND is_deprecated=0 "
            "ORDER BY created_at DESC LIMIT 20"
        )
        row_corr  = db.fetchone("SELECT COUNT(*) AS n FROM corrections")
        row_meta  = db.fetchone(
            "SELECT COUNT(*) AS n FROM memory_fragments "
            "WHERE source_type='meta_learning' AND is_deprecated=0"
        )
        row_cryst = db.fetchone(
            "SELECT COUNT(*) AS n FROM memory_fragments "
            "WHERE source_type='crystallization' AND is_deprecated=0"
        )
        si = self.cfg.self_improvement
        return {
            "status": "ok",
            "models": {
                "embedding":            self.cfg.embedding.model,
                "contradiction":        si.contradiction_model,
                "meta_learn":           si.meta_learn_model,
                "crystallization":      si.crystallization_model,
                "pattern_articulation": si.pattern_articulation_model,
                "rem":                  si.rem_model,
            },
            "flags": {
                "contradiction_detection":      si.contradiction_detection,
                "meta_learn_enabled":           si.meta_learn_enabled,
                "crystallization_enabled":      si.crystallization_enabled,
                "pattern_articulation_enabled": si.pattern_articulation_enabled,
                "rem_enabled":                  si.rem_enabled,
            },
            "recent_runs":    [dict(r) for r in runs],
            "recent_insights": [dict(r) for r in insights],
            "totals": {
                "corrections":   row_corr["n"]  if row_corr  else 0,
                "meta_insights": row_meta["n"]  if row_meta  else 0,
                "crystallized":  row_cryst["n"] if row_cryst else 0,
            },
        }

    async def _handle_savings(self, req: dict) -> dict:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(self._executor, self._get_savings_data)
        return P.ok(**res)

    def _get_savings_data(self) -> dict:
        sessions = [dict(r) for r in self._db.fetchall("SELECT * FROM sessions ORDER BY started_at DESC")]
        total_sessions = len(sessions)
        total_turns = sum(s.get("turn_count", 0) or 0 for s in sessions)
        total_input_tokens = sum(s.get("cost_input_tok", 0) or 0 for s in sessions)
        total_output_tokens = sum(s.get("cost_output_tok", 0) or 0 for s in sessions)

        if total_input_tokens > 0:
            # Real token data: estimate savings from 4.3x context compression ratio
            raw_input_estimate = int(total_input_tokens * 4.3)
            tokens_saved = max(0, raw_input_estimate - total_input_tokens)
        else:
            # No API token data yet — estimate from fragment token counts.
            # Each fragment token represents compressed memory that replaced a
            # much larger raw conversation chunk (empirically ~8x ratio).
            row = self._db.fetchone(
                "SELECT COALESCE(SUM(token_count), 0) AS n FROM memory_fragments WHERE is_deprecated=0"
            )
            fragment_tokens = int(row["n"]) if row else 0
            tokens_saved = fragment_tokens * 7  # (8x original - 1x stored = 7x saved)

        cost_saved_usd = round((tokens_saved / 1_000_000) * 5.40, 2)

        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tokens_saved": tokens_saved,
            "cost_saved_usd": cost_saved_usd,
            "sessions": sessions,
        }

    # ── Obsidian bridge handlers ──────────────────────────────────────────────

    async def _handle_obsidian_correct(self, req: dict) -> dict:
        from ..models import Correction
        from ..ids import new_id
        now = datetime.now(timezone.utc).isoformat()
        c = Correction(
            id=new_id(),
            original_fact=req["original_fact"],
            corrected_fact=req["corrected_fact"],
            project_id=req.get("project_id"),
            created_at=now,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._db.insert_correction, c)
        return P.ok(correction_id=c.id, message="correction inserted")

    async def _handle_obsidian_note(self, req: dict) -> dict:
        from ..project import new_session_id as _new_sid
        project_id  = req["project_id"]
        content     = req["content"]
        source_name = req.get("source_name", "obsidian-note")
        now = datetime.now(timezone.utc).isoformat()
        ingest_req = {
            "op": "ingest",
            "project_id": project_id,
            "session_id": _new_sid(),
            "transcript_segments": [
                {
                    "role": "user",
                    "content": f"[{source_name}] {content}",
                    "timestamp": now,
                    "is_correction": False,
                    "is_error_fix": False,
                    "is_explicit_remember": True,
                    "tool_call_chain_len": 0,
                }
            ],
            "priority": "immediate",
        }
        loop = asyncio.get_running_loop()
        # Archive best-effort: the durable source of a note is the Obsidian vault,
        # so a cold-storage hiccup must not fail note creation.
        try:
            await loop.run_in_executor(self._executor, self._archive_raw_source, ingest_req)
        except Exception as e:
            log.warning("obsidian note cold archival failed: %s", e)
        stats = await loop.run_in_executor(self._executor, self._process_ingest, ingest_req)
        return P.ok(**stats)

    async def _handle_obsidian_export(self, req: dict) -> dict:
        vault_path = req["vault_path"]
        project_id = req.get("project_id")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._do_obsidian_export,
            vault_path,
            project_id,
        )
        return P.ok(**result)

    def _do_obsidian_export(self, vault_path: str, project_id: str | None) -> dict:
        from ..obsidian_bridge.exporter import ObsidianExporter
        exporter = ObsidianExporter(vault_path, cfg=self.cfg, project_id=project_id)
        return exporter.export()

    async def _handle_import(self, req: dict) -> dict:
        raw_fragments = req.get("fragments")
        if not isinstance(raw_fragments, list):
            return P.error("missing or invalid field: fragments (must be a list)")
        project_override = req.get("project_id")

        loop = asyncio.get_running_loop()
        imported, skipped, errors = await loop.run_in_executor(
            self._executor, self._do_import, raw_fragments, project_override
        )
        return P.ok(imported=imported, skipped=skipped, errors=errors)

    def _do_import(
        self,
        raw_fragments: list[dict],
        project_override: Optional[str],
    ) -> tuple[int, int, int]:
        """Blocking — runs in thread pool executor."""
        from ..models import MemoryFragment as MF

        imported = skipped = errors = 0
        new_fragments: list[MF] = []

        for d in raw_fragments:
            try:
                fid = d.get("id", "")
                if not fid or self._db.get_fragment(fid):
                    skipped += 1
                    continue
                project_id = project_override or d.get("project_id", "")
                if not project_id:
                    errors += 1
                    continue
                f = MF(
                    id=fid,
                    project_id=project_id,
                    scope=d.get("scope", "project"),
                    category=d.get("category", "fact"),
                    content=d.get("content", ""),
                    token_count=int(d.get("token_count", 0)),
                    abstraction_lvl=float(d.get("abstraction_lvl", 0.5)),
                    confidence=float(d.get("confidence", 0.8)),
                    source_type=d.get("source_type"),
                    source_session=d.get("source_session"),
                    created_at=d.get("created_at", datetime.now(tz=timezone.utc).isoformat()),
                    last_accessed=d.get("last_accessed", datetime.now(tz=timezone.utc).isoformat()),
                    access_count=int(d.get("access_count", 0)),
                    is_pinned=bool(d.get("is_pinned", False)),
                    graph_centrality=float(d.get("graph_centrality", 0.0)),
                    user_feedback=float(d.get("user_feedback", 0.0)),
                    metadata_json=d.get("metadata_json"),
                )
                self._db.upsert_fragment(f)
                new_fragments.append(f)
                imported += 1
            except Exception as e:
                log.warning("import: skipping fragment due to error: %s", e)
                errors += 1

        # Re-embed and index all newly imported fragments
        if new_fragments:
            _BATCH = 50
            for i in range(0, len(new_fragments), _BATCH):
                batch = new_fragments[i : i + _BATCH]
                try:
                    vecs = self._embedder.embed_batch([f.content for f in batch])
                    for frag, vec in zip(batch, vecs):
                        self._idx.add(frag.id, vec)
                except Exception as e:
                    log.warning("import: embed_batch failed at offset %d: %s", i, e)
            try:
                self._idx.save()
            except Exception as e:
                log.warning("import: vector index save failed: %s", e)

        return imported, skipped, errors

    # (mark_deprecated accepts None as deprecated_by for user-initiated forgets)

    # ── Write handler (async path) ────────────────────────────────────────────

    async def _handle_ingest(self, req: dict) -> dict:
        err = P.validate_ingest(req)
        if err:
            return P.error(err)

        # Secure the raw source to cold storage BEFORE acknowledging the ingest.
        # The Stop hook advances its per-session line pointer the moment we ACK
        # (queued/ok), so if archival ran later in the background worker a failure
        # would silently lose the transcript for good — while the lossy distillation
        # became the only survivor (the exact "fast layer destroys the slow layer's
        # input" failure). Rejecting here makes the hook keep its pointer and
        # re-send these lines next turn. Distillation stays in the background worker
        # because it is recomputable from cold storage; the raw Record is not.
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._archive_raw_source, req)
        except Exception as e:
            log.error("cold storage archival failed; rejecting ingest to preserve source: %s", e)
            return P.error(f"raw transcript archival failed: {e}")

        priority = req.get("priority", "background")
        if priority == "immediate":
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(
                self._executor, self._process_ingest, req
            )
            stats["queue_depth"] = self._ingest_queue.qsize()
            return P.ok(**stats)

        try:
            self._ingest_queue.put_nowait(req)
        except asyncio.QueueFull:
            return P.error("ingest queue full — daemon overloaded")

        return P.queued()

    # ── Background worker ────────────────────────────────────────────────────

    async def _background_worker(self) -> None:
        log.info("background ingest worker started")
        loop = asyncio.get_running_loop()
        while True:
            try:
                req = await self._ingest_queue.get()
                log.debug(
                    "processing ingest for project=%s session=%s",
                    req.get("project_id"), req.get("session_id"),
                )
                try:
                    await loop.run_in_executor(
                        self._executor, self._process_ingest, req
                    )
                except Exception as e:
                    log.error("ingest pipeline failed: %s", e)
                finally:
                    self._ingest_queue.task_done()
            except asyncio.CancelledError:
                break

        log.info("background worker stopped")

    async def _summary_scheduler(self) -> None:
        """Regenerates project summaries for projects that received new ingest."""
        interval_secs = self.cfg.self_improvement.summary_refresh_interval_minutes * 60
        log.info(
            "summary scheduler started (interval=%dm)",
            self.cfg.self_improvement.summary_refresh_interval_minutes,
        )
        try:
            while True:
                await asyncio.sleep(interval_secs)
                if not self._dirty_projects:
                    continue
                dirty = self._dirty_projects.copy()
                self._dirty_projects.clear()
                log.info("regenerating summaries for %d project(s)", len(dirty))
                loop = asyncio.get_running_loop()
                for pid in dirty:
                    try:
                        await loop.run_in_executor(
                            self._executor,
                            generate_project_summary,
                            self._db,
                            pid,
                            self.cfg.compression.distillation_model,
                        )
                    except Exception as e:
                        log.error("summary generation failed for %s: %s", pid, e)
        except asyncio.CancelledError:
            pass
        log.info("summary scheduler stopped")

    async def _audit_scheduler(self) -> None:
        """Runs MemoryAuditor at the configured interval (default weekly)."""
        interval_secs = self.cfg.self_improvement.audit_interval_hours * 3600
        log.info(
            "audit scheduler started (interval=%dh)",
            self.cfg.self_improvement.audit_interval_hours,
        )
        try:
            while True:
                await asyncio.sleep(interval_secs)
                log.info("running scheduled memory audit")
                loop = asyncio.get_running_loop()
                try:
                    report = await loop.run_in_executor(
                        self._executor, self._auditor.audit, None
                    )
                    log.info("scheduled audit done: %s", report)
                except Exception as e:
                    log.error("scheduled audit failed: %s", e)
        except asyncio.CancelledError:
            pass
        log.info("audit scheduler stopped")

    def _archive_raw_source(self, req: dict) -> None:
        """Durably archive the raw transcript to cold storage. Raises on failure.

        Cold storage is the system's only durable copy of the slow-layer input.
        This runs BEFORE the ingest is acknowledged (see _handle_ingest) so the
        raw Record is on disk before the hook can advance past these lines.
        """
        raw_segs = req.get("transcript_segments") or []
        if not raw_segs:
            return
        dest = _cold_append(
            self.cfg.storage.cold_storage_path,
            req["project_id"],
            req["session_id"],
            raw_segs,
            model_id=req.get("model_id"),
            summary=req.get("session_summary"),
            input_tokens=int(req.get("input_tokens") or 0),
            output_tokens=int(req.get("output_tokens") or 0),
        )
        log.debug("cold storage written: %s", dest)

    def _process_ingest(self, req: dict) -> dict:
        """Blocking — runs in thread pool executor. Returns ingest statistics.

        Raw-source archival is NOT done here: it happens before ACK in
        _handle_ingest so a failure is surfaced to the client. This method only
        produces the recomputable distillation, which can be rebuilt from cold
        storage via reindex if it is ever lost.
        """
        project_id  = req["project_id"]
        session_id  = req["session_id"]
        raw_segs    = req["transcript_segments"]
        model_id    = req.get("model_id")
        summary     = req.get("session_summary")
        cost_in     = int(req.get("input_tokens") or 0)
        cost_out    = int(req.get("output_tokens") or 0)

        # Ensure session exists
        now = datetime.now(tz=timezone.utc).isoformat()
        sess = Session(
            id=session_id,
            project_id=project_id,
            started_at=now,
            ended_at=now,
            summary=summary,
            turn_count=len(raw_segs),
            model_id=model_id,
            cost_input_tok=cost_in,
            cost_output_tok=cost_out,
        )
        self._db.upsert_session(sess)

        # Convert raw segment dicts → TranscriptMessage
        messages = [
            _TM(
                role=seg.get("role", "user"),
                content=str(seg.get("content", "")),
                timestamp=seg.get("timestamp"),
                is_correction=bool(seg.get("is_correction")),
                is_error_fix=bool(seg.get("is_error_fix")),
                is_explicit_remember=bool(seg.get("is_explicit_remember")),
                tool_call_chain_len=int(seg.get("tool_call_chain_len", 0)),
            )
            for seg in raw_segs
            if seg.get("content")
        ]

        if not messages:
            return {"messages_processed": 0, "fragments_created": 0, "episodes_created": 0, "facts_created": 0}

        new_fragments = self._pipeline.ingest(messages, session_id=session_id, project_id=project_id)

        # Persist vector index after each successful ingest
        try:
            self._idx.save()
        except Exception as e:
            log.warning("vector index save failed: %s", e)

        # Raw transcript was archived to cold storage before ACK (see
        # _handle_ingest / _archive_raw_source) — not here.

        # Mark project as needing summary refresh
        self._dirty_projects.add(project_id)

        stats = {
            "messages_processed": len(messages),
            "fragments_created": len(new_fragments),
            "episodes_created": sum(1 for f in new_fragments if f.category == "episode"),
            "facts_created": sum(1 for f in new_fragments if f.category == "fact"),
        }
        log.debug(
            "ingest complete: project=%s session=%s messages=%d fragments=%d",
            project_id, session_id, stats["messages_processed"], stats["fragments_created"],
        )
        return stats


# ── State file helpers (delegated to state.py) ───────────────────────────────

def _write_state(host: str, port: int) -> None:
    write_state(host, port)


def _clear_state() -> None:
    clear_state()


# ── Embedder factory ──────────────────────────────────────────────────────────

def _build_embedder(cfg: MemoryConfig) -> CachedEmbedder:
    model = cfg.embedding.model

    if model.startswith("ollama/"):
        local_model = model[len("ollama/"):]
        inner = OllamaEmbedder(model=local_model, dim=cfg.embedding.dimensions)
        log.info("embedder: Ollama %s dim=%d", inner.model_name, inner.dim)
        return CachedEmbedder(inner, max_entries=cfg.embedding.cache_size)

    if model.startswith("text-embedding") and os.environ.get("OPENAI_API_KEY"):
        try:
            inner = OpenAIEmbedder(batch_size=cfg.embedding.batch_size)
            log.info("embedder: OpenAI %s dim=%d", inner.model_name, inner.dim)
            return CachedEmbedder(inner, max_entries=cfg.embedding.cache_size)
        except ImportError:
            log.warning("openai SDK not installed — falling back to RandomEmbedder")

    log.warning("no embedder matched config model=%r — using RandomEmbedder", model)
    inner = RandomEmbedder(dim=cfg.embedding.dimensions)
    return CachedEmbedder(inner, max_entries=cfg.embedding.cache_size)


# ── __main__ entry point ──────────────────────────────────────────────────────

async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Memory daemon")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    daemon = MemoryDaemon(cfg)

    loop = asyncio.get_running_loop()

    def _stop():
        log.info("shutdown signal received")
        daemon.stop()

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, _stop)
        loop.add_signal_handler(signal.SIGINT, _stop)

    await daemon.run(host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(_main())
