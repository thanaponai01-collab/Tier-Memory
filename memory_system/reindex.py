"""
ModelUpgradeReindexJob — re-embed all fragments with a new embedding model.

When you upgrade your embedding model (e.g., from text-embedding-3-small to
text-embedding-3-large), all stored vectors are stale and must be regenerated.
This job:
  1. Iterates all non-deprecated fragments in batches
  2. Calls the new embedder for each batch
  3. Replaces the vector index entry
  4. Updates embedding_model / embedding_dim on the DB row
  5. Optionally re-synthesizes semantic facts using a smarter LLM

Usage:
    job = ModelUpgradeReindexJob(
        db=db,
        vector_index=idx,
        new_embedder=CachedEmbedder(OllamaEmbedder()),
        cfg=cfg.self_improvement,
    )
    report = job.execute(project_id="my-app")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .cold_storage import read_session_from_path
from .config import SelfImprovementConfig
from .embedder import CachedEmbedder, OllamaEmbedder
from .ids import new_id
from .llm import LLMRouter, PRODUCER_VERSION, call_model_json
from .models import MemoryFragment
from .pipeline import ConsolidationPipeline, TranscriptMessage
from .schema import Database
from .vector_index import VectorIndex

log = logging.getLogger("memory.reindex")

_RESYNTHESIS_MODEL = "claude-haiku-4-5-20251001"
_BATCH_SIZE = 50


@dataclass
class HealthSnapshot:
    """A read-only picture of memory health, taken before and after a crank so
    we can show whether the run actually improved the store or just churned."""
    active_fragments: int = 0
    avg_confidence: float = 0.0
    low_confidence_facts: int = 0   # facts below the 0.70 resynthesis threshold
    open_goals: int = 0
    deprecated_fragments: int = 0

    def as_dict(self) -> dict:
        return {
            "active_fragments": self.active_fragments,
            "avg_confidence": round(self.avg_confidence, 4),
            "low_confidence_facts": self.low_confidence_facts,
            "open_goals": self.open_goals,
            "deprecated_fragments": self.deprecated_fragments,
        }


@dataclass
class ReindexReport:
    fragments_reindexed: int = 0
    facts_resynthesized: int = 0
    cold_sessions_reprocessed: int = 0
    errors: int = 0
    before: Optional[HealthSnapshot] = None
    after: Optional[HealthSnapshot] = None

    def __repr__(self) -> str:
        return (
            f"ReindexReport(reindexed={self.fragments_reindexed}, "
            f"resynthesized={self.facts_resynthesized}, "
            f"cold_reprocessed={self.cold_sessions_reprocessed}, "
            f"errors={self.errors})"
        )


class ModelUpgradeReindexJob:
    def __init__(
        self,
        db: Database,
        vector_index: VectorIndex,
        new_embedder: CachedEmbedder,
        cfg: SelfImprovementConfig,
        resynthesize_facts: bool = False,
        reprocess_cold: bool = False,
        pipeline: Optional[ConsolidationPipeline] = None,
        cold_storage_path: Optional[str] = None,
        router: Optional[LLMRouter] = None,
        resynthesis_role: str = "strong",
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.db = db
        self.idx = vector_index
        self.embedder = new_embedder
        self.cfg = cfg
        self.resynthesize_facts = resynthesize_facts
        self.reprocess_cold = reprocess_cold
        self._pipeline = pipeline
        self.cold_storage_path = cold_storage_path
        # Resynthesis routes through the configured 'strong' role — the point of
        # a model upgrade is to re-judge memory with the SHARPEST reachable model,
        # not a local no-op. (A weak local model only reshuffles words; a frontier
        # model genuinely clarifies — verified on a live taste, 2026-06-05.) Falls
        # back through LLMRouter if 'strong' is unset.
        self._router = router
        self._resynthesis_role = resynthesis_role
        # Called as progress_cb(phase, done, total) so a daemon can report a
        # long-running reindex to a client that's polling for status.
        self._progress_cb = progress_cb

    def _report(self, phase: str, done: int, total: int) -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(phase, done, total)
            except Exception:
                pass  # progress reporting must never break the job

    def execute(self, project_id: Optional[str] = None) -> ReindexReport:
        """
        Re-embed all non-deprecated fragments.  Pass project_id=None to
        reindex the entire database (e.g., after a global model upgrade).
        """
        report = ReindexReport()
        report.before = self._snapshot_health(project_id)
        total_active = report.before.active_fragments
        log.info("reindex starting project=%s", project_id or "*")
        self._report("re-embedding", 0, total_active)

        offset = 0
        while True:
            batch = self._fetch_batch(project_id, offset, _BATCH_SIZE)
            if not batch:
                break
            offset += len(batch)

            texts   = [f.content for f in batch]
            try:
                vectors = self.embedder.embed_batch(texts)
            except Exception as e:
                log.error("embed_batch failed (offset=%d): %s", offset, e)
                report.errors += len(batch)
                continue

            for fragment, vec in zip(batch, vectors):
                try:
                    self._update_fragment(fragment, vec)
                    report.fragments_reindexed += 1
                except Exception as e:
                    log.warning("failed to update fragment %s: %s", fragment.id, e)
                    report.errors += 1
            self._report("re-embedding", report.fragments_reindexed, total_active)

        if self.resynthesize_facts:
            report.facts_resynthesized = self._resynthesize_facts(project_id)

        if self.reprocess_cold and self._pipeline and self.cold_storage_path:
            report.cold_sessions_reprocessed = self._reprocess_cold_storage(
                project_id,
                self._pipeline,
                limit=self.cfg.reindex_cold_sessions_limit,
            )

        # Persist the updated vector index
        try:
            self.idx.save()
        except Exception as e:
            log.warning("vector index save failed: %s", e)

        report.after = self._snapshot_health(project_id)
        log.info("reindex complete: %s", report)
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _snapshot_health(self, project_id: Optional[str]) -> HealthSnapshot:
        """Read-only health metrics for the scope being reindexed. Never mutates."""
        snap = HealthSnapshot()
        # Scope clause: a specific project, or the whole store when project_id is None.
        where = "project_id = ?" if project_id else "1=1"
        params: tuple = (project_id,) if project_id else ()
        try:
            row = self.db.fetchone(
                f"""SELECT COUNT(*) AS n, COALESCE(AVG(confidence), 0.0) AS avg_c
                    FROM memory_fragments
                    WHERE is_deprecated = 0 AND {where}""",
                params,
            )
            if row:
                snap.active_fragments = row["n"]
                snap.avg_confidence = float(row["avg_c"])

            row = self.db.fetchone(
                f"""SELECT COUNT(*) AS n FROM memory_fragments
                    WHERE is_deprecated = 0 AND category = 'fact'
                      AND confidence < 0.70 AND {where}""",
                params,
            )
            snap.low_confidence_facts = row["n"] if row else 0

            row = self.db.fetchone(
                f"""SELECT COUNT(*) AS n FROM memory_fragments
                    WHERE is_deprecated = 1 AND {where}""",
                params,
            )
            snap.deprecated_fragments = row["n"] if row else 0

            row = self.db.fetchone(
                f"""SELECT COUNT(*) AS n FROM goals
                    WHERE status = 'open' AND {where}""",
                params,
            )
            snap.open_goals = row["n"] if row else 0
        except Exception as e:
            log.warning("health snapshot failed (project=%s): %s", project_id, e)
        return snap

    def _fetch_batch(
        self,
        project_id: Optional[str],
        offset: int,
        limit: int,
    ) -> list[MemoryFragment]:
        if project_id:
            rows = self.db.fetchall(
                """SELECT * FROM memory_fragments
                   WHERE project_id=? AND is_deprecated=0
                   ORDER BY rowid LIMIT ? OFFSET ?""",
                (project_id, limit, offset),
            )
        else:
            rows = self.db.fetchall(
                """SELECT * FROM memory_fragments
                   WHERE is_deprecated=0
                   ORDER BY rowid LIMIT ? OFFSET ?""",
                (limit, offset),
            )
        from .schema import _row_to_fragment
        return [_row_to_fragment(r) for r in rows]

    def _update_fragment(self, fragment: MemoryFragment, vec: list[float]) -> None:
        model_name = getattr(self.embedder, "model_name", "unknown")
        dim        = len(vec)

        # Replace in vector index (remove old, add new)
        try:
            self.idx.remove(fragment.id)
        except Exception:
            pass
        self.idx.add(fragment.id, vec)

        # Update DB metadata
        self.db.execute(
            """UPDATE memory_fragments
               SET embedding_model=?, embedding_dim=?
               WHERE id=?""",
            (model_name, dim, fragment.id),
        )

    def _reprocess_cold_storage(
        self,
        project_id: Optional[str],
        pipeline: ConsolidationPipeline,
        limit: int = 500,
    ) -> int:
        """
        Decompress recent cold sessions and run them through the consolidation
        pipeline again with the current (smarter) model. Returns session count processed.
        """
        cold_root = Path(self.cold_storage_path)
        pattern = f"{project_id or '*'}/*/*.jsonl.zst"
        paths = sorted(
            cold_root.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        selected = paths[:limit]
        total = len(selected)
        self._report("replaying cold sessions", 0, total)
        processed = 0
        for i, path in enumerate(selected, 1):
            try:
                meta, segments = read_session_from_path(path)
                messages = [
                    TranscriptMessage(role=s.get("role", "user"), content=s.get("content", ""))
                    for s in segments
                    if s.get("content")
                ]
                if not messages:
                    continue
                pipeline.ingest(
                    messages,
                    session_id=meta.get("session_id", path.stem),
                    project_id=meta.get("project_id", project_id or "unknown"),
                )
                processed += 1
            except Exception as e:
                log.warning("cold reprocess failed for %s: %s", path, e)
            if i % 5 == 0 or i == total:
                self._report("replaying cold sessions", i, total)
        return processed

    def _resynthesize_facts(self, project_id: Optional[str]) -> int:
        """
        Ask a (potentially smarter) LLM to rewrite semantic facts for clarity.
        Only re-synthesizes low-confidence facts (< 0.70) to limit LLM calls.

        Rewriting improves *wording* and records new provenance, but it does NOT
        change a fragment's confidence. Confidence means "how sure are we this is
        true"; a rephrase adds no new evidence of truth, so letting it raise the
        score just inflated trust crank after crank without the memory actually
        getting more reliable. Confidence moves only for real reasons elsewhere
        (corroboration, contradiction, staleness decay).
        """
        if project_id:
            facts = self.db.list_fragments(
                project_id, category="fact", include_deprecated=False
            )
        else:
            rows = self.db.fetchall(
                "SELECT * FROM memory_fragments WHERE category='fact' AND is_deprecated=0"
            )
            from .schema import _row_to_fragment
            facts = [_row_to_fragment(r) for r in rows]

        candidates = [f for f in facts if f.confidence < 0.70]
        if not candidates:
            return 0

        # The model that will actually do the rewriting, and the provenance we
        # stamp on each rewritten fact so it moves off NULL (pre-provenance).
        if self._router is not None:
            producer = self._router.model_for(self._resynthesis_role)
        else:
            producer = _RESYNTHESIS_MODEL

        total = len(candidates)
        self._report("resynthesizing facts", 0, total)
        updated = 0
        for i, fact in enumerate(candidates, 1):
            prompt = (
                "Rewrite the following memory fragment as a clear, precise, "
                "self-contained factual statement. Remove vagueness. "
                "Keep it under 80 words.\n\n"
                f"Original: {fact.content}\n\n"
                'Respond with JSON: {"rewritten": "..."}'
            )
            try:
                if self._router is not None:
                    result = self._router.call_json(self._resynthesis_role, prompt, max_tokens=200)
                else:
                    result = call_model_json(
                        prompt, model=_RESYNTHESIS_MODEL, max_tokens=200
                    )
                rewritten = result.get("rewritten", "").strip()
                # Adopt a genuinely changed rewrite (clearer wording) and stamp the
                # producer that made it — but leave confidence ALONE. A rephrase is
                # not evidence of truth, so it must not raise how trustworthy the
                # memory looks.
                if rewritten and rewritten != fact.content:
                    self.db.execute(
                        """UPDATE memory_fragments
                           SET content=?,
                               producer_model=?, producer_version=?
                           WHERE id=?""",
                        (rewritten, producer, PRODUCER_VERSION, fact.id),
                    )
                    updated += 1
            except Exception as e:
                log.warning("resynthesis failed for %s: %s", fact.id, e)
            if i % 10 == 0 or i == total:
                self._report("resynthesizing facts", i, total)

        return updated
