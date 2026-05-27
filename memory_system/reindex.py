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
from typing import Optional

from .cold_storage import read_session_from_path
from .config import SelfImprovementConfig
from .embedder import CachedEmbedder, OllamaEmbedder
from .ids import new_id
from .llm import call_model_json
from .models import MemoryFragment
from .pipeline import ConsolidationPipeline, TranscriptMessage
from .schema import Database
from .vector_index import VectorIndex

log = logging.getLogger("memory.reindex")

_RESYNTHESIS_MODEL = "claude-haiku-4-5-20251001"
_BATCH_SIZE = 50


@dataclass
class ReindexReport:
    fragments_reindexed: int = 0
    facts_resynthesized: int = 0
    cold_sessions_reprocessed: int = 0
    errors: int = 0

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
    ):
        self.db = db
        self.idx = vector_index
        self.embedder = new_embedder
        self.cfg = cfg
        self.resynthesize_facts = resynthesize_facts
        self.reprocess_cold = reprocess_cold
        self._pipeline = pipeline
        self.cold_storage_path = cold_storage_path

    def execute(self, project_id: Optional[str] = None) -> ReindexReport:
        """
        Re-embed all non-deprecated fragments.  Pass project_id=None to
        reindex the entire database (e.g., after a global model upgrade).
        """
        report = ReindexReport()
        log.info("reindex starting project=%s", project_id or "*")

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

        log.info("reindex complete: %s", report)
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

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
        processed = 0
        for path in paths[:limit]:
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
        return processed

    def _resynthesize_facts(self, project_id: Optional[str]) -> int:
        """
        Ask a (potentially smarter) LLM to rewrite semantic facts for clarity.
        Only re-synthesizes low-confidence facts (< 0.70) to limit LLM calls.
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

        updated = 0
        for fact in candidates:
            prompt = (
                "Rewrite the following memory fragment as a clear, precise, "
                "self-contained factual statement. Remove vagueness. "
                "Keep it under 80 words.\n\n"
                f"Original: {fact.content}\n\n"
                'Respond with JSON: {"rewritten": "..."}'
            )
            try:
                result = call_model_json(
                    prompt, model=_RESYNTHESIS_MODEL, max_tokens=200
                )
                rewritten = result.get("rewritten", "").strip()
                if rewritten and rewritten != fact.content:
                    self.db.execute(
                        "UPDATE memory_fragments SET content=?, confidence=? WHERE id=?",
                        (rewritten, min(fact.confidence + 0.10, 1.0), fact.id),
                    )
                    updated += 1
            except Exception as e:
                log.warning("resynthesis failed for %s: %s", fact.id, e)

        return updated
