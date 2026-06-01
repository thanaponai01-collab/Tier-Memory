"""
MemoryAuditor — Section 8 self-improvement loop.

Runs periodically (default weekly) to maintain memory health:
  1. Contradiction detection  — find conflicting fact pairs, deprecate weaker one
  2. Confidence decay         — halve confidence of old unvalidated facts
  3. PageRank centrality      — recompute graph_centrality from the triples graph
  4. Orphan pruning           — remove entities with no linked fragments
  5. Graph consistency        — validate triple endpoints still exist

Usage:
    auditor = MemoryAuditor(db, cfg.self_improvement, embedder)
    report = auditor.audit(project_id="my-app")   # returns AuditReport
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from .cold_storage import append_session
from .config import EvictionConfig, SelfImprovementConfig, StorageConfig
from .embedder import CachedEmbedder
from .ids import new_id
from .llm import call_model_json, LLMRouter
from .models import Correction, EpistemicEvent, MemoryFragment, PendingPattern, Simulation, StructuralPattern
from .schema import Database
from .scoring import composite_relevance_score
from .embedder import cosine_similarity as _cos_sim
from .v4_reranker import CRSWeightStore, learn_crs_weights, learn_global_weights

log = logging.getLogger("memory.auditor")

_PAGERANK_ITERATIONS = 20
_PAGERANK_DAMPING    = 0.85


# ── Public dataclass ─────────────────────────────────────────────────────────

class AuditReport:
    def __init__(self):
        self.contradictions_found: int = 0
        self.fragments_decayed: int = 0
        self.orphans_pruned: int = 0
        self.triples_removed: int = 0
        self.centrality_updated: int = 0
        self.evicted: int = 0
        # new passes
        self.patterns_articulated: int = 0
        self.simulations_run: int = 0
        self.simulations_resolved: int = 0
        self.simulations_expired: int = 0
        self.crystallized: int = 0
        self.meta_insights: int = 0
        self.llm_calls: int = 0
        self.reranker_updated: int = 0

    def __repr__(self) -> str:
        return (
            f"AuditReport(contradictions={self.contradictions_found}, "
            f"decayed={self.fragments_decayed}, orphans={self.orphans_pruned}, "
            f"triples_removed={self.triples_removed}, "
            f"centrality_updated={self.centrality_updated}, "
            f"evicted={self.evicted}, "
            f"patterns_articulated={self.patterns_articulated}, "
            f"simulations_run={self.simulations_run}, "
            f"simulations_expired={self.simulations_expired}, "
            f"crystallized={self.crystallized}, "
            f"meta_insights={self.meta_insights}, "
            f"reranker_updated={self.reranker_updated}, "
            f"llm_calls={self.llm_calls})"
        )


# ── Auditor ──────────────────────────────────────────────────────────────────

class MemoryAuditor:
    def __init__(
        self,
        db: Database,
        cfg: SelfImprovementConfig,
        embedder: Optional[CachedEmbedder] = None,
        eviction_cfg: Optional[EvictionConfig] = None,
        storage_cfg: Optional[StorageConfig] = None,
        weight_store: Optional[CRSWeightStore] = None,
        router: Optional[LLMRouter] = None,
    ):
        self.db = db
        self.cfg = cfg
        self.embedder = embedder
        self.eviction_cfg = eviction_cfg
        self.storage_cfg = storage_cfg
        self.weight_store = weight_store
        if router is None:
            from .config import LLMRolesConfig
            roles = LLMRolesConfig(
                cheap_model=cfg.contradiction_model,
                cheap_endpoint=cfg.contradiction_endpoint,
            )
            self._router = LLMRouter(roles)
        else:
            self._router = router

    def audit(self, project_id: Optional[str] = None) -> AuditReport:
        """
        Run all audit passes.  If project_id is None, audits every project.
        """
        report = AuditReport()
        started_at = datetime.now(tz=timezone.utc).isoformat()
        log.info("audit starting project=%s", project_id or "*")

        project_ids = (
            [project_id] if project_id
            else self._all_project_ids()
        )

        # Pass 10 (global) — refresh the fallback weight vector before per-project loop
        if self.weight_store is not None and self.db._conn is not None:
            try:
                _g = learn_global_weights(self.db._conn, self.weight_store)
                log.debug("learn_global_weights: %s", _g)
            except Exception as e:
                log.warning("learn_global_weights failed: %s", e)

        for pid in project_ids:
            if self.cfg.contradiction_detection:
                n_contra = self._detect_contradictions(pid)
                report.contradictions_found += n_contra
                report.llm_calls += n_contra  # 1 LLM call per cluster

            report.fragments_decayed  += self._velocity_weighted_decay(pid)
            report.centrality_updated += self._recalculate_centrality(pid)
            orphans, triples           = self._prune_orphaned_entities(pid)
            report.orphans_pruned     += orphans
            report.triples_removed    += triples
            self._validate_graph_consistency(pid)

            if self.cfg.pattern_articulation_enabled:
                n_art = self._articulate_patterns(pid)
                report.patterns_articulated += n_art
                report.llm_calls += n_art

            if self.cfg.rem_enabled and self.cfg.rem_cycles_per_audit > 0:
                n_sim, n_res, n_exp, lc = self._rem_cycle(pid)
                report.simulations_run     += n_sim
                report.simulations_resolved += n_res
                report.simulations_expired  += n_exp
                report.llm_calls           += lc

            if self.eviction_cfg and self.storage_cfg:
                evicted, cryst, lc = self._evict_to_cold(pid)
                report.evicted      += evicted
                report.crystallized += cryst
                report.llm_calls    += lc

            if self.cfg.meta_learn_enabled:
                n_insights = self._meta_learn(pid, report)
                report.meta_insights += n_insights
                report.llm_calls     += min(1, n_insights)  # 1 LLM call per project

            # Pass 10 — learn CRS weights from this project's retrieval outcomes
            if self.weight_store is not None and self.db._conn is not None:
                try:
                    _r = learn_crs_weights(self.db._conn, pid, self.weight_store)
                    if _r.get("status") == "updated":
                        report.reranker_updated += 1
                    log.debug("learn_crs_weights %s: %s", pid, _r)
                except Exception as e:
                    log.warning("learn_crs_weights failed for %s: %s", pid, e)

        ended_at = datetime.now(tz=timezone.utc).isoformat()
        try:
            self.db.log_daemon_run(
                started_at=started_at, ended_at=ended_at,
                llm_calls=report.llm_calls,
                mutations=report.fragments_decayed + report.evicted + report.crystallized,
                by_pass={
                    "contradictions": report.contradictions_found,
                    "decayed": report.fragments_decayed,
                    "patterns_articulated": report.patterns_articulated,
                    "simulations_run": report.simulations_run,
                    "crystallized": report.crystallized,
                    "meta_insights": report.meta_insights,
                    "reranker_updated": report.reranker_updated,
                },
            )
        except Exception:
            pass  # observability must not break audit

        log.info("audit complete: %s", report)
        return report

    # ── Pass 1: contradiction detection ──────────────────────────────────────

    def _detect_contradictions(self, project_id: str) -> int:
        """
        Cluster facts by embedding similarity, then ask LLM to identify
        contradictions within each cluster.  The lower-confidence fragment
        is deprecated.
        """
        facts = self.db.list_fragments(
            project_id, category="fact", include_deprecated=False
        )
        if len(facts) < 2:
            return 0

        # Load embeddings
        if self.embedder:
            for f in facts:
                if not f.embedding:
                    f.embedding = self.embedder.embed(f.content)

        # Build similarity clusters (O(n²) — acceptable for typical fact counts)
        clusters: list[list] = []
        assigned = set()

        for i, fa in enumerate(facts):
            if i in assigned:
                continue
            cluster = [fa]
            assigned.add(i)
            if fa.embedding:
                for j, fb in enumerate(facts):
                    if j <= i or j in assigned:
                        continue
                    if fb.embedding and _cosine(fa.embedding, fb.embedding) > 0.70:
                        cluster.append(fb)
                        assigned.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        contradictions = 0
        for cluster in clusters:
            contradictions += self._resolve_cluster(cluster, project_id)
        return contradictions

    def _resolve_cluster(self, cluster: list, project_id: str) -> int:
        contents = [{"idx": i, "content": f.content} for i, f in enumerate(cluster)]
        prompt = (
            "Given these memory fragments, identify any pairs that directly "
            "contradict each other. A contradiction means the two statements "
            "cannot both be true at the same time.\n\n"
            f"Fragments:\n{json.dumps(contents, indent=2)}\n\n"
            "Respond with JSON:\n"
            '{"contradictions": [{"keep_idx": 0, "deprecate_idx": 1, "reason": "..."}]}'
        )
        try:
            result = self._router.call_json("cheap", prompt, max_tokens=512)
        except Exception as e:
            log.warning("contradiction LLM call failed: %s", e)
            return 0

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        found = 0
        for pair in result.get("contradictions", []):
            keep_idx = pair.get("keep_idx")
            dep_idx  = pair.get("deprecate_idx")
            reason   = pair.get("reason", "")
            if keep_idx is None or dep_idx is None:
                continue
            if dep_idx >= len(cluster) or keep_idx >= len(cluster):
                continue
            victim = cluster[dep_idx]
            winner = cluster[keep_idx]
            if victim.confidence <= winner.confidence:
                self.db.mark_deprecated(victim.id, winner.id)
                corr = Correction(
                    id=new_id(),
                    project_id=project_id,
                    original_fact=victim.content,
                    corrected_fact=winner.content,
                    applied_to=json.dumps([victim.id]),
                )
                self.db.insert_correction(corr)
                # §3.4 — emit epistemic events for both winner and loser
                self.db.insert_epistemic_event(EpistemicEvent(
                    fragment_id=victim.id,
                    event_type="contradiction",
                    delta_conf=-(victim.confidence - max(0.05, victim.confidence * 0.5)),
                    evidence_frag=winner.id,
                    ts=now_iso,
                ))
                self.db.insert_epistemic_event(EpistemicEvent(
                    fragment_id=winner.id,
                    event_type="contradiction",
                    delta_conf=0.0,  # winner survives; velocity still increments
                    evidence_frag=victim.id,
                    ts=now_iso,
                ))
                # update contradiction_count on fragments
                self.db.execute(
                    "UPDATE memory_fragments SET contradiction_count = contradiction_count + 1 WHERE id = ?",
                    (victim.id,),
                )
                log.debug(
                    "deprecated contradiction %s (kept %s): %s",
                    victim.id, winner.id, reason,
                )
                found += 1
        return found

    # ── Pass 2: velocity-weighted confidence decay (§3.4, Gap #1) ───────────

    def _velocity_weighted_decay(self, project_id: str) -> int:
        """
        Replaces uniform decay.  Each fragment's mutation_velocity (EWMA of
        contradiction events) amplifies its decay rate:
            c ← c · exp(-(λ₀ + κ·v) · Δt_days)
        Fast-mutating facts lose confidence proportionally to their volatility.
        """
        now = datetime.now(tz=timezone.utc)
        since_iso = (now - timedelta(days=self.cfg.confidence_decay_after_days)).isoformat()
        alpha = self.cfg.velocity_ewma_alpha
        lam0  = self.cfg.velocity_decay_lambda0
        kappa = self.cfg.velocity_decay_kappa

        facts = self.db.list_fragments(
            project_id, category="fact", include_deprecated=False
        )
        decayed = 0
        for f in facts:
            # ── Update mutation_velocity EWMA ──────────────────────────
            contradiction_hit = float(
                self.db.count_contradiction_events(f.id, since_iso) > 0
            )
            new_velocity = (1.0 - alpha) * f.mutation_velocity + alpha * contradiction_hit
            new_velocity = round(new_velocity, 6)

            # ── Compute decay ──────────────────────────────────────────
            try:
                last = datetime.fromisoformat(f.last_accessed)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                delta_days = (now - last).total_seconds() / 86400.0
            except (ValueError, TypeError):
                delta_days = 0.0

            if delta_days <= 0:
                if new_velocity != f.mutation_velocity:
                    self.db.execute(
                        "UPDATE memory_fragments SET mutation_velocity=? WHERE id=?",
                        (new_velocity, f.id),
                    )
                continue

            decay_rate = lam0 + kappa * new_velocity
            new_conf = max(0.05, f.confidence * math.exp(-decay_rate * delta_days))
            self.db.execute(
                "UPDATE memory_fragments SET confidence=?, mutation_velocity=? WHERE id=?",
                (round(new_conf, 4), new_velocity, f.id),
            )
            decayed += 1
        return decayed

    # ── Pass 3: PageRank centrality ───────────────────────────────────────────

    def _recalculate_centrality(self, project_id: str) -> int:
        """
        Compute a simple PageRank over the entity graph, then propagate
        per-entity centrality to linked memory fragments.
        """
        rows = self.db.fetchall(
            "SELECT subject_id, object_id FROM triples WHERE project_id=? AND confidence>0.5",
            (project_id,),
        )
        if not rows:
            return 0

        # Build adjacency
        out_edges: dict[str, list[str]] = defaultdict(list)
        nodes: set[str] = set()
        for r in rows:
            s, o = r["subject_id"], r["object_id"]
            out_edges[s].append(o)
            nodes.add(s); nodes.add(o)

        n = len(nodes)
        if n == 0:
            return 0

        node_list = list(nodes)
        idx = {nid: i for i, nid in enumerate(node_list)}
        scores = {nid: 1.0 / n for nid in node_list}

        for _ in range(_PAGERANK_ITERATIONS):
            new_scores: dict[str, float] = {nid: (1.0 - _PAGERANK_DAMPING) / n for nid in node_list}
            for src, dests in out_edges.items():
                share = _PAGERANK_DAMPING * scores[src] / len(dests)
                for dst in dests:
                    new_scores[dst] = new_scores.get(dst, 0.0) + share
            scores = new_scores

        # Normalize to [0,1]
        max_score = max(scores.values()) or 1.0
        scores = {k: v / max_score for k, v in scores.items()}

        # Push centrality back to fragment rows via linked triples
        updated = 0
        for entity_id, centrality in scores.items():
            fragment_ids = self.db.fragments_linked_to_entity(entity_id)
            for fid in fragment_ids:
                self.db.execute(
                    "UPDATE memory_fragments SET graph_centrality=? WHERE id=?",
                    (round(centrality, 4), fid),
                )
                updated += 1

        return updated

    # ── Pass 4: orphan pruning ────────────────────────────────────────────────

    def _prune_orphaned_entities(self, project_id: str) -> tuple[int, int]:
        """
        Remove entities that have no linked fragments and appear in no triples.
        Also removes dangling triples that reference non-existent fragments.
        Returns (entities_removed, triples_removed).
        """
        # Remove triples whose source_fragment no longer exists
        dangling = self.db.fetchall("""
            SELECT t.id FROM triples t
            LEFT JOIN memory_fragments f ON t.source_fragment = f.id
            WHERE t.project_id = ?
              AND t.source_fragment IS NOT NULL
              AND f.id IS NULL
        """, (project_id,))
        for r in dangling:
            self.db.execute("DELETE FROM triples WHERE id=?", (r["id"],))

        # Remove entities with no triples and no linked fragments
        orphaned = self.db.fetchall("""
            SELECT e.id FROM entities e
            LEFT JOIN triples ts ON ts.subject_id = e.id OR ts.object_id = e.id
            WHERE e.project_id = ?
              AND ts.id IS NULL
        """, (project_id,))
        for r in orphaned:
            self.db.execute("DELETE FROM entities WHERE id=?", (r["id"],))

        return len(orphaned), len(dangling)

    # ── Pass 5: graph consistency ─────────────────────────────────────────────

    def _validate_graph_consistency(self, project_id: str) -> None:
        """
        Remove triples whose subject or object entity no longer exists.
        """
        broken = self.db.fetchall("""
            SELECT t.id FROM triples t
            LEFT JOIN entities es ON t.subject_id = es.id
            LEFT JOIN entities eo ON t.object_id  = eo.id
            WHERE t.project_id = ?
              AND (es.id IS NULL OR eo.id IS NULL)
        """, (project_id,))
        for r in broken:
            self.db.execute("DELETE FROM triples WHERE id=?", (r["id"],))
        if broken:
            log.debug("removed %d broken triples in project=%s", len(broken), project_id)

    # ── Pass 7: pattern articulation (§3.1, Gap #2) ──────────────────────────

    def _articulate_patterns(self, project_id: str) -> int:
        """
        For pending patterns that have reached quorum (3+ exemplars, 2+ projects),
        call the LLM to articulate the abstract form and promote to structural_patterns.
        """
        pending = self.db.get_pending_patterns_at_quorum(
            min_occurrences=self.cfg.pattern_quorum_occurrences,
            min_projects=self.cfg.pattern_quorum_projects,
        )
        articulated = 0
        for p in pending:
            exemplars = json.loads(p.exemplars_json)
            prompt = _PROMPT_PATTERN_ARTICULATOR.format(
                signature_hash=p.signature_hash,
                arity=p.arity,
                exemplars=json.dumps(exemplars[:5], indent=2),
            )
            try:
                result = self._router.call_json("cheap", prompt, max_tokens=512)
            except Exception as e:
                log.warning("pattern articulation LLM failed: %s", e)
                continue

            abstract_form = result.get("abstract_form") or ""
            if not abstract_form:
                # LLM decided no coherent pattern — delete pending
                self.db.delete_pending_pattern(p.id)
                continue

            now_iso = datetime.now(tz=timezone.utc).isoformat()
            pattern = StructuralPattern(
                id=new_id(),
                signature_hash=p.signature_hash,
                arity=p.arity,
                abstract_form=abstract_form,
                failure_mode=result.get("failure_mode"),
                exemplars_json=p.exemplars_json,
                embedding_dim=0,
                occurrence_count=len(exemplars),
                created_at=now_iso,
                last_seen=now_iso,
            )
            self.db.upsert_structural_pattern(pattern)
            self.db.delete_pending_pattern(p.id)
            articulated += 1
            log.debug("articulated pattern %s: %s", pattern.signature_hash[:8], abstract_form[:60])

        return articulated

    # ── Pass 8: REM cycle (§3.2, Gap #3) ─────────────────────────────────────

    def _rem_cycle(self, project_id: str) -> tuple[int, int, int, int]:
        """
        Generates counterfactual simulations from high-centrality entities.
        Returns (simulations_created, simulations_resolved_by_expiry, expired, llm_calls).
        Simulation resolution against live fragments happens in pipeline.py Stage 4.
        """
        now = datetime.now(tz=timezone.utc)
        now_iso = now.isoformat()

        # Expire old unresolved simulations
        cutoff = (now - timedelta(days=self.cfg.rem_sim_expiry_days)).isoformat()
        expired = self.db.expire_old_simulations(cutoff)

        high_cent = self.db.get_high_centrality_entities(
            project_id, limit=self.cfg.rem_seed_count
        )
        sim_created = 0
        llm_calls = 0

        for ent_row in high_cent:
            # Build neighborhood view
            triples = self.db.get_triples_for_entity(ent_row["id"])
            ctx_triples = [
                f"{r['subj_name']} --[{r['predicate']}]--> {r['obj_name']}"
                for r in triples
            ]
            prompt = _PROMPT_PERTURBATION.format(
                entity_type=ent_row["entity_type"],
                entity_name=ent_row["qualified_name"],
                neighborhood="\n".join(ctx_triples[:20]),
                n_perturbations=self.cfg.rem_perturbations_per_seed,
            )
            try:
                result = self._router.call_json("cheap", prompt, max_tokens=1024)
                llm_calls += 1
            except Exception as e:
                log.warning("REM LLM call failed: %s", e)
                continue

            for p in result.get("perturbations", [])[:self.cfg.rem_perturbations_per_seed]:
                prior = float(p.get("prior", 0.0))
                if prior < self.cfg.rem_min_prior:
                    continue
                predicted_facts = p.get("predicted_facts", [])
                if not predicted_facts:
                    continue

                sim = Simulation(
                    id=new_id(),
                    project_id=project_id,
                    seed_entity_id=ent_row["id"],
                    perturbation=str(p.get("name", "")),
                    hypothesis=str(p.get("mechanism", "")),
                    predicted_facts_json=json.dumps([{"text": t} for t in predicted_facts]),
                    prior=prior,
                    created_at=now_iso,
                )
                try:
                    self.db.insert_simulation(sim)
                    sim_created += 1
                except Exception as e:
                    log.warning("simulation insert failed: %s", e)

        return sim_created, 0, expired, llm_calls

    # ── Pass 6: warm → cold eviction with crystallization (§3.3) ──────────────

    def _evict_to_cold(self, project_id: str) -> tuple[int, int, int]:
        """
        Moves fragments below warm_threshold to cold storage.
        Before evicting a cluster, asks LLM to crystallize a durable principle.
        Returns (evicted_count, crystallized_count, llm_calls).
        """
        if not self.storage_cfg or not self.eviction_cfg:
            return 0, 0, 0

        fragments = self.db.list_fragments(project_id, include_deprecated=False)
        cold_root = Path(self.storage_cfg.cold_storage_path)
        threshold = self.eviction_cfg.warm_threshold
        llm_calls = 0
        crystallized = 0

        # Collect candidates
        candidates: list[MemoryFragment] = []
        for frag in fragments:
            if frag.is_pinned:
                continue
            if composite_relevance_score(frag) < threshold:
                candidates.append(frag)

        if not candidates:
            return 0, 0, 0

        # Load embeddings for clustering
        if self.embedder:
            for f in candidates:
                if not f.embedding:
                    try:
                        f.embedding = self.embedder.embed(f.content)
                    except Exception:
                        pass

        # Cluster candidates by embedding similarity (greedy, same threshold as contradiction)
        clusters = _cluster_by_similarity(candidates, threshold=0.70)
        evicted_total = 0

        for cluster in clusters:
            # Attempt crystallization on clusters large enough
            cryst_frag_id: Optional[str] = None
            if (self.cfg.crystallization_enabled
                    and len(cluster) >= self.cfg.crystallization_min_cluster):
                cryst_frag_id = self._crystallize_cluster(cluster, project_id)
                if cryst_frag_id:
                    crystallized += 1
                    llm_calls += 1

            # Evict each fragment in the cluster
            for frag in cluster:
                try:
                    append_session(
                        cold_root,
                        project_id=frag.project_id,
                        session_id=f"evicted_{frag.id}",
                        segments=[{
                            "role": "memory",
                            "content": frag.content,
                            "timestamp": frag.last_accessed,
                            "fragment_id": frag.id,
                        }],
                    )
                    self.db.mark_deprecated(frag.id, deprecated_by=cryst_frag_id)
                    evicted_total += 1
                except Exception as e:
                    log.warning("eviction failed for %s: %s", frag.id, e)

        return evicted_total, crystallized, llm_calls

    def _crystallize_cluster(
        self, cluster: list[MemoryFragment], project_id: str
    ) -> Optional[str]:
        """
        Calls the LLM to distill a cluster of low-CRS fragments into a durable principle.
        Writes the principle as a new category=fact fragment; returns its ID.
        """
        frags_payload = [
            {"id": f.id, "content": f.content,
             "category": f.category, "confidence": round(f.confidence, 3)}
            for f in cluster
        ]
        prompt = _PROMPT_CRYSTALLIZATION.format(
            fragments=json.dumps(frags_payload, indent=2)
        )
        try:
            result = self._router.call_json("cheap", prompt, max_tokens=256)
        except Exception as e:
            log.warning("crystallization LLM failed: %s", e)
            return None

        principle = result.get("principle")
        abstraction_lvl = float(result.get("abstraction_lvl", 0.0))
        if not principle or abstraction_lvl < 0.8:
            return None

        avg_conf = sum(f.confidence for f in cluster) / len(cluster)
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        from .ids import new_id as _new_id
        frag_id = _new_id()
        cryst_frag = MemoryFragment(
            id=frag_id,
            project_id=project_id,
            scope="project",
            category="fact",
            content=principle,
            token_count=max(1, len(principle) // 4),
            abstraction_lvl=abstraction_lvl,
            confidence=round(avg_conf * 0.9, 4),
            source_type="crystallization",
            created_at=now_iso,
            last_accessed=now_iso,
            epistemic_class="consolidated",
        )
        try:
            self.db.upsert_fragment(cryst_frag)
            log.debug("crystallized principle for %d fragments: %s", len(cluster), principle[:60])
            return frag_id
        except Exception as e:
            log.warning("crystallized fragment write failed: %s", e)
            return None

    # ── Pass 9: meta-learning (§3.7) ─────────────────────────────────────────

    def _meta_learn(self, project_id: str, report: AuditReport) -> int:
        """
        Synthesize patterns from what went wrong into durable meta-knowledge.
        Reads recent corrections and high-contradiction fragments, asks LLM to
        identify recurring error patterns, and writes them back as
        epistemic_class="reflected" facts so future retrievals surface the caution.
        """
        lookback_days = getattr(self.cfg, "meta_learn_lookback_days", 30)
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        ).isoformat()

        corrections = self.db.fetchall(
            """
            SELECT original_fact, corrected_fact
            FROM corrections
            WHERE (project_id = ? OR project_id IS NULL) AND created_at >= ?
            ORDER BY created_at DESC LIMIT 20
            """,
            (project_id, cutoff),
        )

        volatile = self.db.fetchall(
            """
            SELECT content, contradiction_count, mutation_velocity
            FROM memory_fragments
            WHERE project_id = ?
              AND is_deprecated = 0
              AND contradiction_count > 0
            ORDER BY contradiction_count DESC LIMIT 15
            """,
            (project_id,),
        )

        if not corrections and not volatile:
            return 0

        payload = {
            "recent_corrections": [
                {"was": r["original_fact"][:200], "now": r["corrected_fact"][:200]}
                for r in corrections
            ],
            "volatile_fragments": [
                {
                    "content": r["content"][:200],
                    "contradictions": r["contradiction_count"],
                    "velocity": round(r["mutation_velocity"], 3),
                }
                for r in volatile
            ],
        }

        prompt = _PROMPT_META_LEARN.format(payload=json.dumps(payload, indent=2))
        try:
            result = self._router.call_json("cheap", prompt, max_tokens=512)
        except Exception as e:
            log.warning("meta_learn LLM call failed: %s", e)
            return 0

        insights = result.get("insights", [])
        if not insights:
            return 0

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        written = 0
        for item in insights[:5]:
            text = str(item.get("insight", "")).strip()
            confidence = float(item.get("confidence", 0.7))
            if not text or confidence < 0.5:
                continue

            content = f"[Meta-insight] {text}"
            frag = MemoryFragment(
                id=new_id(),
                project_id=project_id,
                scope="project",
                category="fact",
                content=content,
                token_count=max(1, len(content) // 4),
                confidence=confidence,
                abstraction_lvl=0.8,
                source_type="meta_learning",
                created_at=now_iso,
                last_accessed=now_iso,
                epistemic_class="reflected",
            )
            if self.embedder:
                try:
                    frag.embedding = self.embedder.embed(content)
                    frag.embedding_dim = len(frag.embedding)
                    frag.embedding_model = self.embedder.model_name
                except Exception:
                    pass
            try:
                self.db.upsert_fragment(frag)
                written += 1
                log.debug("meta-insight written: %s", text[:80])
            except Exception as e:
                log.warning("meta_learn fragment write failed: %s", e)

        return written

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _all_project_ids(self) -> list[str]:
        rows = self.db.fetchall(
            "SELECT DISTINCT project_id FROM memory_fragments WHERE is_deprecated=0"
        )
        return [r["project_id"] for r in rows]


# ── Math helpers ─────────────────────────────────────────────────────────────

# Single canonical implementation lives in embedder.cosine_similarity (§6.6)
_cosine = _cos_sim


def _cluster_by_similarity(
    fragments: list[MemoryFragment], threshold: float = 0.70
) -> list[list[MemoryFragment]]:
    """Greedy O(n²) similarity clustering for eviction batches."""
    clusters: list[list[MemoryFragment]] = []
    assigned = set()
    for i, fa in enumerate(fragments):
        if i in assigned:
            continue
        cluster = [fa]
        assigned.add(i)
        if fa.embedding:
            for j, fb in enumerate(fragments):
                if j <= i or j in assigned:
                    continue
                if fb.embedding and _cosine(fa.embedding, fb.embedding) >= threshold:
                    cluster.append(fb)
                    assigned.add(j)
        clusters.append(cluster)
    return clusters


# ── Cognitive orchestration prompts (§4) ─────────────────────────────────────

_PROMPT_PERTURBATION = """\
You generate adversarial-but-plausible failure scenarios for a code/system entity.

INPUT
- entity: {entity_type} "{entity_name}"
- neighborhood: 2-hop subgraph triples
{neighborhood}

GENERATE {n_perturbations} perturbations. Each must be:
- Concrete and named (a version change, a deprecation, a behavior shift), not vague.
- Anchored in observable signal from neighborhood/facts, or marked prior ≤ 0.15.
- Distinct in mechanism — do not enumerate variants of the same scenario.

OUTPUT JSON
{{"perturbations": [
    {{"name": "str",
      "mechanism": "str",
      "prior": 0.0,
      "predicted_facts": ["str"],
      "evidence_signals": ["str"]
    }}
]}}

If the entity has no plausible adversarial future, return {{"perturbations": []}}.
Do not fabricate. Empty is the correct answer when you have no signal.\
"""

_PROMPT_PATTERN_ARTICULATOR = """\
You are shown ≥3 subgraphs from DIFFERENT projects that share the same structural
signature. Articulate the abstract pattern they instantiate.

INPUT
- signature_hash: {signature_hash}
- arity: {arity}
- exemplars:
{exemplars}

RULES
- No proper nouns. Replace concrete names with role tokens drawn from:
  Producer, Consumer, Gate, Conduit, Sink, Store, Validator, Coordinator.
- One sentence: the pattern.
- One sentence: the failure mode this pattern exhibits in practice.
- One sentence: the confirmation signal.

OUTPUT JSON
{{"abstract_form": "str|null",
  "failure_mode": "str|null",
  "confirmation_signal": "str|null",
  "role_glossary": {{}}}}

If the exemplars do NOT share a coherent pattern, return all fields null with
"reason": "...". Do not fabricate a pattern from coincidence.\
"""

_PROMPT_META_LEARN = """\
You are analyzing a memory system's own error patterns to extract meta-knowledge
about where its knowledge is unreliable.

AUDIT DATA
{payload}

INSTRUCTIONS
- Identify recurring patterns: what topics or knowledge types keep being wrong
  or unstable (high contradiction count, high mutation velocity)?
- Express each pattern as a single durable statement a future retrieval could act on.
- Keep it abstract — no proper nouns where avoidable. The insight must transfer.
- Confidence: how certain are you this is a real pattern vs. noise?
- Only write insights with clear signal (2+ supporting data points). Silence is correct for noise.

OUTPUT JSON
{{"insights": [
    {{"insight": "concise statement of what tends to go wrong and why",
      "confidence": 0.0
    }}
]}}

Return {{"insights": []}} when there is insufficient signal.\
"""

_PROMPT_CRYSTALLIZATION = """\
You receive a cluster of K low-CRS fragments about to evict to cold storage.
Decide if a durable principle exists across them.

INPUT
{fragments}

PROCEDURE
1. Find the load-bearing claim that survives stripping timestamps, identifiers,
   file paths, exact syntax, and operator names.
2. If no such claim exists: return null. Compression is opt-in. Silence is
   correct for noise.
3. If it exists: phrase as a single declarative engineering principle, ≤30 words,
   no hedging.

OUTPUT JSON
{{"principle": "str|null",
  "abstraction_lvl": 0.0,
  "supporting_fragment_ids": [],
  "predicted_contradiction_count": 0
}}\
"""
