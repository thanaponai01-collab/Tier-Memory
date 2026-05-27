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
from .llm import call_model_json
from .models import Correction, EpistemicEvent, MemoryFragment, PendingPattern, Simulation, StructuralPattern
from .schema import Database
from .scoring import composite_relevance_score, _cosine_similarity as _cos_sim

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
        self.llm_calls: int = 0

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
    ):
        self.db = db
        self.cfg = cfg
        self.embedder = embedder
        self.eviction_cfg = eviction_cfg
        self.storage_cfg = storage_cfg

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
            result = call_model_json(
                prompt,
                model=self.cfg.contradiction_model,
                endpoint=getattr(self.cfg, "contradiction_endpoint", None),
                max_tokens=512,
            )
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

    # ── Pass 6: warm → cold eviction ─────────────────────────────────────────

    def _evict_to_cold(self, project_id: str) -> int:
        """
        Move fragments whose CRS < warm_threshold to cold storage.
        Marks them is_deprecated=True in SQLite so they are excluded from retrieval.
        Does NOT delete — cold storage retains provenance.
        Returns count of evicted fragments.
        """
        fragments = self.db.list_fragments(project_id, include_deprecated=False)
        cold_root = Path(self.storage_cfg.cold_storage_path)
        threshold = self.eviction_cfg.warm_threshold
        evicted = 0
        for frag in fragments:
            if frag.is_pinned:
                continue
            crs = composite_relevance_score(frag)
            if crs >= threshold:
                continue
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
                self.db.mark_deprecated(frag.id, deprecated_by=None)
                evicted += 1
            except Exception as e:
                log.warning("eviction failed for fragment %s: %s", frag.id, e)
        return evicted

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
                result = call_model_json(
                    prompt,
                    model=self.cfg.pattern_articulation_model,
                    endpoint=getattr(self.cfg, "pattern_articulation_endpoint", None),
                    max_tokens=512,
                )
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
                result = call_model_json(
                    prompt,
                    model=self.cfg.rem_model,
                    endpoint=getattr(self.cfg, "rem_endpoint", None),
                    max_tokens=1024,
                )
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

    # ── Pass 6 (rewritten): crystallized cold eviction (§3.3) ────────────────

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
            result = call_model_json(
                prompt,
                model=self.cfg.crystallization_model,
                endpoint=getattr(self.cfg, "crystallization_endpoint", None),
                max_tokens=256,
            )
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _all_project_ids(self) -> list[str]:
        rows = self.db.fetchall(
            "SELECT DISTINCT project_id FROM memory_fragments WHERE is_deprecated=0"
        )
        return [r["project_id"] for r in rows]


# ── Math helpers ─────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


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
