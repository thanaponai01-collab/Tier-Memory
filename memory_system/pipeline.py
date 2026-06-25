"""
4-stage compression pipeline — Section 2.3 of the architecture.

Stage 1: Segmentation      — split transcript into semantic episodes
Stage 2: Distillation      — per-episode structured summary (cheap model)
Stage 3: Entity extraction — named entities + graph triples
Stage 4: Consolidation     — dedup, contradiction check, promote to semantic fact

Also implements:
- Attention-weighted memory formation (Section 7.2.3)
- Episodic → semantic consolidation trigger (Section 7.2.1)
- End-of-session reflection (Section 7.2.2)
"""

from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import hashlib

log = logging.getLogger("memory.pipeline")

from .config import CompressionConfig
from .embedder import Embedder, RandomEmbedder
from .ids import new_id
from .llm import call_model_json, call_model, LLMRouter, PRODUCER_VERSION
from .models import Entity, MemoryFragment, PendingPattern, Simulation, Triple
from .schema import Database
from .scoring import _cosine_similarity
from .vector_index import VectorIndex


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class TranscriptMessage:
    role: str                        # 'user' | 'assistant' | 'tool'
    content: str
    timestamp: Optional[str] = None  # ISO 8601; defaults to ingest time

    # Attention-weight signals (Section 7.2.3)
    is_correction: bool = False
    is_error_fix: bool = False
    tool_call_chain_len: int = 0
    is_explicit_remember: bool = False
    is_repeated_topic: bool = False


@dataclass
class Episode:
    messages: list[TranscriptMessage]
    start_idx: int
    end_idx: int
    embedding: Optional[list[float]] = field(default=None, repr=False)
    attention_weight: float = 1.0


@dataclass
class DistilledEpisode:
    episode: Episode
    fragment: MemoryFragment
    entities_raw: list[dict]
    triples_raw: list[dict]


# ── Attention weights (Section 7.2.3) ────────────────────────────────────────

_ATTENTION_WEIGHTS = {
    "is_correction":       3.0,
    "is_error_fix":        2.5,
    "is_explicit_remember": 5.0,
    "is_repeated_topic":   2.0,
    "small_talk":          0.1,    # applied when none of the above are set
}

_DEFAULT_WEIGHT = 1.0


def _attention_weight(msg: TranscriptMessage) -> float:
    if msg.is_explicit_remember:
        return _ATTENTION_WEIGHTS["is_explicit_remember"]
    if msg.is_correction:
        return _ATTENTION_WEIGHTS["is_correction"]
    if msg.is_error_fix:
        return _ATTENTION_WEIGHTS["is_error_fix"]
    if msg.tool_call_chain_len >= 3:
        return 2.0
    if msg.is_repeated_topic:
        return _ATTENTION_WEIGHTS["is_repeated_topic"]
    return _DEFAULT_WEIGHT


# ── Pipeline ─────────────────────────────────────────────────────────────────

class ConsolidationPipeline:
    """
    Ingests a raw session transcript and writes compressed memory fragments
    to the database and vector index.

    Usage:
        pipeline = ConsolidationPipeline(db, vector_index, cfg, embedder)
        pipeline.ingest(messages, session_id="sess_xyz", project_id="my-app")
    """

    def __init__(
        self,
        db: Database,
        vector_index: VectorIndex,
        cfg: CompressionConfig,
        embedder: Optional[Embedder] = None,
        dry_run: bool = False,
        router: Optional[LLMRouter] = None,
        distill_role: str = "cheap",
    ):
        self.db = db
        self.vector_index = vector_index
        self.cfg = cfg
        self.embedder: Embedder = embedder or RandomEmbedder(dim=vector_index.dim)
        self.dry_run = dry_run  # if True, compute but don't write to DB
        # Which router role distillation runs through. Live ingest uses "cheap"
        # (local qwen3:8b) because it fires on every session and must stay free.
        # Cold-replay (the flywheel's re-judgment pass) builds a SEPARATE pipeline
        # with "strong" so a smarter model re-distills the archive — the weak
        # local model only reshuffles words and floods the store with vague,
        # low-confidence episodes (verified: a full crank regressed it 947→3079).
        self._distill_role = distill_role
        if router is None:
            from .config import LLMRolesConfig
            roles = LLMRolesConfig(
                cheap_model=cfg.distillation_model,
                cheap_endpoint=cfg.distillation_endpoint,
            )
            self._router = LLMRouter(roles)
        else:
            self._router = router
        # §5.6 producer provenance — stamp every fragment this pipeline writes
        # with the model + logic generation that produced it, so a future
        # (smarter) model can tell whose judgment it is safe to overrule.
        self._producer_model = self._router.model_for(distill_role)
        self._producer_version = PRODUCER_VERSION

    # ── Public entrypoint ────────────────────────────────────────────────────

    def ingest(
        self,
        messages: list[TranscriptMessage],
        session_id: str,
        project_id: str,
        stats_out: Optional[dict] = None,
    ) -> list[MemoryFragment]:
        """
        Run all 4 stages. Returns the new MemoryFragment objects that were
        written (or would have been written in dry_run mode).

        If `stats_out` is provided, it is populated with {"episodes": N,
        "distill_failures": M} — the signal the redrive worker uses to tell a
        genuine outage (distillation threw for every episode → 0 fragments) apart
        from a session that simply had nothing worth keeping.
        """
        if not messages:
            if stats_out is not None:
                stats_out["episodes"] = 0
                stats_out["distill_failures"] = 0
            return []

        # Stage 1: segment into episodes
        episodes = self._stage1_segment(messages)

        # Stage 2: distil each episode into a structured summary
        distilled, distill_failures = self._stage2_distill(episodes, session_id, project_id)

        # Stage 3: extract entities and graph triples
        self._stage3_extract_entities(distilled, project_id)

        # Stage 3b: structural pattern fingerprinting (§3.1)
        self._stage3b_structural(distilled, project_id)

        # Stage 4: consolidate new fragments against existing memory
        new_fragments = [d.fragment for d in distilled]
        self._stage4_consolidate(new_fragments, project_id)

        if stats_out is not None:
            stats_out["episodes"] = len(episodes)
            stats_out["distill_failures"] = distill_failures

        return new_fragments

    def reflect(
        self,
        messages: list[TranscriptMessage],
        session_id: str,
        project_id: str,
    ) -> Optional[MemoryFragment]:
        """
        End-of-session reflection (Section 7.2.2).
        Asks the cheap model what it learned. Returns a high-confidence
        fragment, or None if the reflection produced nothing useful.
        """
        if not messages:
            return None

        transcript_text = _format_transcript(messages, max_chars=6000)
        prompt = f"""You just completed a work session. Here is the transcript:

{transcript_text}

Reflect on this session and extract reusable knowledge.

Output JSON only:
{{
  "new_facts": ["short statement of a new fact learned", ...],
  "corrected_assumptions": [{{"was": "old belief", "now": "corrected belief"}}, ...],
  "lessons": ["what to do differently next time", ...],
  "proactive_suggestions": ["something to remember proactively", ...]
}}

If there is nothing meaningful to extract, return {{"new_facts":[],"corrected_assumptions":[],"lessons":[],"proactive_suggestions":[]}}"""

        try:
            data = self._router.call_json("cheap", prompt)
        except Exception:
            return None

        all_items: list[str] = (
            data.get("new_facts", []) +
            [f"Corrected: was '{c['was']}', now '{c['now']}'"
             for c in data.get("corrected_assumptions", []) if isinstance(c, dict)] +
            data.get("lessons", []) +
            data.get("proactive_suggestions", [])
        )
        all_items = [s for s in all_items if isinstance(s, str) and s.strip()]
        if not all_items:
            return None

        content = "Session reflection:\n" + "\n".join(f"- {s}" for s in all_items)
        embedding = self.embedder.embed(content)
        now = datetime.now(tz=timezone.utc).isoformat()
        frag = MemoryFragment(
            id=new_id(),
            project_id=project_id,
            scope="project",
            category="fact",
            content=content,
            token_count=_estimate_tokens(content),
            confidence=0.85,
            source_type="reflection",
            producer_model=self._producer_model,
            producer_version=self._producer_version,
            source_session=session_id,
            created_at=now,
            last_accessed=now,
            embedding_model=self.embedder.model_name,
            embedding_dim=self.embedder.dim,
            embedding=embedding,
        )
        if not self.dry_run:
            self.db.upsert_fragment(frag)
            self.vector_index.add(frag.id, embedding)
        return frag

    # ── Stage 1: Segmentation ────────────────────────────────────────────────

    def _stage1_segment(self, messages: list[TranscriptMessage]) -> list[Episode]:
        """
        Split messages into semantic episodes by measuring cosine distance
        between adjacent message embeddings. Boundary when distance > 0.3.
        Falls back to fixed-window chunking when fewer than 4 messages.
        """
        if len(messages) < 4:
            return [Episode(messages=messages, start_idx=0, end_idx=len(messages) - 1)]

        # Embed each message content
        texts = [m.content for m in messages]
        embeddings = self.embedder.embed_batch(texts)

        # Compute cosine distances between adjacent embeddings
        BOUNDARY_THRESHOLD = 0.30
        boundaries: list[int] = [0]
        for i in range(1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            dist = 1.0 - sim
            if dist > BOUNDARY_THRESHOLD:
                boundaries.append(i)
        boundaries.append(len(messages))

        episodes: list[Episode] = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            seg_messages = messages[start:end]
            seg_embeddings = embeddings[start:end]

            # Episode embedding = mean of its message embeddings
            mean_emb = _mean_embedding(seg_embeddings)

            # Episode attention weight = max of constituent message weights
            max_weight = max(_attention_weight(m) for m in seg_messages)

            episodes.append(Episode(
                messages=seg_messages,
                start_idx=start,
                end_idx=end - 1,
                embedding=mean_emb,
                attention_weight=max_weight,
            ))

        return episodes

    # ── Stage 2: Distillation ────────────────────────────────────────────────

    def _stage2_distill(
        self,
        episodes: list[Episode],
        session_id: str,
        project_id: str,
    ) -> list[DistilledEpisode]:
        results: list[DistilledEpisode] = []
        n_llm_failures = 0   # episodes dropped because the distill LLM threw
        now = datetime.now(tz=timezone.utc).isoformat()

        for episode in episodes:
            transcript_text = _format_transcript(
                episode.messages,
                max_chars=self.cfg.max_episode_tokens * 4,  # ~4 chars/token
            )
            prompt = f"""Extract structured information from this conversation episode.

{transcript_text}

Output JSON only:
{{
  "recall": "ONE self-contained sentence stating the single most useful thing to remember — phrased to answer the question you'd later search to find it. Name the specific subject (the actual decision/fact/fix), not a vague reference like 'the change' or 'this issue'.",
  "intent": "what the user was trying to accomplish",
  "outcome": "what was achieved or decided",
  "key_decisions": ["list of important decisions made"],
  "tools_used": ["list of tools or commands run"],
  "files_modified": ["list of files created or changed"],
  "errors_encountered": ["list of errors or problems hit"],
  "summary": "one or two sentence summary of this episode",
  "confidence": 0.0,
  "abstraction_level": 0.5
}}

The "recall" line is the headline a future search has to match — make it crisp and standalone, readable with no other context. Keep each field concise. confidence should reflect how clearly the episode communicates its purpose (0.0-1.0).

GROUND EVERY FIELD STRICTLY IN THE CONVERSATION ABOVE. Never introduce a technology, library, tool, product, framework, or proper noun that does not literally appear in the transcript — if you name something specific, it must be quotable from the text, not inferred or assumed typical. A thin or exploratory episode is NOT licence to invent a plausible-sounding answer: if there is no specific grounded takeaway, keep "recall" a plain literal description of what actually happened ("reviewed the config and data-flow files; no decision reached") and set confidence below 0.3.

abstraction_level: float 0.0-1.0. 1.0 = general principle transferable across any project (e.g. "always use pnpm"). 0.0 = project-specific detail only (e.g. "auth middleware is in src/middleware/auth.ts")."""

            try:
                data = self._router.call_json(self._distill_role, prompt)
            except Exception:
                # Distillation failed (LLM down / budget exhausted). Do NOT fall
                # back to storing the raw transcript: an undistilled
                # "User:/Assistant:" turn is verbatim junk, and because it lands
                # as an `episode` the temporal digest ranks it FIRST. A
                # 2026-06-11 audit found 509 such fragments — 16% of live memory —
                # polluting recall from past API-budget outages. A failed episode
                # has no durable value, so we drop it: the session keeps its
                # successfully-distilled episodes, and the LLM failure itself is
                # already surfaced upstream by llm.call_model's budget
                # Issue-Catcher. The budget-fallback (route to local qwen3:8b /
                # claude-code) makes genuine failures rare, so dropping one
                # episode loses almost nothing while keeping memory clean.
                # Count it: if EVERY episode fails this way the session produced
                # nothing, and the redrive worker uses that signal to re-distill
                # the session from cold storage once the LLM is reachable again.
                n_llm_failures += 1
                continue

            summary = data.get("summary", "")
            if not summary:
                continue

            # Boost confidence by attention weight, cap at 1.0
            base_confidence = float(data.get("confidence", 0.7))
            confidence = min(1.0, base_confidence * (0.5 + 0.5 * episode.attention_weight / 5.0))

            # Confidence gate: drop episodes the distiller couldn't make sense of
            # before paying to embed/store them. These are the never-retrieved noise
            # (tool output, acks, command narration) that bloated the store ~40%.
            if confidence < self.cfg.min_episode_confidence:
                continue

            content = _format_episode_content(data)
            embedding = self.embedder.embed(content)

            frag = MemoryFragment(
                id=new_id(),
                project_id=project_id,
                scope="project",
                category="episode",
                content=content,
                token_count=_estimate_tokens(content),
                abstraction_lvl=float(data.get("abstraction_level", 0.5)),
                confidence=confidence,
                source_type="distillation",
                producer_model=self._producer_model,
                producer_version=self._producer_version,
                source_session=session_id,
                created_at=now,
                last_accessed=now,
                embedding_model=self.embedder.model_name,
                embedding_dim=self.embedder.dim,
                embedding=embedding,
            )

            if not self.dry_run:
                self.db.upsert_fragment(frag)
                self.vector_index.add(frag.id, embedding)

            results.append(DistilledEpisode(
                episode=episode,
                fragment=frag,
                entities_raw=data.get("entities", []),
                triples_raw=data.get("relations", []),
            ))

        return results, n_llm_failures

    # ── Stage 3: Entity & relation extraction ────────────────────────────────

    def _stage3_extract_entities(
        self,
        distilled: list[DistilledEpisode],
        project_id: str,
    ) -> None:
        if not distilled:
            return

        _ALLOWED_PREDICATES = frozenset(
            ("depends_on", "calls", "modifies", "caused_by", "resolved_by")
        )
        _ALLOWED_TYPES = frozenset(("file", "function", "class", "package", "config"))

        combined = "\n\n".join(d.fragment.content for d in distilled)
        if len(combined) > 8000:
            combined = combined[:8000]

        # ── Call 1: entities only ─────────────────────────────────────────────
        entity_prompt = f"""Extract all concrete code entities from these session notes.

{combined}

Output JSON only:
{{
  "entities": [
    {{"type": "file|function|class|package|config", "name": "qualified name"}}
  ]
}}

Only include concrete code artifacts (files, functions, classes, packages, config files).
Skip generic terms like 'user', 'error', 'issue', 'service', 'database'."""

        try:
            entity_data = self._router.call_json("cheap", entity_prompt)
        except Exception as e:
            import warnings
            warnings.warn(f"[stage3] entity extraction failed: {e}")
            return

        now = datetime.now(tz=timezone.utc).isoformat()

        entity_id_map: dict[str, str] = {}
        for e_raw in entity_data.get("entities", []):
            name = str(e_raw.get("name", "")).strip()
            etype = str(e_raw.get("type", "file")).strip()
            if not name or etype not in _ALLOWED_TYPES:
                continue

            existing = self.db.get_entity_by_name(project_id, name)
            if existing:
                entity_id_map[name] = existing.id
                continue

            entity = Entity(
                id=new_id(),
                project_id=project_id,
                entity_type=etype,
                qualified_name=name,
                last_seen=now,
            )
            if not self.dry_run:
                self.db.upsert_entity(entity)
            entity_id_map[name] = entity.id

        if not entity_id_map:
            return

        # ── Call 2: relations — grounded to the extracted entity names ────────
        entity_list = "\n".join(f"  - {n}" for n in entity_id_map)
        relation_prompt = f"""Given these code entities extracted from session notes:
{entity_list}

And these session notes:
{combined}

Extract relationships between the entities listed above.
Use ONLY the exact entity names from the list above as subject and object values.
Use ONLY these predicates: depends_on, calls, modifies, caused_by, resolved_by

Output JSON only:
{{
  "relations": [
    {{"subject": "<exact entity name from list>", "predicate": "depends_on|calls|modifies|caused_by|resolved_by", "object": "<exact entity name from list>"}}
  ]
}}

If no clear relationships exist between the listed entities, return {{"relations": []}}"""

        try:
            relation_data = self._router.call_json("cheap", relation_prompt)
        except Exception as e:
            import warnings
            warnings.warn(f"[stage3] relation extraction failed: {e}")
            return

        source_frag_id = distilled[0].fragment.id if distilled else None
        dropped = 0
        for r_raw in relation_data.get("relations", []):
            subj = str(r_raw.get("subject", "")).strip()
            pred = str(r_raw.get("predicate", "")).strip()
            obj  = str(r_raw.get("object", "")).strip()
            if not (subj and pred and obj):
                continue
            if pred not in _ALLOWED_PREDICATES:
                import warnings
                warnings.warn(f"[stage3] dropping triple with unknown predicate '{pred}': {subj!r} → {obj!r}")
                dropped += 1
                continue
            subj_id = entity_id_map.get(subj)
            obj_id  = entity_id_map.get(obj)
            if not (subj_id and obj_id):
                import warnings
                warnings.warn(
                    f"[stage3] dropping triple — name(s) not in entity map: "
                    f"subj={subj!r} (found={subj_id is not None}), "
                    f"obj={obj!r} (found={obj_id is not None})"
                )
                dropped += 1
                continue

            triple = Triple(
                id=new_id(),
                subject_id=subj_id,
                predicate=pred,
                object_id=obj_id,
                project_id=project_id,
                source_fragment=source_frag_id,
                created_at=now,
            )
            if not self.dry_run:
                self.db.upsert_triple(triple)

    # ── Stage 3b: Structural fingerprinting (§3.1) ───────────────────────────

    def _stage3b_structural(
        self,
        distilled: list[DistilledEpisode],
        project_id: str,
    ) -> None:
        """
        For each newly-touched entity, pull its 2-hop subgraph, compute a
        Weisfeiler-Lehman hash, and stage the pattern in pending_structural_patterns.
        No LLM calls — purely hash-based. Articulation fires later in audit().
        """
        if self.dry_run or not distilled:
            return
        # Deferred by default (ledger N2): skip the WL-fingerprint compute on the
        # ingest hot path unless the structural lane is explicitly enabled.
        if not getattr(self.cfg, "structural_fingerprinting_enabled", False):
            return

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        fragment_id = distilled[0].fragment.id

        # Collect all entity IDs touched in this ingest
        touched_names = set()
        for d in distilled:
            for e in d.entities_raw:
                n = str(e.get("name", "")).strip()
                if n:
                    touched_names.add(n)

        for name in touched_names:
            entity = self.db.get_entity_by_name(project_id, name)
            if entity is None:
                continue

            # Pull 2-hop subgraph triples
            neighbor_ids = self.db.get_neighbors(entity.id, max_hops=2)
            all_ids = {entity.id} | set(neighbor_ids)
            triples_raw = []
            for eid in all_ids:
                for row in self.db.get_triples_for_entity(eid):
                    triples_raw.append((
                        row["subj_type"], row["predicate"], row["obj_type"],
                        row["subj_name"], row["obj_name"],
                    ))
            if len(triples_raw) < 2:
                continue

            sig_hash, arity = _wl_hash(triples_raw)

            exemplar = {
                "project_id": project_id,
                "entity_ids": [entity.id],
                "fragment_ids": [fragment_id],
            }

            existing = self.db.get_structural_pattern(sig_hash)
            if existing:
                # Already promoted to structural_patterns — append exemplar
                exemplars = json.loads(existing.exemplars_json)
                exemplars.append(exemplar)
                existing.exemplars_json = json.dumps(exemplars)
                existing.occurrence_count += 1
                existing.last_seen = now_iso
                self.db.upsert_structural_pattern(existing)
                continue

            # Check / update pending
            pending = self.db.get_pending_pattern(sig_hash)
            if pending:
                exemplars = json.loads(pending.exemplars_json)
                exemplars.append(exemplar)
                pending.exemplars_json = json.dumps(exemplars)
                pending.last_seen = now_iso
                self.db.upsert_pending_pattern(pending)
            else:
                new_pending = PendingPattern(
                    id=new_id(),
                    signature_hash=sig_hash,
                    arity=arity,
                    exemplars_json=json.dumps([exemplar]),
                    first_seen=now_iso,
                    last_seen=now_iso,
                )
                self.db.upsert_pending_pattern(new_pending)

    # ── Stage 4: Consolidation & deduplication ────────────────────────────────

    def _stage4_consolidate(
        self,
        new_fragments: list[MemoryFragment],
        project_id: str,
    ) -> None:
        """
        For each new episode fragment:
        1. Check for near-duplicate existing fragments (sim > 0.90) → deprecate old.
        2. Check consolidation trigger: 3+ similar episodes → create semantic fact.
        """
        already_degraded: set[str] = set()
        # §3.2 — fetch open simulations once per ingest, not once per fragment.
        # With REM off by default (rem_enabled=False) this table is always empty,
        # so the per-fragment query was pure dead weight on the hot path (ledger
        # N2). When REM is on, an empty list short-circuits resolution cleanly.
        try:
            open_sims = self.db.get_unresolved_simulations(project_id)
        except Exception as e:
            log.debug("simulation resolution skipped (query failed): %s", e)
            open_sims = []
        for frag in new_fragments:
            if not frag.embedding or frag.category != "episode":
                continue
            if open_sims:
                self._resolve_simulations(frag, project_id, open_sims)
            self._dedup(frag, project_id)
            self._consolidation_check(frag, project_id, already_degraded)

    def _resolve_simulations(
        self,
        new_frag: MemoryFragment,
        project_id: str,
        open_sims: list,
    ) -> None:
        """
        §3.2 — Check whether new_frag's embedding confirms any open simulation prediction.
        Cosine threshold 0.82 (per blueprint). Promotes to simulated_realized on match.
        `open_sims` is fetched once per ingest by the caller (see _stage4_consolidate).
        """
        import base64

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        for sim in open_sims:
            try:
                predicted = json.loads(sim.predicted_facts_json)
            except Exception:
                continue
            for pred in predicted:
                pred_emb_b64 = pred.get("embedding_b64") or pred.get("embedding")
                if not pred_emb_b64:
                    continue
                try:
                    if isinstance(pred_emb_b64, str):
                        pred_bytes = base64.b64decode(pred_emb_b64)
                        pred_emb = json.loads(pred_bytes.decode())
                    else:
                        pred_emb = pred_emb_b64  # already a list
                except Exception:
                    continue
                if not new_frag.embedding:
                    continue
                if _cosine_similarity(new_frag.embedding, pred_emb) >= 0.82:
                    # Promote to simulated_realized fact
                    confidence = min(1.0, sim.prior * 0.8 + 0.2)
                    realized_frag = MemoryFragment(
                        id=new_id(),
                        project_id=project_id,
                        scope="project",
                        category="fact",
                        content=pred.get("text", new_frag.content),
                        token_count=_estimate_tokens(pred.get("text", new_frag.content)),
                        confidence=confidence,
                        source_type="sim_realized",
                        producer_model=self._producer_model,
                        producer_version=self._producer_version,
                        source_session=new_frag.source_session,
                        created_at=now_iso,
                        last_accessed=now_iso,
                        embedding_model=self.embedder.model_name,
                        embedding_dim=self.embedder.dim,
                        embedding=new_frag.embedding,
                        epistemic_class="simulated_realized",
                        realized_from_sim_id=sim.id,
                    )
                    if not self.dry_run:
                        self.db.upsert_fragment(realized_frag)
                        self.vector_index.add(realized_frag.id, new_frag.embedding)
                        self.db.resolve_simulation(
                            sim.id, outcome="confirmed",
                            resolved_at=now_iso,
                            evidence=new_frag.id,
                        )
                    break  # one confirmation per simulation is enough

    def _dedup(self, new_frag: MemoryFragment, project_id: str) -> None:
        """Deprecate near-identical existing fragments (sim > 0.90)."""
        # A batch can contain several near-identical new fragments (e.g. an LLM
        # that distils every segment to the same summary). They are all in the
        # index by stage 4, so without this guard each one deprecates the others
        # — a mutual-deprecation cycle that leaves zero survivors. If a sibling
        # already deprecated THIS fragment, it must not deprecate anything; the
        # canonical copy is whichever one ran _dedup first and is still live.
        current = self.db.get_fragment(new_frag.id)
        if current is None or current.is_deprecated:
            return
        in_scope = self.db.list_fragment_ids([project_id])
        candidates = self.vector_index.query(
            new_frag.embedding, k=5,
            filter_fn=lambda fid: fid in in_scope and fid != new_frag.id,
        )
        for fid, sim in candidates:
            if sim < 0.90:
                break
            existing = self.db.get_fragment(fid)
            if existing and not existing.is_deprecated:
                if not self.dry_run:
                    self.db.mark_deprecated(fid, deprecated_by=new_frag.id)

    def _consolidation_check(
        self,
        new_frag: MemoryFragment,
        project_id: str,
        already_degraded: set[str] | None = None,
    ) -> None:
        """
        If 3+ similar past episodes exist (sim > 0.80), extract a semantic fact.
        Section 7.2.1 — the 'sleep consolidation' analogy.
        """
        if not new_frag.embedding:
            return

        in_scope = self.db.list_fragment_ids([project_id])
        candidates = self.vector_index.query(
            new_frag.embedding, k=10,
            filter_fn=lambda fid: fid in in_scope and fid != new_frag.id,
        )
        similar_episodes = []
        for fid, sim in candidates:
            if sim < 0.80:
                continue
            existing = self.db.get_fragment(fid)
            if (existing and not existing.is_deprecated
                    and existing.category == "episode"):
                similar_episodes.append(existing)

        if len(similar_episodes) < self.cfg.consolidation_threshold:
            return

        # Extract the common pattern across similar episodes
        episodes_text = "\n\n".join(
            f"Episode {i+1}:\n{ep.content}"
            for i, ep in enumerate(similar_episodes[:5])  # cap at 5 to control cost
        )
        prompt = f"""These {len(similar_episodes)} episodes describe similar situations:

{episodes_text}

Extract the GENERAL PRINCIPLE or REUSABLE KNOWLEDGE that is common across all of them.
State it as a timeless fact, not as a specific event. Be concise (1-3 sentences).

Output JSON only:
{{"fact": "the general principle", "confidence": 0.0}}"""

        try:
            # Follow the pipeline's distill role, not a hardcoded "cheap": live
            # ingest stays local/free (role="cheap"), but a strong-role cold-replay
            # re-crystallizes these consolidation facts through the smarter model
            # instead of regenerating them with the weak local one.
            data = self._router.call_json(self._distill_role, prompt)
        except Exception:
            return

        fact_text = str(data.get("fact", "")).strip()
        if not fact_text:
            return

        base_confidence = float(data.get("confidence", 0.7))
        # More episodes → higher confidence, capped at 1.0
        confidence = min(1.0, base_confidence + 0.1 * min(len(similar_episodes), 3))

        # Confidence gate — same floor the episode path uses (see _stage2_distill).
        # The consolidation path also emits category="fact" through the cheap local
        # model; without this gate a vague "general principle" the model wasn't sure
        # of lands in the active retrieval pool and dilutes recall. (Verified hole:
        # 393 active consolidation facts sat below this floor, ~6 citations total.)
        if confidence < self.cfg.min_episode_confidence:
            return

        embedding = self.embedder.embed(fact_text)
        now = datetime.now(tz=timezone.utc).isoformat()

        fact_frag = MemoryFragment(
            id=new_id(),
            project_id=project_id,
            scope="project",
            category="fact",
            content=fact_text,
            token_count=_estimate_tokens(fact_text),
            confidence=confidence,
            source_type="consolidation",
            producer_model=self._producer_model,
            producer_version=self._producer_version,
            source_session=new_frag.source_session,
            created_at=now,
            last_accessed=now,
            embedding_model=self.embedder.model_name,
            embedding_dim=self.embedder.dim,
            embedding=embedding,
        )

        if already_degraded is None:
            already_degraded = set()

        if not self.dry_run:
            self.db.upsert_fragment(fact_frag)
            self.vector_index.add(fact_frag.id, embedding)
            # Halve access counts on source episodes to reduce their CRS.
            # Guard against repeated halving within the same ingest call.
            degraded_now: list[str] = []
            for ep in similar_episodes:
                if ep.id not in already_degraded:
                    self.db.execute(
                        "UPDATE memory_fragments SET access_count = access_count / 2 WHERE id=?",
                        (ep.id,),
                    )
                    already_degraded.add(ep.id)
                    degraded_now.append(ep.id)
            # Record the consolidation so the CRS degradation is auditable: a fact
            # was crystallized from these episodes and their access_count halved.
            if degraded_now:
                try:
                    self.db.log_event("consolidation", now, {
                        "project_id": project_id,
                        "fact_id": fact_frag.id,
                        "source_episode_ids": degraded_now,
                        "access_count_op": "halved",
                        "producer_model": self._producer_model,
                    })
                except Exception:
                    pass  # observability must never break the write path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_transcript(messages: list[TranscriptMessage], max_chars: int = 6000) -> str:
    lines: list[str] = []
    for m in messages:
        prefix = {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(m.role, m.role)
        lines.append(f"{prefix}: {m.content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    # Keep both ends — the outcome (tail) matters as much as the setup (head).
    half = max_chars // 2
    return text[:half] + "\n...[truncated]\n" + text[-half:]


def _format_episode_content(data: dict) -> str:
    parts: list[str] = []
    # The crisp standalone claim leads, so the fragment opens with the answer a
    # future query is looking for rather than topical metadata. Falls back to
    # summary-first when an older/cheaper model didn't produce a recall line.
    if data.get("recall"):
        parts.append(str(data["recall"]).strip())
    if data.get("summary"):
        parts.append(data["summary"])
    if data.get("intent"):
        parts.append(f"Intent: {data['intent']}")
    if data.get("outcome"):
        parts.append(f"Outcome: {data['outcome']}")
    for key, label in [("key_decisions", "Decisions"), ("tools_used", "Tools"),
                       ("files_modified", "Files"), ("errors_encountered", "Errors")]:
        items = data.get(key, [])
        if items:
            parts.append(f"{label}: {', '.join(str(i) for i in items)}")
    return "\n".join(parts)


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    mean = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean)) or 1.0
    return [x / norm for x in mean]


def _estimate_tokens(text: str) -> int:
    # ~4 characters per token is a reasonable approximation for English prose
    return max(1, len(text) // 4)


def _wl_hash(
    triples: list[tuple[str, str, str, str, str]],
    iterations: int = 3,
) -> tuple[str, int]:
    """
    §3.1 — Weisfeiler-Lehman hash of a typed subgraph.
    Input triples: (subj_type, predicate, obj_type, subj_name, obj_name)
    Returns (hash_hex, arity) where arity = distinct node count.

    Each node is identified by its concrete name but labeled by its entity_type,
    so two graphs with isomorphic (type, predicate) structure produce the same hash.
    """
    # Build adjacency using concrete names as node IDs but entity_type as labels
    node_labels: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []  # (subj_name, predicate, obj_name)

    for subj_type, pred, obj_type, subj_name, obj_name in triples:
        node_labels[subj_name] = subj_type
        node_labels[obj_name]  = obj_type
        edges.append((subj_name, pred, obj_name))

    if not node_labels:
        return hashlib.sha256(b"empty").hexdigest(), 0

    for _ in range(iterations):
        new_labels: dict[str, str] = {}
        for node, label in node_labels.items():
            out_nbrs = sorted(
                f"{pred}:{node_labels.get(obj, '?')}"
                for (s, pred, obj) in edges if s == node
            )
            in_nbrs = sorted(
                f"<-{pred}:{node_labels.get(subj, '?')}"
                for (subj, pred, o) in edges if o == node
            )
            combined = label + "|" + ",".join(out_nbrs + in_nbrs)
            new_labels[node] = hashlib.sha256(combined.encode()).hexdigest()[:12]
        node_labels = new_labels

    canonical = "|".join(sorted(node_labels.values()))
    return hashlib.sha256(canonical.encode()).hexdigest(), len(node_labels)
