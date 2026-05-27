"""
Fusion retrieval — Section 5.2 of the architecture.

Four-signal RRF (Reciprocal Rank Fusion):
  Signal 1: Vector semantic search (hnswlib HNSW)
  Signal 2: Graph neighbourhood expansion (2-hop from matched entities)
  Signal 3: BM25 full-text search (SQLite FTS5)
  Signal 4: Structural pattern lane (§3.1) — fires when literal lane is weak

Also applies:
  §3.6 Correction injection   — pinned L1 facts from corrections table
  §3.4 Chaos flag             — warns when top-K mean mutation_velocity is high
"""

from __future__ import annotations
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from .models import MemoryFragment, RetrievalResult
from .schema import Database
from .vector_index import VectorIndex
from .scoring import composite_relevance_score
from .config import CrossProjectConfig, RetrievalConfig
from .scoring import _cosine_similarity

if TYPE_CHECKING:
    from .embedder import CachedEmbedder


# ── SemanticGate (§3.3) ──────────────────────────────────────────────────────

def _passes_semantic_gate(
    frag: MemoryFragment,
    query_embedding: list[float],
    abstraction_threshold: float,
    similarity_threshold: float,
    transferable_categories: set,
) -> bool:
    if frag.project_id != "__global__":
        return True
    if frag.abstraction_lvl < abstraction_threshold:
        return False
    if frag.category not in transferable_categories:
        return False
    if frag.embedding:
        if _cosine_similarity(frag.embedding, query_embedding) < similarity_threshold:
            return False
    return True


# ── Public API ───────────────────────────────────────────────────────────────

def fused_retrieval(
    db: Database,
    vector_index: VectorIndex,
    query_embedding: list[float],
    query_text: str,
    project_id: str,
    token_budget: int,
    cfg: RetrievalConfig,
    include_global: bool = True,
    cross_project_cfg: Optional[CrossProjectConfig] = None,
    patterns_index: Optional[VectorIndex] = None,   # §3.1 structural lane
    embedder: Optional["CachedEmbedder"] = None,    # §3.6 correction injection
) -> RetrievalResult:
    """
    Returns a RetrievalResult with fragments packed within token_budget.
    Side-effect: updates last_accessed + access_count for every returned fragment.
    """
    project_ids = [project_id]
    if include_global:
        project_ids.append("__global__")

    # ── Signal 1: vector search ─────────────────────────────────────────────
    # Pre-load the in-scope ID set so the HNSW filter is an O(1) lookup rather
    # than a SQL query per candidate (which compounds badly at high ef_search).
    in_scope_ids = db.list_fragment_ids(project_ids)

    vec_results: list[tuple[str, float]] = vector_index.query(
        query_embedding, k=30, filter_fn=lambda fid: fid in in_scope_ids
    )

    # ── Signal 2: graph neighbourhood ──────────────────────────────────────
    matched_entities = db.fuzzy_match_entities(project_id, query_text, limit=5)
    graph_frag_ids: list[str] = []
    for entity in matched_entities:
        neighbors = db.get_neighbors(entity.id, max_hops=2)
        for nid in neighbors:
            for fid in db.fragments_linked_to_entity(nid):
                if fid not in graph_frag_ids:
                    graph_frag_ids.append(fid)
        for fid in db.fragments_linked_to_entity(entity.id):
            if fid not in graph_frag_ids:
                graph_frag_ids.append(fid)

    # ── Signal 3: BM25 ─────────────────────────────────────────────────────
    bm25_results: list[tuple[str, float]] = []
    for pid in project_ids:
        bm25_results.extend(db.bm25_search(query_text, pid, limit=20))

    K = cfg.rrf_k

    # ── Signal 4: structural pattern lane (§3.1) ───────────────────────────
    top_vec_sim = max((s for _, s in vec_results), default=0.0)
    structural_frag_ids: list[tuple[str, float]] = []  # (frag_id, attenuated_score)
    if patterns_index is not None and top_vec_sim < cfg.structural_gate:
        pattern_hits = patterns_index.query(query_embedding, k=5)
        for rank, (pid, _) in enumerate(pattern_hits):
            pattern_row = db.get_structural_pattern_by_id(pid)
            if pattern_row is None:
                continue
            exemplars = json.loads(pattern_row.exemplars_json)
            attenuation = cfg.structural_weight / (K + rank) * (1.0 - top_vec_sim)
            for ex in exemplars:
                for fid in ex.get("fragment_ids", []):
                    structural_frag_ids.append((fid, attenuation))

    # ── Reciprocal Rank Fusion ──────────────────────────────────────────────
    scores: dict[str, float] = defaultdict(float)

    for rank, (fid, _) in enumerate(vec_results):
        scores[fid] += cfg.vector_weight / (K + rank)

    for rank, fid in enumerate(graph_frag_ids):
        scores[fid] += cfg.graph_weight / (K + rank)

    for rank, (fid, _) in enumerate(bm25_results):
        scores[fid] += cfg.bm25_weight / (K + rank)

    for fid, score in structural_frag_ids:
        scores[fid] += score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── Load fragments & compute CRS ────────────────────────────────────────
    fragments_with_crs: list[tuple[float, MemoryFragment]] = []
    for fid, _ in ranked:
        frag = db.get_fragment(fid)
        if frag is None or frag.is_deprecated:
            continue
        crs = composite_relevance_score(frag, query_embedding)
        if crs < cfg.min_crs:
            continue
        frag.crs = crs
        fragments_with_crs.append((crs, frag))

    # Sort by CRS descending (RRF rank already broke most ties; CRS re-ranks)
    fragments_with_crs.sort(key=lambda x: x[0], reverse=True)

    # ── Token-budget knapsack packing ───────────────────────────────────────
    _cp = cross_project_cfg or CrossProjectConfig()
    _transferable = set(_cp.transferable_categories)

    selected: list[MemoryFragment] = []
    tokens_used = 0
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    for crs, frag in fragments_with_crs:
        if len(selected) >= cfg.max_fragments:
            break
        if not _passes_semantic_gate(
            frag, query_embedding,
            _cp.abstraction_threshold,
            _cp.similarity_threshold,
            _transferable,
        ):
            continue
        if tokens_used + frag.token_count > token_budget:
            continue  # skip but keep scanning (greedy knapsack)
        selected.append(frag)
        tokens_used += frag.token_count
        db.touch_fragment(frag.id, now_iso)

    # ── §3.6 Correction injection ───────────────────────────────────────────
    # Scan recent corrections for original_fact that closely matches the query;
    # prepend corrected_fact as a pinned fragment so it lands at L1.
    injected: list[MemoryFragment] = []
    if embedder is not None:
        try:
            injected = _inject_corrections(
                db, embedder, query_embedding, project_id,
                cfg.correction_injection_threshold,
            )
        except Exception:
            pass  # never let correction injection break retrieval

    # ── §3.4 Chaos flag ────────────────────────────────────────────────────
    chaos_flag = False
    if selected:
        velocities = [getattr(f, "mutation_velocity", 0.0) for f in selected]
        mean_vel = sum(velocities) / len(velocities)
        all_vels = db.get_mutation_velocities(project_id)
        if all_vels:
            threshold_idx = int(len(all_vels) * cfg.chaos_velocity_percentile)
            threshold = sorted(all_vels)[min(threshold_idx, len(all_vels) - 1)]
            chaos_flag = mean_vel > threshold

    return RetrievalResult(
        fragments=injected + selected,
        project_summary=_build_project_summary(db, project_id),
        global_profile_hash=None,   # populated by the daemon layer (Step 5)
        token_budget_used=tokens_used,
        chaos_flag=chaos_flag,
    )


# ── Prompt assembly ──────────────────────────────────────────────────────────

def assemble_prompt(
    system_prompt: str,
    global_profile: Optional[str],
    result: RetrievalResult,
    user_query: str,
    file_contents: Optional[str] = None,
) -> str:
    """
    Builds the final prompt with stable L0+L1 prefix ordering to maximise
    prompt-cache hit rate (Section 2.2).

    L0 → system_prompt + global_profile  (most stable)
    L1 → project_memory                  (semi-stable)
    L2 → recalled_memories               (dynamic, ordered by CRS for consistency)
    L3 → active_files + user_message     (fully dynamic)
    """
    sections: list[str] = []

    # L0
    sections.append(system_prompt)
    if global_profile:
        sections.append(f"<global_profile>\n{global_profile}\n</global_profile>")

    # L1
    if result.project_summary:
        sections.append(
            f"<project_memory>\n{result.project_summary}\n</project_memory>"
        )
    if result.chaos_flag:
        sections.append(
            "<chaos_warning>The knowledge domain for this query is currently "
            "highly volatile — recent contradictions suggest facts may be "
            "changing rapidly. Treat retrieved memories with extra skepticism.</chaos_warning>"
        )

    # L2
    if result.fragments:
        frag_block = "<recalled_memories>\n"
        for frag in result.fragments:  # already sorted by CRS descending
            frag_block += (
                f'<memory id="{frag.id}" scope="{frag.scope}" '
                f'confidence="{frag.confidence:.2f}" '
                f'from="{frag.created_at[:10]}">\n'
                f"{frag.content}\n"
                f"</memory>\n"
            )
        frag_block += "</recalled_memories>"
        sections.append(frag_block)

    # L3
    if file_contents:
        sections.append(f"<active_files>\n{file_contents}\n</active_files>")
    sections.append(f"<user_message>\n{user_query}\n</user_message>")

    return "\n\n".join(sections)


# ── Private helpers ──────────────────────────────────────────────────────────

def _inject_corrections(
    db: Database,
    embedder: "CachedEmbedder",
    query_embedding: list[float],
    project_id: str,
    threshold: float,
) -> list[MemoryFragment]:
    """
    §3.6 — Scan recent corrections for original_fact that cosine-matches the query.
    Returns pinned-style fragments carrying the corrected_fact text.
    """
    corrections = db.get_corrections(project_id)
    injected: list[MemoryFragment] = []
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    for corr in corrections[-50:]:  # cap scan to 50 most recent
        orig_emb = embedder.embed(corr.original_fact)
        if _cosine_similarity(orig_emb, query_embedding) >= threshold:
            frag = MemoryFragment(
                id=f"corr_{corr.id}",
                project_id=project_id,
                scope="project",
                category="correction",
                content=f"[CORRECTION] {corr.corrected_fact}",
                token_count=max(1, len(corr.corrected_fact) // 4),
                confidence=1.0,
                is_pinned=True,
                created_at=corr.created_at,
                last_accessed=now_iso,
                epistemic_class="observed",
            )
            injected.append(frag)
    return injected


def _build_project_summary(db: Database, project_id: str) -> Optional[str]:
    """
    Return the synthesized project summary if one exists, otherwise fall back
    to the most recent session summary.
    """
    synthesized = db.get_project_summary(project_id)
    if synthesized:
        return synthesized
    row = db.fetchone(
        "SELECT summary FROM sessions WHERE project_id=? AND summary IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 1",
        (project_id,),
    )
    return row["summary"] if row else None
