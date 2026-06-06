"""
Configuration loader. Reads memory.yaml from ~/.agent/ or a path override.
All values have safe defaults so the system works out of the box.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class DaemonConfig:
    socket_path: str = str(Path.home() / ".agent" / "memoryd.sock")
    max_memory_mb: int = 512
    write_queue_size: int = 1000
    # How often the background watchdog actively probes the embedder so a silent
    # failure is caught and timestamped even when nobody is looking at status.
    embedder_watchdog_interval_secs: int = 300


@dataclass
class StorageConfig:
    db_path: str = str(Path.home() / ".agent" / "memory" / "memory.db")
    cold_storage_path: str = str(Path.home() / ".agent" / "memory" / "cold")
    backup_path: str = str(Path.home() / ".agent" / "memory" / "backups")
    backup_interval_hours: int = 168  # weekly


@dataclass
class EmbeddingConfig:
    model: str = "ollama/nomic-embed-text"
    dimensions: int = 768
    batch_size: int = 100
    cache_size: int = 10_000


@dataclass
class RetrievalConfig:
    default_token_budget: int = 4_000
    max_fragments: int = 15
    min_crs: float = 0.30
    rrf_k: int = 60
    vector_k: int = 30         # how deep the vector lane looks for candidates
    # HyDE query expansion: sketch a hypothetical answer and embed THAT for the
    # vector lane, to bridge the query↔memory phrasing gap. Wired but OFF by
    # default: measured 2026-06-04 (mem eval) it tied/regressed recall because
    # the only locally-available model (qwen3:8b) is too slow (~13s/call on the
    # pre-answer hot path) and too uninformed to write accurate hypothetical
    # answers — it hallucinates generic ones that steer toward the wrong
    # memories. Turn on (hyde_role="cheap") once a fast, capable model with an
    # API key is configured; re-run `mem eval` before trusting it.
    hyde_enabled: bool = False
    hyde_role: str = "cheap"   # LLMRouter role used for expansion
    hyde_max_tokens: int = 120
    vector_weight: float = 1.0
    graph_weight: float = 0.7
    bm25_weight: float = 0.5
    # §3.1 structural lane
    structural_gate: float = 0.55    # invoke structural lane when top vec sim < this
    structural_weight: float = 0.4   # RRF weight for structural hits
    # §3.4 chaos flag
    chaos_velocity_percentile: float = 0.85  # flag if top-K mean velocity > this percentile
    correction_injection_threshold: float = 0.85  # cosine threshold for §3.6


@dataclass
class CompressionConfig:
    distillation_model: str = "claude-haiku-4-5-20251001"
    distillation_endpoint: str = "http://localhost:11434"
    entity_extraction_model: str = "claude-haiku-4-5-20251001"
    entity_extraction_endpoint: str = "http://localhost:11434"
    consolidation_threshold: int = 3
    max_episode_tokens: int = 500
    # §3.1 structural fingerprinting (Weisfeiler-Lehman) on the ingest hot path.
    # OFF by default (deferred 2026-06-06, ledger N2): after 4,487 fragments of
    # real single-user use the structural_patterns / pending_structural_patterns
    # tables sat at 0 rows — the cross-project quorum (pattern_quorum_projects=2)
    # is unsatisfiable in a one-project system, so this lane only ever burned
    # 2-hop subgraph + WL-hash compute per ingest with nothing to show. Turn on
    # together with pattern_articulation_enabled once there's a second project.
    structural_fingerprinting_enabled: bool = False


@dataclass
class EvictionConfig:
    recency_decay_lambda: float = 0.003   # half-life ≈ 10 days
    hot_threshold: float = 0.60
    warm_threshold: float = 0.15
    sweep_interval_hours: int = 168       # weekly


@dataclass
class CrossProjectConfig:
    enabled: bool = True
    abstraction_threshold: float = 0.6
    similarity_threshold: float = 0.70
    transferable_categories: list[str] = field(default_factory=lambda: [
        "coding_style", "architectural_pattern", "debugging_technique",
        "tool_preference", "workflow_convention", "language_idiom",
    ])


@dataclass
class SelfImprovementConfig:
    audit_interval_hours: int = 168
    summary_refresh_interval_minutes: int = 60
    contradiction_detection: bool = True
    contradiction_model: str = "ollama/qwen3:8b"
    contradiction_endpoint: str = "http://localhost:11434"
    confidence_decay_factor: float = 0.85
    confidence_decay_after_days: int = 90
    model_upgrade_reindex: bool = True
    reindex_cold_sessions_limit: int = 500
    # §3.4 velocity-weighted decay
    velocity_ewma_alpha: float = 0.15       # α for EWMA update each audit sweep
    velocity_decay_lambda0: float = 0.003   # base decay rate (matching existing RECENCY_DECAY_LAMBDA)
    velocity_decay_kappa: float = 0.02      # velocity multiplier for extra decay
    epistemic_audit_model: str = "ollama/qwen3:8b"
    epistemic_audit_endpoint: str = "http://localhost:11434"
    # §3.3 crystallization
    crystallization_enabled: bool = True
    crystallization_model: str = "claude-haiku-4-5-20251001"
    crystallization_endpoint: str = "http://localhost:11434"
    crystallization_min_cluster: int = 3
    # §3.1 structural lane — OFF by default (deferred 2026-06-06, ledger N2):
    # nothing to articulate. pending_structural_patterns never fills (see
    # CompressionConfig.structural_fingerprinting_enabled) and promotion needs a
    # cross-project quorum a single-project system can't reach. Re-enable in
    # lockstep with structural_fingerprinting_enabled once a 2nd project exists.
    pattern_articulation_enabled: bool = False
    pattern_articulation_model: str = "claude-haiku-4-5-20251001"
    pattern_articulation_endpoint: str = "http://localhost:11434"
    pattern_quorum_occurrences: int = 3
    pattern_quorum_projects: int = 2
    # §3.2 REM cycle
    rem_enabled: bool = False               # off by default; enable per-project once cost measured
    rem_seed_count: int = 5
    rem_perturbations_per_seed: int = 3
    rem_cycles_per_audit: int = 1
    rem_min_prior: float = 0.10
    rem_sim_expiry_days: int = 60
    rem_model: str = "claude-haiku-4-5-20251001"
    rem_endpoint: str = "http://localhost:11434"
    # §3.7 meta-learning — learn from patterns of bad memory
    meta_learn_enabled: bool = True
    meta_learn_model: str = "claude-haiku-4-5-20251001"
    meta_learn_endpoint: str = "http://localhost:11434"
    meta_learn_lookback_days: int = 30


@dataclass
class LLMRolesConfig:
    """
    §6.2 — One model per role. Replaces the per-stage model+endpoint scattered
    across CompressionConfig and SelfImprovementConfig.
    LLMRouter reads these values; existing per-stage fields remain for backward compat.
    """
    cheap_model: str = "claude-haiku-4-5-20251001"
    cheap_endpoint: Optional[str] = None              # None → Anthropic SDK
    medium_model: str = "ollama/qwen3:8b"
    medium_endpoint: str = "http://localhost:11434"
    strong_model: str = "claude-sonnet-4-6"
    strong_endpoint: Optional[str] = None
    # When a paid (Anthropic) call fails because credit is exhausted (or the key
    # is unusable), LLM work degrades to this local model instead of failing
    # silently — so memory keeps forming once the API budget runs out. Must be a
    # local 'ollama/...' model (free at point of use).
    budget_fallback_model: str = "ollama/qwen3:8b"
    budget_fallback_endpoint: str = "http://localhost:11434"


@dataclass
class MemoryConfig:
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    cross_project: CrossProjectConfig = field(default_factory=CrossProjectConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)
    llm_roles: LLMRolesConfig = field(default_factory=LLMRolesConfig)


def load_config(path: Optional[str | Path] = None) -> MemoryConfig:
    """
    Load config from a YAML file. Falls back to all defaults if the file
    doesn't exist or PyYAML is not installed.
    """
    if path is None:
        env = os.environ.get("AGENT_MEMORY_CONFIG", "")
        path = Path(env) if env else Path.home() / ".agent" / "memory.yaml"
    path = Path(path)

    if not path.exists() or not _HAS_YAML:
        return MemoryConfig()

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    mem = raw.get("memory", {})

    def _get(section: str, key: str, default):
        return mem.get(section, {}).get(key, default)

    cfg = MemoryConfig()

    # Daemon
    d = mem.get("daemon", {})
    if d:
        cfg.daemon = DaemonConfig(
            socket_path=d.get("socket_path", cfg.daemon.socket_path),
            max_memory_mb=d.get("max_memory_mb", cfg.daemon.max_memory_mb),
            write_queue_size=d.get("write_queue_size", cfg.daemon.write_queue_size),
            embedder_watchdog_interval_secs=d.get(
                "embedder_watchdog_interval_secs",
                cfg.daemon.embedder_watchdog_interval_secs,
            ),
        )

    # Storage
    s = mem.get("storage", {})
    if s:
        cfg.storage = StorageConfig(
            db_path=s.get("db_path", cfg.storage.db_path),
            cold_storage_path=s.get("cold_storage_path", cfg.storage.cold_storage_path),
            backup_path=s.get("backup_path", cfg.storage.backup_path),
            backup_interval_hours=s.get("backup_interval_hours", cfg.storage.backup_interval_hours),
        )

    # Embedding
    e = mem.get("embedding", {})
    if e:
        cfg.embedding = EmbeddingConfig(
            model=e.get("model", cfg.embedding.model),
            dimensions=e.get("dimensions", cfg.embedding.dimensions),
            batch_size=e.get("batch_size", cfg.embedding.batch_size),
            cache_size=e.get("cache_size", cfg.embedding.cache_size),
        )

    # Retrieval
    r = mem.get("retrieval", {})
    if r:
        cfg.retrieval = RetrievalConfig(
            default_token_budget=r.get("default_token_budget", cfg.retrieval.default_token_budget),
            max_fragments=r.get("max_fragments", cfg.retrieval.max_fragments),
            min_crs=r.get("min_crs", cfg.retrieval.min_crs),
            rrf_k=r.get("rrf_k", cfg.retrieval.rrf_k),
            vector_k=r.get("vector_k", cfg.retrieval.vector_k),
            hyde_enabled=r.get("hyde_enabled", cfg.retrieval.hyde_enabled),
            hyde_role=r.get("hyde_role", cfg.retrieval.hyde_role),
            hyde_max_tokens=r.get("hyde_max_tokens", cfg.retrieval.hyde_max_tokens),
            vector_weight=r.get("vector_weight", cfg.retrieval.vector_weight),
            graph_weight=r.get("graph_weight", cfg.retrieval.graph_weight),
            bm25_weight=r.get("bm25_weight", cfg.retrieval.bm25_weight),
            structural_gate=r.get("structural_gate", cfg.retrieval.structural_gate),
            structural_weight=r.get("structural_weight", cfg.retrieval.structural_weight),
            chaos_velocity_percentile=r.get("chaos_velocity_percentile", cfg.retrieval.chaos_velocity_percentile),
            correction_injection_threshold=r.get("correction_injection_threshold", cfg.retrieval.correction_injection_threshold),
        )

    # Compression
    c = mem.get("compression", {})
    if c:
        cfg.compression = CompressionConfig(
            distillation_model=c.get("distillation_model", cfg.compression.distillation_model),
            distillation_endpoint=c.get("distillation_endpoint", cfg.compression.distillation_endpoint),
            entity_extraction_model=c.get("entity_extraction_model", cfg.compression.entity_extraction_model),
            entity_extraction_endpoint=c.get("entity_extraction_endpoint", cfg.compression.entity_extraction_endpoint),
            consolidation_threshold=c.get("consolidation_threshold", cfg.compression.consolidation_threshold),
            max_episode_tokens=c.get("max_episode_tokens", cfg.compression.max_episode_tokens),
            structural_fingerprinting_enabled=c.get("structural_fingerprinting_enabled", cfg.compression.structural_fingerprinting_enabled),
        )

    # Eviction
    ev = mem.get("eviction", {})
    if ev:
        cfg.eviction = EvictionConfig(
            recency_decay_lambda=ev.get("recency_decay_lambda", cfg.eviction.recency_decay_lambda),
            hot_threshold=ev.get("hot_threshold", cfg.eviction.hot_threshold),
            warm_threshold=ev.get("warm_threshold", cfg.eviction.warm_threshold),
            sweep_interval_hours=ev.get("sweep_interval_hours", cfg.eviction.sweep_interval_hours),
        )

    # Cross-project
    cp = mem.get("cross_project", {})
    if cp:
        cfg.cross_project = CrossProjectConfig(
            enabled=cp.get("enabled", cfg.cross_project.enabled),
            abstraction_threshold=cp.get("abstraction_threshold", cfg.cross_project.abstraction_threshold),
            similarity_threshold=cp.get("similarity_threshold", cfg.cross_project.similarity_threshold),
            transferable_categories=cp.get("transferable_categories", cfg.cross_project.transferable_categories),
        )

    # Self-improvement
    si = mem.get("self_improvement", {})
    if si:
        c0 = cfg.self_improvement
        cfg.self_improvement = SelfImprovementConfig(
            audit_interval_hours=si.get("audit_interval_hours", c0.audit_interval_hours),
            summary_refresh_interval_minutes=si.get("summary_refresh_interval_minutes", c0.summary_refresh_interval_minutes),
            contradiction_detection=si.get("contradiction_detection", c0.contradiction_detection),
            contradiction_model=si.get("contradiction_model", c0.contradiction_model),
            contradiction_endpoint=si.get("contradiction_endpoint", c0.contradiction_endpoint),
            confidence_decay_factor=si.get("confidence_decay_factor", c0.confidence_decay_factor),
            confidence_decay_after_days=si.get("confidence_decay_after_days", c0.confidence_decay_after_days),
            model_upgrade_reindex=si.get("model_upgrade_reindex", c0.model_upgrade_reindex),
            reindex_cold_sessions_limit=si.get("reindex_cold_sessions_limit", c0.reindex_cold_sessions_limit),
            velocity_ewma_alpha=si.get("velocity_ewma_alpha", c0.velocity_ewma_alpha),
            velocity_decay_lambda0=si.get("velocity_decay_lambda0", c0.velocity_decay_lambda0),
            velocity_decay_kappa=si.get("velocity_decay_kappa", c0.velocity_decay_kappa),
            rem_enabled=si.get("rem_enabled", c0.rem_enabled),
            rem_seed_count=si.get("rem_seed_count", c0.rem_seed_count),
            rem_perturbations_per_seed=si.get("rem_perturbations_per_seed", c0.rem_perturbations_per_seed),
            rem_cycles_per_audit=si.get("rem_cycles_per_audit", c0.rem_cycles_per_audit),
            rem_min_prior=si.get("rem_min_prior", c0.rem_min_prior),
            rem_sim_expiry_days=si.get("rem_sim_expiry_days", c0.rem_sim_expiry_days),
            rem_model=si.get("rem_model", c0.rem_model),
            rem_endpoint=si.get("rem_endpoint", c0.rem_endpoint),
            meta_learn_enabled=si.get("meta_learn_enabled", c0.meta_learn_enabled),
            meta_learn_model=si.get("meta_learn_model", c0.meta_learn_model),
            meta_learn_endpoint=si.get("meta_learn_endpoint", c0.meta_learn_endpoint),
            meta_learn_lookback_days=si.get("meta_learn_lookback_days", c0.meta_learn_lookback_days),
            crystallization_enabled=si.get("crystallization_enabled", c0.crystallization_enabled),
            crystallization_model=si.get("crystallization_model", c0.crystallization_model),
            crystallization_endpoint=si.get("crystallization_endpoint", c0.crystallization_endpoint),
            crystallization_min_cluster=si.get("crystallization_min_cluster", c0.crystallization_min_cluster),
            pattern_articulation_enabled=si.get("pattern_articulation_enabled", c0.pattern_articulation_enabled),
            pattern_articulation_model=si.get("pattern_articulation_model", c0.pattern_articulation_model),
            pattern_articulation_endpoint=si.get("pattern_articulation_endpoint", c0.pattern_articulation_endpoint),
            pattern_quorum_occurrences=si.get("pattern_quorum_occurrences", c0.pattern_quorum_occurrences),
            pattern_quorum_projects=si.get("pattern_quorum_projects", c0.pattern_quorum_projects),
        )

    # LLM roles
    lr = mem.get("llm_roles", {})
    if lr:
        c0 = cfg.llm_roles
        cfg.llm_roles = LLMRolesConfig(
            cheap_model=lr.get("cheap_model", c0.cheap_model),
            cheap_endpoint=lr.get("cheap_endpoint", c0.cheap_endpoint),
            medium_model=lr.get("medium_model", c0.medium_model),
            medium_endpoint=lr.get("medium_endpoint", c0.medium_endpoint),
            strong_model=lr.get("strong_model", c0.strong_model),
            strong_endpoint=lr.get("strong_endpoint", c0.strong_endpoint),
            budget_fallback_model=lr.get("budget_fallback_model", c0.budget_fallback_model),
            budget_fallback_endpoint=lr.get("budget_fallback_endpoint", c0.budget_fallback_endpoint),
        )

    return cfg
