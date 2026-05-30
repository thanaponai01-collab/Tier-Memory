"""
memory_system — Steps 1–5: Foundation + retrieval + pipeline + daemon + self-improvement.

Public surface:
    MemoryConfig, load_config        — configuration
    LLMRolesConfig                   — §6.2 role-based LLM routing config
    Database                         — SQLite wrapper + all CRUD ops
    MemoryFragment, Session, Entity, Triple, Correction  — domain models
    Score                            — §6.5 rich CRS result with components
    RetrievalEvent                   — §5.2 retrieval event log entry
    new_id                           — ULID generation
    VectorIndex                      — HNSW vector index (requires usearch/hnswlib)
    composite_relevance_score, tier  — CRS calculator + tier classification
    fused_retrieval, assemble_prompt — 4-signal RRF retrieval + prompt builder
    Embedder, RandomEmbedder,
    CachedEmbedder, OpenAIEmbedder   — embedding backends
    cosine_similarity                — §6.6 canonical cosine math
    LLMRouter                        — §6.2 role-keyed LLM dispatch
    get_llm_stats                    — §5.4 cumulative call / token counter
    ConsolidationPipeline            — 4-stage compression pipeline
    TranscriptMessage                — input type for the pipeline
    MemoryAuditor, AuditReport       — self-improvement audit loop
    ModelUpgradeReindexJob,
    ReindexReport                    — vector re-embedding after model upgrade
"""

from .config import (
    MemoryConfig, DaemonConfig, StorageConfig, EmbeddingConfig,
    RetrievalConfig, CompressionConfig, EvictionConfig,
    CrossProjectConfig, SelfImprovementConfig, LLMRolesConfig, load_config,
)
from .models import (
    MemoryFragment, Session, Entity, Triple, Correction, RetrievalResult,
    Score, RetrievalEvent, Event,
)
from .schema import Database
from .ids import new_id
from .vector_index import VectorIndex
from .scoring import composite_relevance_score, tier
from .retrieval import fused_retrieval, assemble_prompt
from .embedder import Embedder, RandomEmbedder, CachedEmbedder, OpenAIEmbedder, cosine_similarity
from .llm import LLMRouter, get_llm_stats
from .pipeline import ConsolidationPipeline, TranscriptMessage
from .auditor import MemoryAuditor, AuditReport
from .reindex import ModelUpgradeReindexJob, ReindexReport

__all__ = [
    "MemoryConfig", "DaemonConfig", "StorageConfig", "EmbeddingConfig",
    "RetrievalConfig", "CompressionConfig", "EvictionConfig",
    "CrossProjectConfig", "SelfImprovementConfig", "LLMRolesConfig", "load_config",
    "MemoryFragment", "Session", "Entity", "Triple", "Correction",
    "RetrievalResult", "Score", "RetrievalEvent", "Event",
    "Database", "new_id",
    "VectorIndex",
    "composite_relevance_score", "tier",
    "fused_retrieval", "assemble_prompt",
    "Embedder", "RandomEmbedder", "CachedEmbedder", "OpenAIEmbedder", "cosine_similarity",
    "LLMRouter", "get_llm_stats",
    "ConsolidationPipeline", "TranscriptMessage",
    "MemoryAuditor", "AuditReport",
    "ModelUpgradeReindexJob", "ReindexReport",
]
