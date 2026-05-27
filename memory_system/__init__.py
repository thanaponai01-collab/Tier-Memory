"""
memory_system — Steps 1–5: Foundation + retrieval + pipeline + daemon + self-improvement.

Public surface:
    MemoryConfig, load_config        — configuration
    Database                         — SQLite wrapper + all CRUD ops
    MemoryFragment, Session, Entity, Triple, Correction  — domain models
    new_id                           — ULID generation
    VectorIndex                      — HNSW vector index (requires usearch/hnswlib)
    composite_relevance_score, tier  — CRS calculator + tier classification
    fused_retrieval, assemble_prompt — 3-signal RRF retrieval + prompt builder
    Embedder, RandomEmbedder,
    CachedEmbedder, OpenAIEmbedder   — embedding backends
    ConsolidationPipeline            — 4-stage compression pipeline
    TranscriptMessage                — input type for the pipeline
    MemoryAuditor, AuditReport       — self-improvement audit loop
    ModelUpgradeReindexJob,
    ReindexReport                    — vector re-embedding after model upgrade
"""

from .config import (
    MemoryConfig, DaemonConfig, StorageConfig, EmbeddingConfig,
    RetrievalConfig, CompressionConfig, EvictionConfig,
    CrossProjectConfig, SelfImprovementConfig, load_config,
)
from .models import (
    MemoryFragment, Session, Entity, Triple, Correction, RetrievalResult,
)
from .schema import Database
from .ids import new_id
from .vector_index import VectorIndex
from .scoring import composite_relevance_score, tier
from .retrieval import fused_retrieval, assemble_prompt
from .embedder import Embedder, RandomEmbedder, CachedEmbedder, OpenAIEmbedder
from .pipeline import ConsolidationPipeline, TranscriptMessage
from .auditor import MemoryAuditor, AuditReport
from .reindex import ModelUpgradeReindexJob, ReindexReport

__all__ = [
    "MemoryConfig", "DaemonConfig", "StorageConfig", "EmbeddingConfig",
    "RetrievalConfig", "CompressionConfig", "EvictionConfig",
    "CrossProjectConfig", "SelfImprovementConfig", "load_config",
    "MemoryFragment", "Session", "Entity", "Triple", "Correction",
    "RetrievalResult", "Database", "new_id",
    "VectorIndex",
    "composite_relevance_score", "tier",
    "fused_retrieval", "assemble_prompt",
    "Embedder", "RandomEmbedder", "CachedEmbedder", "OpenAIEmbedder",
    "ConsolidationPipeline", "TranscriptMessage",
    "MemoryAuditor", "AuditReport",
    "ModelUpgradeReindexJob", "ReindexReport",
]
