"""
Domain models for the agentic memory system.
All fields map directly to the SQLite schema in schema.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class MemoryFragment:
    id: str                          # ULID
    project_id: str
    scope: str                       # 'project' | 'global' | 'session'
    category: str                    # 'fact' | 'episode' | 'preference' | 'correction'
    content: str
    token_count: int
    abstraction_lvl: float = 0.5     # [0,1] how transferable across projects
    confidence: float = 0.8          # [0,1] self-assessed confidence
    source_type: Optional[str] = None  # 'distillation' | 'user_explicit' | 'reflection'
    source_session: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0
    is_pinned: bool = False
    is_deprecated: bool = False
    deprecated_by: Optional[str] = None
    graph_centrality: float = 0.0
    user_feedback: float = 0.0       # [-1,1]
    embedding_model: str = ""
    embedding_dim: int = 0
    metadata_json: Optional[str] = None

    # §2.1 — epistemic tracking (Gap #1 & #4)
    epistemic_class: str = "observed"        # observed|reflected|consolidated|simulated|simulated_realized|analogical
    mutation_velocity: float = 0.0           # EWMA contradiction rate; drives velocity-weighted decay
    last_verified_at: Optional[str] = None
    contradiction_count: int = 0
    corroboration_count: int = 0
    realized_from_sim_id: Optional[str] = None  # set when promoted from a simulation

    # Computed at retrieval time, not stored in DB
    crs: float = 0.0
    embedding: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class Session:
    id: str                          # ULID
    project_id: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    summary: Optional[str] = None
    turn_count: int = 0
    model_id: Optional[str] = None
    cost_input_tok: int = 0
    cost_output_tok: int = 0


@dataclass
class Entity:
    id: str                          # ULID
    project_id: str
    entity_type: str                 # 'file' | 'function' | 'class' | 'package' | 'config'
    qualified_name: str              # e.g., 'src/services/UserService.ts::createUser'
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    properties_json: Optional[str] = None


@dataclass
class Triple:
    id: str                          # ULID
    subject_id: str
    predicate: str                   # 'depends_on' | 'calls' | 'modifies' | 'caused_by'
    object_id: str
    project_id: str
    source_fragment: Optional[str] = None
    confidence: float = 0.8
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_validated: Optional[str] = None


@dataclass
class Correction:
    id: str                          # ULID
    original_fact: str
    corrected_fact: str
    project_id: Optional[str] = None  # None = global correction
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    applied_to: Optional[str] = None  # JSON array of fragment IDs updated


@dataclass
class RetrievalResult:
    fragments: list[MemoryFragment]
    project_summary: Optional[str]
    global_profile_hash: Optional[str]
    token_budget_used: int
    chaos_flag: bool = False  # §3.4 — True when top-K mean mutation_velocity > 85th percentile


@dataclass
class StructuralPattern:
    """§3.1 — Abstract structural pattern extracted from the triples graph."""
    id: str
    signature_hash: str              # WL-hash of (entity_type, predicate)-typed subgraph
    arity: int
    abstract_form: str               # LLM role-token description (empty until articulated)
    exemplars_json: str              # [{project_id, entity_ids, fragment_ids}, ...]
    embedding_dim: int
    failure_mode: Optional[str] = None
    occurrence_count: int = 1
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    embedding: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class PendingPattern:
    """§3.1 — Holds pattern candidates until quorum (3+ exemplars, 2+ projects)."""
    id: str
    signature_hash: str
    arity: int
    exemplars_json: str
    first_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Simulation:
    """§3.2 — REM-cycle counterfactual prediction."""
    id: str
    project_id: str
    perturbation: str
    hypothesis: str
    predicted_facts_json: str        # [{text, embedding_b64}, ...]
    prior: float
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    seed_entity_id: Optional[str] = None
    resolved_at: Optional[str] = None
    outcome: Optional[str] = None   # confirmed|rejected|partial|expired
    realization_evidence: Optional[str] = None


@dataclass
class EpistemicEvent:
    """§3.4 — Records confidence-affecting events for velocity computation."""
    fragment_id: str
    event_type: str                  # contradiction|corroboration|staleness|sim_realized|correction_applied
    delta_conf: float
    ts: str
    id: Optional[int] = None        # AUTOINCREMENT, None before insert
    evidence_frag: Optional[str] = None
