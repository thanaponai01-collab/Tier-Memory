# Tier Memory — System Overview

A compounding, lab-grade **agent memory system**. It ingests raw session
transcripts, compresses them into durable knowledge, retrieves the right slice
back into a model's context under a token budget, and keeps the store healthy as
models get smarter. Local-first (Ollama by default), portable, and token-saving.

This document summarizes every pipeline and flow in the system. It is a map, not
a spec — file paths point at the real code.

---

## 1. The big picture

```
                    ┌──────────────────────────────────────────────┐
                    │              memoryd (daemon)                 │
   agent / CLI ───► │  READ path  : retrieve  (sync, fast)          │ ───► context
   transcripts ───► │  WRITE path : ingest    (enqueue → ACK → bg)  │
                    │  background : pipeline · auditor · reindex     │
                    └───────┬───────────────┬──────────────┬────────┘
                            │               │              │
                      SQLite (WAL)     HNSW vector     Embedder cache
                      schema.py        index           (Ollama)
                                       vector_index.py  embedder.py
```

Two flows dominate everything:

- **Write flow** — a transcript becomes compressed, deduplicated, graph-linked
  memory. (Pipeline, §3)
- **Read flow** — a query pulls back the most relevant fragments within a token
  budget. (Retrieval + scoring, §4)

Everything else (agent, orchestrator, auditor, mirror, upgrade) sits on top of
these two.

---

## 2. The daemon — the spine

`memory_system/daemon/server.py`

A single long-running process. It owns the DB connection, the HNSW index, the
shared embedding cache, the learned-weight store, and a background worker.

- **Read handler** (`OP_RETRIEVE`, `OP_SEARCH`, `OP_STATS`, …) — synchronous and
  fast: vector + graph + BM25 + ranking, answered inline.
- **Write handler** (`OP_INGEST`) — enqueues the transcript, **ACKs immediately**,
  and returns. The caller never blocks on compression.
- **Background worker** — drains the queue, runs the 4-stage pipeline, and
  periodically runs the auditor.
- Serves an HTTP API **and** a static web dashboard (`daemon/web/index.html`).

Clients reach it through `daemon/client.py` (`get_client()`), used by the agent,
orchestrator, CLI, and hook ingester.

---

## 3. Write flow — the compression pipeline

`memory_system/pipeline.py` · `ConsolidationPipeline.ingest()`

A raw transcript (`list[TranscriptMessage]`) is compressed in four stages, plus a
structural pass and an optional end-of-session reflection.

```
transcript
   │
   ▼  Stage 1  _stage1_segment
[episodes]                  split on adjacent-embedding cosine distance > 0.30
   │                        each episode carries an attention_weight
   ▼  Stage 2  _stage2_distill
[fragments]                 cheap LLM → {intent, outcome, decisions, files,
   │                        errors, summary, confidence, abstraction_level}
   │                        confidence boosted by attention weight
   ▼  Stage 3  _stage3_extract_entities
[entities + triples]        2 calls: entities first, then relations grounded
   │                        to those exact names. Predicates restricted to
   │                        {depends_on, calls, modifies, caused_by, resolved_by}
   ▼  Stage 3b _stage3b_structural
[pending patterns]          Weisfeiler-Lehman hash of each 2-hop subgraph
   │                        (no LLM). Repeated shapes get staged for articulation.
   ▼  Stage 4  _stage4_consolidate
durable memory              per new episode:
                              • resolve open simulations  (cos ≥ 0.82)
                              • dedup near-duplicates      (cos > 0.90 → deprecate)
                              • consolidation: 3+ similar episodes → semantic FACT
```

**Attention weighting** (`_attention_weight`) — corrections (3.0), error fixes
(2.5), explicit "remember" (5.0), long tool chains, repeated topics all raise an
episode's confidence; small talk is damped (0.1).

**Consolidation** is the "sleep" step: when ≥ `consolidation_threshold` similar
episodes exist (cos > 0.80), the model extracts a timeless *fact* and **halves
the access counts** of the source episodes so they decay out of the hot tier. The
event is logged for auditability.

**Producer provenance** (§5.6) — every fragment is stamped with the model and
logic generation (`producer_model`, `producer_version`) that produced it. A
distillation that *failed* and fell back to raw text is deliberately left
**unstamped**, so the upgrade detector knows it was never really judged.

**Reflection** (`reflect()`) — at session end, the cheap model is asked what it
learned (new facts, corrected assumptions, lessons, proactive suggestions); the
result becomes a single high-confidence fact fragment.

---

## 4. Read flow — fusion retrieval + scoring

`memory_system/retrieval.py` · `fused_retrieval()`
`memory_system/scoring.py` · `composite_relevance_score()`

```
query ─► embed
            │
   ┌────────┼─────────────┬───────────────┬─────────────────────┐
   ▼        ▼             ▼               ▼                     │
Signal 1  Signal 2     Signal 3        Signal 4 (gated)         │
vector    graph 2-hop  BM25 (FTS5)     structural lane          │
(HNSW)    from matched                 fires only when the      │
          entities                     vector lane is weak      │
   └────────┴─────────────┴───────────────┘                     │
            │  Reciprocal Rank Fusion (weighted 1/(K+rank))      │
            ▼                                                    │
       ranked candidate ids                                      │
            │  load fragments, compute CRS (carry vector cosine  │
            ▼  in as the semantic signal)                        │
       re-rank by CRS, drop below min_crs                        │
            │                                                    │
            ▼  greedy knapsack within token_budget               │
       selected fragments ──► semantic gate (global → project)   │
            │                                                    │
            ├─ correction injection (§3.6): a matching past      │
            │  correction is pinned at L1 with confidence 1.0    │
            ├─ chaos flag (§3.4): high mutation velocity → warn   │
            └─ log RetrievalEvent (§5.2)                          ◄┘
```

**CRS — Composite Relevance Score** weights six signals:

| signal | weight | meaning |
|---|---|---|
| semantic | 0.30 | query↔fragment cosine (carried from the vector lane) |
| recency | 0.25 | exp decay, ~10-day half-life |
| frequency | 0.10 | log-scaled access count |
| importance | 0.15 | graph centrality (PageRank) |
| confidence | 0.15 | producer's stated confidence |
| feedback | 0.05 | thumbs up/down |

An **epistemic multiplier** is applied *outside* the weighted sum so simulated
content can surface but never out-rank observed fact: `observed 1.0`,
`reflected 0.95`, `consolidated 0.90`, `simulated_realized 0.85`,
`analogical 0.80`, `simulated 0.40`. Pinned fragments always score 1.0.

CRS also maps to **storage tiers**: hot ≥ 0.60 (L1-eligible) · warm ≥ 0.15
(L2 retrieval) · cold < 0.15 (compressed cold storage).

In **v4**, the flat weights become per-project **learned** weights
(`v4_reranker.py`, threaded in explicitly — no process global).

---

## 5. Prompt assembly — the cache ladder

`retrieval.py` · `assemble_prompt()` and `agent.py` · `_apply_cache_breakpoint()`

Fragments are layered most-stable-first so Anthropic's prompt cache stays warm:

```
L0  system prompt + global profile        (most stable)
L1  project memory + chaos warning         (semi-stable)
L2  recalled memories, ordered by CRS       (dynamic but consistent)
L3  active files (git diff) + user message  (fully dynamic)
```

The agent stamps an **ephemeral `cache_control` breakpoint** on the last block
every call, so the whole stable prefix is served from cache on the next tool
round instead of re-billed. `mem status` reports the cache hit rate as a churn
sensor.

---

## 6. The agent turn

`memory_system/agent.py` · `AgentRunner.run()`

```
retrieve ─► assemble (L0–L3) ─► stream w/ tools (≤10 rounds) ─► reflect ─► ingest
```

1. **Retrieve** from the daemon for the query.
2. **Assemble** on the first turn (global profile + git diff injected); append on
   later REPL turns.
3. **Inference loop** — stream, run tools, accumulate token + cache usage.
4. **Reflect** (hidden, best-effort) into a structured lesson.
5. **Ingest** the transcript fire-and-forget back to the daemon, with real token
   usage attached.

---

## 7. The orchestrator (multi-agent)

`memory_system/orchestrator.py` · `Orchestrator.run()`

```
retrieve ─► plan ─► execute subagents (parallel + sequential) ─► synthesize ─► ingest lesson
```

Uses the **Claude Code CLI** (`claude -p --output-format json`) for every LLM
call — no API key, inherits the active session's model, and accrues real
token/cache usage from the JSON envelope. The planner decomposes a goal into ≤ 6
self-contained subagent tasks with a dependency DAG; ready tasks run in a thread
pool, dependents wait. Synthesis produces both an answer and one durable
**lesson**, which is written back to memory.

---

## 8. Self-improvement — the auditor

`memory_system/auditor.py` · `MemoryAuditor.audit()`

Runs periodically (default weekly) to keep the store healthy:

1. **Contradiction detection** — conflicting fact pairs → deprecate the weaker.
2. **Confidence decay** — old, unvalidated facts lose confidence.
3. **PageRank centrality** — recompute `graph_centrality` over the triple graph.
4. **Orphan pruning** — drop entities with no linked fragments.
5. **Graph consistency** — validate triple endpoints still exist.
6. **Pattern articulation / crystallization / simulations** — promote staged
   structural patterns, run and resolve simulations.
7. **Reranker learning** — refit per-project and global CRS weights (v4).

---

## 9. The flywheel — upgrade & reindex

`memory_system/upgrade.py` · `memory_system/reindex.py`

The load-bearing promise: **when you run a smarter model, your memory can level
up to match it.**

- **Detect** (`detect_upgrade`, read-only) — compares the `producer_model`
  stamped on fragments against the model configured now, using an explicit
  capability ranking (`ollama/qwen3 → … → claude-opus-4-8`). Null producers (pre-
  provenance) rank below everything real, so they're correctly seen as "behind."
- **Safe mix** — never auto-reprocesses. It returns *what* would be re-judged and
  *how much* it would cost; `mem upgrade` requires an explicit human yes.
- **Reindex** — re-embeds all fragments after an embedding-model change.

**Durability invariant:** a durable *Record* is kept separate from a disposable
*Judgment*; a fast layer must never destroy a slow layer's input. That is what
makes re-judging safe.

---

## 10. The intent mirror & goals

`memory_system/mirror.py` · goals table in `schema.py`

On-demand only — nothing runs on a timer. The mirror reflects the gap between
what the user **said** they were trying to do (the `goals` table) and what they
**actually did** (recent distilled episodes/facts). The reflection is synthesized
by the user's configured LLM (`medium` role, local Ollama by default) into plain
language, not raw rows. `mem propose` lets the mirror notice recurring intentions
and suggest them as goals; `mem goal add/list/done/confirm/dismiss` manage them.

---

## 11. Storage model

`memory_system/schema.py` — SQLite, WAL mode, raw parameterized SQL (no ORM).

Core tables:

- `memory_fragments` (+ `memory_fragments_fts` FTS5, trigger-synced) — the unit
  of memory: fact / episode / preference / correction, with confidence,
  abstraction level, provenance, embedding metadata, pin/deprecate flags.
- `sessions` — per-session summary, turn count, model, token cost.
- `goals` — stated intent for the mirror (§5.3).
- `entities` + `triples` — the typed knowledge graph.
- `corrections` — pinned L1 truth, injected on matching queries.
- `structural_patterns` + `pending_patterns` — WL-hashed subgraph shapes.
- `simulations` — predicted facts awaiting confirmation.
- `events` + `retrieval_events` — unified observability log.

Vectors live in the **HNSW index** (`vector_index.py`), not a DB column — which
is why retrieval carries the vector lane's cosine into CRS explicitly.

---

## 12. Surfaces & integrations

- **CLI** (`memory_system/cli.py`, `mem …`) — `status`, `search`, `pin`,
  `forget`, `feedback`, `audit`, `stats`, `profile`, `reindex`, `upgrade`,
  `export`/`import`, `orchestrate`, `run`, `goal`, `mirror`, `propose`,
  `obsidian`, `dashboard`, `savings`, `daemon start/stop`.
- **Web dashboard** (`daemon/web/index.html`) — served by the daemon.
- **Savings analytics** (`daemon/savings.py`) — weekly + all-time token savings.
- **Obsidian bridge** (`obsidian_bridge/`) — bidirectional vault sync: export DB
  to markdown, watch notes/corrections back into the daemon.
- **Hook ingest** (`hook_ingest.py`) — feeds Claude Code session transcripts into
  the pipeline automatically.
- **MCP server** (`mcp_server.py`) — exposes memory over the Model Context
  Protocol.

---

## 13. Configuration

`memory_system/config.py` · `load_config()`

Typed config groups: storage (`db_path`), embedding, retrieval (RRF weights,
gates, budgets), compression (models, thresholds), eviction, cross-project,
self-improvement, and LLM role routing (`LLMRolesConfig` → `cheap` / `medium` /
… mapped to local or hosted models). The real store lives at the configured
`storage.db_path`, not the `~/.agent` default.

---

### One-paragraph summary

A transcript enters the daemon's write path, gets segmented, distilled,
entity-linked, and consolidated into durable fragments stamped with the model
that judged them. A query enters the read path, fuses four retrieval signals via
RRF, re-ranks by a six-component relevance score with an epistemic safety
multiplier, packs the result into a token budget, and layers it L0→L3 to keep the
prompt cache warm. An auditor keeps the store honest over time, and an upgrade
detector lets the whole memory level up — safely, with a human in the loop — when
a smarter model arrives. That self-leveling loop is the flywheel.
