# Senior Review — Agentic Memory System

_Reviewed: 2026-05-31 · branch `master` · ~12k LOC Python · test suite: `python -m memory_system.test_smoke` (21/22 pass before review)_

This is an ambitious, genuinely interesting codebase: a tiered, self-improving
memory system with a clean persistence boundary and a real end-to-end test that
runs offline. The findings below are not about polish — they are the three or
four structural facts that will bite you when this moves from "works on my
machine with one user" to "runs continuously and is changed often."

I separated **violations** (I can name the concrete failure) from **deviations**
(just not how I'd do it). Only violations became findings. The deviations I
liked are in the Verdict and the ledger.

---

## [LENS 2] — The semantic component of CRS is inert at retrieval time

**File:** `memory_system/schema.py:935-960` (`_row_to_fragment`) →
`memory_system/retrieval.py:142-151` → `memory_system/scoring.py:88-93`
**Problem:** `_row_to_fragment` never populates `MemoryFragment.embedding`
(it isn't a DB column — vectors live in the HNSW index). But `fused_retrieval`
scores each candidate with `composite_relevance_score(frag, query_embedding)`
immediately after `db.get_fragment(fid)`, and scoring does:

```python
if query_embedding is not None and fragment.embedding:   # embedding is always None here
    semantic_sim = _cosine_similarity(...)
else:
    semantic_sim = 0.5
```

So the **largest CRS weight** (`W_SEMANTIC = 0.30`) multiplies a constant `0.5`
for every fragment. The CRS re-rank that is supposed to refine the RRF ordering
can't distinguish fragments on semantic grounds at all — it's effectively a
recency/frequency/confidence re-rank. Retrieval still returns sane results only
because Signal 1 (vector RRF) already did the real semantic ranking upstream;
the documented CRS behaviour silently does not execute.

**Why the tests don't catch it:** `test_crs_and_tiers` and `test_fused_retrieval`
build fragments with `embedding=emb.embed(...)` set **in-process**, so they never
round-trip through the DB load path where the field is dropped. `test_fused_retrieval`
only asserts `frag.crs > 0`, which is true from recency alone. This is false
safety — the seam between "fragment in memory" and "fragment loaded from DB" is
untested.

**Fix:** attach the embedding before scoring. Cheapest: have the vector index
return the stored vector alongside the id (it already holds them), and set
`frag.embedding` on the candidate before calling `composite_relevance_score`.
Then add a regression test that **inserts → closes → reloads** a fragment and
asserts a high-similarity query out-scores a low-similarity one. If that test
can't be written without the embedding, that proves the seam was missing.

**Open question (decides severity):** was CRS-semantic *meant* to be live at
retrieval, or is vector-RRF intended as the only semantic signal with CRS as a
pure recency/confidence re-rank? If the latter, delete `W_SEMANTIC` and the
`semantic` branch — don't keep a 0.30 weight on a dead input.
**Effort:** Small (the fix); the decision is yours.
**Triage: Blocker** — your headline ranking signal is running on a stub.

> **RESOLVED 2026-05-31** — chose to keep CRS-semantic live. The vector lane
> already computes the exact query↔fragment cosine, so rather than re-loading
> the stored vector (which Finding 5 flags as hot-path cost), `fused_retrieval`
> now threads that similarity into `composite_relevance_score(...,
> semantic_override=...)` (`retrieval.py`, `scoring.py`). Added regression
> `test_crs_semantic_roundtrip` (`test_smoke.py`): insert → close → reopen from
> disk → an exact-match query out-scores an unrelated fragment by Δ≈0.15 on
> otherwise-identical signals; pre-fix both collapse to the 0.5 fallback and tie.
> Suite 24/24. **Caveat:** graph/BM25-only candidates (no vector hit) still get
> the 0.5 neutral fallback, and the same dead-embedding bug still silently skips
> the `_passes_semantic_gate` similarity check for cross-project (`__global__`)
> fragments — that path needs the actual embedding attached, tracked separately.

---

## [LENS 1/2] — One SQLite connection shared across daemon threads

**File:** `memory_system/schema.py:253-289` (single `self._conn`,
`isolation_level=None`, manual `BEGIN`/`COMMIT`) ·
`memory_system/daemon/server.py:104-188` (HTTP-thread direct DB reads) vs
`:92-101` (loop-thread writes via `run_coroutine_threadsafe`)
**Problem:** The daemon opens one `sqlite3.Connection` with
`check_same_thread=False` and drives transactions by hand. Most API calls are
correctly funnelled onto the asyncio loop thread via
`asyncio.run_coroutine_threadsafe(self.daemon._handle_*(...), self.loop)` — but
several GET endpoints read the **same connection directly from the HTTP server
thread**:

```python
# do_GET runs on the HTTP worker thread, not the loop thread:
rows = self.daemon._db.fetchall("SELECT * FROM memory_fragments ...")   # :106
entities = ... self.daemon._db.fetchall("SELECT * FROM entities")        # :115
```

Meanwhile an ingest on the loop thread is inside a `with self.transaction():`
(`BEGIN ... COMMIT`) on that **same** connection. Two threads issuing statements
on one autocommit connection with a hand-managed transaction is a data race:
the concrete failures are `sqlite3.OperationalError: cannot start a transaction
within a transaction`, `Recursive use of cursors not allowed`, or a dashboard
read landing mid-write. `check_same_thread=False` silences the guardrail that
would normally stop this; it does not make it safe.

Note `mirror.py` does the right thing — it opens its **own** connection per call
(WAL makes cross-connection reads fine). The hazard is specifically the daemon's
shared connection used off-loop.

**Fix:** route **all** DB access through the loop thread (make the GET handlers
use `_handle_*` coroutines like the rest), or give the HTTP handler its own
read-only connection. The boundary rule: exactly one thread owns a connection.
**Verification:** a stress test that hammers `/api/fragments` while ingesting
concurrently should currently be able to trip an `OperationalError`.
**Open question:** is the daemon ever expected to serve concurrent requests, or
is real-world load effectively serial? If serial, this is latent, not active —
but it's a loaded gun in the boundary.
**Effort:** Small–Medium.
**Triage: Blocker** (downgrade to Strong if you confirm access is always serial).

---

## [LENS 5] — `Score` is a half-built numeric type (`__format__` missing)

**File:** `memory_system/models.py:208-245`
**Problem:** `Score` (§6.5) was introduced to replace a bare float return from
`composite_relevance_score` while staying float-compatible. It implements
`__float__` and the comparison operators — but **not** `__format__`, so any
f-string format spec blows up. This is the one test that fails today:

```
TypeError: unsupported format string passed to Score.__format__
# test_smoke.py:417 →  f"CRS (...): {crs_with_query:.4f}"
```

`f"{score:.2f}"` is the single most natural thing to write about a relevance
score, and it raises. Runtime blast radius is currently small only by luck:
`retrieval.py:149` casts `frag.crs = float(score)`, and `cli.py:146` /
`mcp_server.py:95` format a float pulled back out of JSON — so the live paths
dodge it. But `auditor.py:475` holds a raw `Score`, and the next person who logs
it with `:.2f` gets a crash far from here.

**Fix:** make the façade complete — add `__format__` (and likely `__round__`):

```python
def __format__(self, spec: str) -> str:
    return format(self.value, spec)
def __round__(self, n=None):
    return round(self.value, n)
```

**Deviation I'm flagging into a question (see ledger N1):** returning a rich
object that *pretends* to be a float is a maintenance tax — every numeric
protocol method you forget is a latent crash. A `(value: float, components:
dict)` return, or `score.value` at call sites, is harder to get subtly wrong.
Add `__format__` now; decide the bigger question via the ledger.
**Effort:** Small. **Triage: Strong suggestion** (the failing test makes it Blocker-adjacent).

> **RESOLVED 2026-05-31** — added `__format__` and `__round__` to `Score`
> (`models.py`). The failing `test_crs_and_tiers` now passes; suite is 23/23
> green. The broader "is the rich object worth it" question stays open as
> ledger entry N1.

---

## [LENS 1] — FTS external-content index is updated incorrectly on re-write

**File:** `memory_system/schema.py:53-55` (FTS decl) and `:346-351` (sync)
**Problem:** `memory_fragments_fts` is an **external-content** FTS5 table
(`content='memory_fragments'`). On every `upsert_fragment`, the code re-runs:

```python
INSERT INTO memory_fragments_fts(rowid, id, content, project_id) SELECT ... WHERE id = ?
```

On first insert this is fine. On the `ON CONFLICT DO UPDATE` path (content
changed) it inserts a **second** posting for the same rowid without first issuing
the FTS5 `'delete'` command. External-content tables require the delete-then-insert
dance (normally wired via triggers) or the index accumulates stale postings and
can report `database disk image is malformed` on `optimize`/`merge`. Today you're
mostly insert-once so it rarely fires — but it's hand-maintained correctness at a
single call site, which is exactly the kind of invariant that rots.

**Fix:** use the standard trigger pattern so sync is automatic and update-safe:

```sql
CREATE TRIGGER frag_ai AFTER INSERT ON memory_fragments BEGIN
  INSERT INTO memory_fragments_fts(rowid,id,content,project_id)
  VALUES (new.rowid,new.id,new.content,new.project_id); END;
CREATE TRIGGER frag_ad AFTER DELETE ON memory_fragments BEGIN
  INSERT INTO memory_fragments_fts(memory_fragments_fts,rowid,id,content,project_id)
  VALUES('delete',old.rowid,old.id,old.content,old.project_id); END;
CREATE TRIGGER frag_au AFTER UPDATE ON memory_fragments BEGIN
  INSERT INTO memory_fragments_fts(memory_fragments_fts,rowid,...) VALUES('delete',old....);
  INSERT INTO memory_fragments_fts(rowid,...) VALUES (new....); END;
```

Then drop the manual FTS insert from `upsert_fragment`. **Effort:** Small.
**Triage: Strong suggestion.**

---

## [LENS 2/4] — N+1 fan-out and per-query re-embedding on the retrieval hot path

**File:** `memory_system/retrieval.py:142-151` (per-candidate `get_fragment`),
`:90-98` (per-neighbor `fragments_linked_to_entity`), `:304-306` (`_inject_corrections`)
**Problem:** `fused_retrieval` resolves the fused candidate set by calling
`db.get_fragment(fid)` **one row per candidate** in a loop (30+ vector hits +
graph + BM25). Graph expansion calls `fragments_linked_to_entity` once per
neighbour. Worst of all, `_inject_corrections` calls `embedder.embed(...)` for up
to **50 corrections on every single retrieval** — that's up to 50 network/model
round-trips on the user-facing hot path. The structure couples "rank ids" to
"hydrate rows" to "embed corrections" with no batching seam.

**Fix:**
- Batch the hydrate: `WHERE id IN (?,?,...)` once, build a `dict[id]->fragment`.
  (`list_fragment_ids` already proves you know this trick — apply it here.)
- Cache correction embeddings (they change rarely): store the embedding with the
  correction, or memoise by `original_fact` hash. Don't re-embed per query.

**Effort:** Small–Medium. **Triage: Strong suggestion** (becomes Blocker once a
project has thousands of fragments or corrections + a remote embedder).

---

## [LENS 1/5] — Two LLM-calling conventions; the router migration stalled

**File:** `memory_system/llm.py:154-196` (`LLMRouter`) vs
`memory_system/pipeline.py:185,313,406,466,737` (module-level `call_model_json`)
**Problem:** `LLMRouter` (§6.2) exists precisely to stop hard-coding a model
string per pipeline stage — pick a *role* (`cheap`/`medium`/`strong`) and let
config resolve the model in one place. But only `mirror.py` uses it; the whole
consolidation pipeline still calls `call_model_json(..., model=self.cfg.distillation_model)`
with per-stage model strings. So there are two ways to answer "which model runs
this?", and the router's entire benefit (swap a model in one place) is unrealized
while the busiest caller bypasses it. New code has to guess which convention to
follow.
**Fix:** either route the pipeline through `LLMRouter` (map distillation→cheap,
consolidation→medium, etc.) and delete the per-stage model config, or delete the
router and standardise on the module functions. Pick one. **Effort:** Medium.
**Triage: Strong suggestion.**

---

## [LENS 4/5] — Daemon HTTP dispatch is a copy-paste `if/elif` ladder

**File:** `memory_system/daemon/server.py:83-346` (`do_GET` / `do_POST`)
**Problem:** Every endpoint repeats the same ~12-line envelope:
`run_coroutine_threadsafe(self.daemon._handle_x({...}), self.loop)` → `.result()`
→ `try/except → send_json(500, ...)`. Adding an endpoint means cloning the
boilerplate; changing the error contract means editing ~10 identical blocks.
This is also what produced the Finding-2 inconsistency (some branches went
straight to `_db` because the per-branch copy made it easy to skip the funnel).
**Fix:** a route table maps `(method, path) -> (coro_name, arg_extractor)`, and a
single dispatcher applies the run-on-loop + error envelope once. New endpoint =
one table row. **Effort:** Medium. **Triage: Strong suggestion.**

---

## [LENS 3] — Feedback-loop friction (grouped — one finding, not five)

- **No `requirements.txt` / `pyproject.toml`.** Setup is prose in
  `START_HERE.md` ("run `pip install numpy anthropic ...`"). One `pyproject.toml`
  with pinned deps + an entry point makes the environment reproducible and the
  `mem` command installable. **Small.**
- **Hand-rolled test framework.** `test_smoke.py` reinvents pytest: a `TestResult`
  accumulator, a `@test` decorator, manual pass/fail bookkeeping (1245 lines).
  It runs offline and tests real logic — genuinely good — but you're maintaining
  a test runner instead of using `pytest` (fixtures for the temp DB, `assert`
  rewriting, `-k`, parametrize, CI-friendly exit codes). Port the bodies; keep
  the offline RandomEmbedder/mock-LLM design. **Medium.**
- These two together are why a load-path bug (Finding 1) and a `__format__`
  crash (Finding 3) could sit unnoticed: in-process construction sidesteps the
  DB seam, and a custom runner swallows a `TypeError` into one red line among 22.
  **Triage: Strong suggestion (compounding).**

---

## Questions for you

1. **CRS semantic (F1):** intended to be live at retrieval, or is vector-RRF the
   only semantic signal and CRS a recency/confidence re-rank? Your answer decides
   whether F1 is "fix the load path" or "delete the dead weight."
2. **Daemon concurrency (F2):** is the daemon ever meant to serve concurrent
   requests, or is real load effectively serial? Decides Blocker vs latent.
3. **`Score` object (F3):** full float-replacement (then it needs every numeric
   dunder) or boundary-only (then call sites should use `.value`)? See ledger N1.

---

## Verdict

**Overall:** Solid Foundation — with two load-bearing bugs to clear first.

**The one thing to fix first:** the CRS semantic signal is computed against a
constant `0.5` because fragment embeddings are never loaded on the retrieval
path (Finding 1) — fix the load seam and add the round-trip test that would have
caught it.

**What is already good:**
- **The persistence boundary is genuinely strong.** All SQL lives in
  `schema.py`. Apply the skill's own test — "swap the DB, how many files touch?"
  — and the answer is *one*. That is rare in a junior codebase and it's the
  reason the rest of the system is testable at all. Don't let anyone talk you
  into scattering queries back into call sites because the file got big.
- The offline, no-API-key, no-network smoke suite (RandomEmbedder + mocked LLM)
  is exactly the right instinct for a system this entangled with external models.
- Defensive `try/except … pass` around *non-essential* side effects (retrieval
  event logging, correction injection) so observability can never break the user
  path — that's mature judgment.

**Novel approaches worth keeping:**
- The `_passes_semantic_gate` design — cross-project (`__global__`) fragments must
  clear an abstraction + similarity bar before leaking into a project's context —
  is a clean, well-placed policy seam. Quietly good.
- Routing LLM calls by *role* instead of model string (`LLMRouter`) is the right
  abstraction; the only problem is that you stopped halfway (Finding 6).

**Open questions:** the three above — the review isn't finished until they're
answered, because two of them change a finding's severity.

**Estimated session to unblock:** ~2–3 focused hours. Finding 1 (attach embedding
+ regression test) and Finding 3 (`__format__`) are ~30 min combined and clear
the red test. Finding 2 (funnel all DB access onto the loop thread) is the bulk.
After that the suite is green and the structural items (router, route table, FTS
triggers) are safe cleanup, not emergencies.
