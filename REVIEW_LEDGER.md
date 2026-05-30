# Review Ledger — unproven bets to revisit, not re-flag

Watchlist, not guilt list. On each review, check whether the `Resolves when`
trigger has fired; flip status or update `Last checked`. Prune entries that go
many runs with no movement.

---

### N1 — `Score` as a float-impersonating rich object   (status: unproven)
- **First seen:** 2026-05-31, `models.py:208-245`
- **The bet:** `composite_relevance_score` returns a `Score` (value + per-signal
  components + multiplier) that impersonates a float via `__float__` and
  comparison dunders, instead of returning a bare float or a `(float, dict)`
  tuple. The upside: components travel with the score for the retrieval event log
  / learned reranker without changing every signature.
- **Why it might be worse:** an object that pretends to be a float must implement
  *every* numeric protocol or it crashes far from its definition — `__format__`
  was already missing (the one failing test). Each forgotten dunder is a latent
  `TypeError`.
- **Resolves when:** either (a) the learned-reranker (v4) actually consumes
  `Score.components` off live retrieval and the rich object has clearly paid for
  itself → confirmed-good; or (b) a second missing-dunder crash shows up in a
  live path (not a test) → confirmed-worse, replace with `.value` at call sites.
- **Last checked:** 2026-05-31 — `__format__`/`__round__` now added so the
  façade no longer crashes on format specs. This patched the symptom, not the
  bet: the open question (does the rich object earn its keep vs. returning
  `.value`?) is unchanged. Still watching for trigger (a) or (b).

### N2 — Speculative "digital subconscious" machinery vs. data volume   (status: unproven)
- **First seen:** 2026-05-31, `schema.py:125-176` (structural_patterns,
  pending_structural_patterns, simulations, epistemic_events) + `pipeline.py`
  stage 3b (`_wl_hash`) + `_resolve_simulations`.
- **The bet:** a large amount of architecture (Weisfeiler-Lehman structural
  fingerprinting, counterfactual REM simulations, epistemic-class multipliers,
  cross-project pattern quorum) is built ahead of the data that would exercise
  it. For a mostly single-user system this could be the "overengineered
  abstraction for a problem with one case" smell — or it could be exactly the
  long-horizon bet the blueprint intends. I cannot name a concrete failure, so
  it is **not** a finding — it's a watch.
- **Why it might be worse:** every one of these tables is a maintenance and
  correctness surface (migrations, JSON exemplars, dry-run branches) that earns
  its keep only if it's populated and read in real use.
- **Resolves when:** after real usage, check row counts: do
  `structural_patterns` (promoted), `simulations` (resolved!=expired), and
  `simulated_realized` fragments actually accumulate and get *retrieved*? If yes
  → confirmed-good, the bet paid off. If these tables sit near-empty or never
  surface in retrieval after sustained use → confirmed-worse (dead weight),
  consider deferring them behind a flag until there's signal.
- **Last checked:** 2026-05-31 — first sighting; no usage data yet.
