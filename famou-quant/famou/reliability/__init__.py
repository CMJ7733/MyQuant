"""Reliability-gated evolution layer for Famou (EvoQuant paper branch).

This package implements the reliability architecture on top of the core
evolution loop:

- ``types``: frozen protocol manifests, TaskSpec, budgets, evidence
  vectors, candidate packages, gate verdicts, decision records, transitions.
- ``budget``: atomic BudgetLedger.
- ``archives``: Search Archive (everything tried) vs Certified Archive
  (what we actually believe), plus CertifiedAdmission checks.
- ``barrier``: BarrierCommit — the single transactional writer.
- ``promotion``: PromotionPolicy (when to spend a sealed query) and
  SealedGateService (the only component allowed to touch sealed data).
- ``observation``: compressed AgentObservation built from archives + budget.
- ``judge``: FailureAnalyzer — raw errors into a stable failure taxonomy.
- ``experts``: proposal experts (GBDT / NN family) that turn a
  StructuredAction into CandidatePackages.
- ``trajectory``: Transition + TrajectoryStore for Agentic RL data.
- ``rl``: observation/action encoding, BC + offline-RL trainers, and the
  LearnedMetaPolicy that serves a trained checkpoint.
- ``experience``: evidence-grounded experience layer. Indexes what the search
  has learned as retrievable records that REFERENCE the trajectory and
  archives rather than copying them, and retrieves at a stated state version.
  Ships failure memory: aggregated (FailureKind, model_family) patterns with
  repair statistics, built with no LLM and no sealed data. Two retrievals per
  decision — a cheap summary into the observation, and a policy-sized query
  into the proposal phase whose token cost is charged back through the
  reward. See its module doc for invariants E1-E6.
- ``final_test``: FreezeManifest and the one-shot FinalTestService.

Design invariants (do not weaken without a protocol amendment):

1. The agent never sees sealed/final data or any numeric sealed metric.
   The gate returns only a discrete verdict + coarse margin band.
2. Failed candidates still produce Evidence — failure is a learning signal.
3. Search Archive membership implies nothing about trust; only a PROMOTE
   verdict through CertifiedAdmission enters the Certified Archive.
4. Sealed/final budgets are per-episode; the Certified Archive is
   experiment-global but every entry records its episode_id.
5. One sealed query per candidate per evidence generation. A candidate that
   has already been through the gate is not re-submitted on the same
   evidence (``SearchArchive.gate_attempts``), otherwise a REJECTed
   candidate keeps re-qualifying and drains the episode budget.
6. ICIR is a within-run time-series ratio reported by the harness. It is
   never reconstructed from cross-seed dispersion.
7. (C1) ``BarrierCommit`` is the ONLY writer of either archive. Rollout
   results are staged, and the whole batch is applied as one atomic update
   that bumps ``state_version``. ``CommitGuard`` enforces this: a write
   outside a commit window raises ``ArchiveWriteOutsideBarrier``. Because
   there is exactly one transactional writer, "live archive" and "last
   committed state" coincide, so ``ObservationBuilder`` reads the archives
   directly and no shadow snapshot exists to drift.
8. Candidates own the prediction model and its training configuration —
   nothing else. The portfolio builder, transaction costs, backtester and
   split protocol live in ``TaskSpec`` and are enforced by the F0 static
   check. An agent that can edit its own scorer will optimise the scorer.
9. ``final_test`` is one-shot and one-way. ``FinalTestService`` is the only
   reader, it refuses to run twice, and it holds no archive / trajectory /
   observation reference it could write a PaperResult back into. Freezing
   an episode closes its search: the strategy raises ``SearchClosed``.
10. The judge classifies and advises; it never admits. Only a sealed-gate
   PROMOTE routed through ``CertifiedAdmission`` changes what is trusted.
   ``FailureKind`` is part of the observation space — adding or renaming a
   member invalidates existing policy checkpoints, same as ENCODING_VERSION.
11. What the RL layer learns is the SEARCH policy (which expert, which
   family, what fidelity, how wide, whether to gate, how much experience to
   consult) — never the stock model itself. Candidates stay the experts' job.
12. The experience layer holds REFERENCES, never copies of events. Records
   point at transition/evidence/candidate ids; the Trajectory Store and
   Search Archive stay the single source of truth. It is written only from
   inside the barrier's window and read only at a named state version, so
   "what the agent could have known at version n" is reconstructible. See
   ``famou.reliability.experience`` for invariants E1-E6 and for the
   cross-episode gate-verdict channel that is deliberately still open.
13. An action records what the registry SERVED, not only what the policy
   asked for. Not every ``ExpertKind`` is implemented and not every declared
   family has an expert, so ``_pick_expert`` substitutes; ``resolved_expert``
   / ``resolved_family`` make that visible and ``ActionCodec`` encodes them.
   Encoding the request instead would put two labels on one behaviour, and
   an iterated offline loop would keep manufacturing such samples. A learned
   policy is additionally masked to the reachable head values — see
   ``ReliabilityAwareQuantStrategy.reachable_action_space``. This covers
   registry-level aliasing only: ``linear``/``temporal_transformer`` also map
   onto other trainers inside the candidate runtime, which is a separate,
   documented choice.

Batching is limited to single-island runs: ``_IslandTracker`` hands out
iteration slots round-robin across islands, while a batch belongs to one
island's context. Multi-island runs keep one rollout per decision.

Note for maintainers: the ThreadPool backend deep-copies each rollout, so
``_ReliabilityEvaluate`` overrides ``__deepcopy__`` to keep the evaluator
(and the BudgetLedger inside it) shared. Copying it would give every worker
a private ledger and silently stop enforcing budgets.
"""
