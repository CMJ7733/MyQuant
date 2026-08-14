"""Reliability-gated evolution layer for Famou (EvoQuant paper branch).

This package implements the reliability architecture on top of the core
evolution loop:

- ``types``: frozen protocol manifests, budgets, evidence vectors, gate
  verdicts, decision records, transitions.
- ``budget``: atomic BudgetLedger.
- ``archives``: Search Archive (everything tried) vs Certified Archive
  (what we actually believe), plus CertifiedAdmission checks.
- ``promotion``: PromotionPolicy (when to spend a sealed query) and
  SealedGateService (the only component allowed to touch sealed data).
- ``observation``: compressed AgentObservation built from archives + budget.
- ``experts``: proposal experts (GBDT / NN family) that turn a
  StructuredAction into candidate Programs.
- ``trajectory``: Transition + TrajectoryStore for Agentic RL data.

Design invariants (do not weaken without a protocol amendment):

1. The agent never sees sealed/final data or any numeric sealed metric.
   The gate returns only a discrete verdict + coarse margin band.
2. Failed candidates still produce Evidence — failure is a learning signal.
3. Search Archive membership implies nothing about trust; only a PROMOTE
   verdict through CertifiedAdmission enters the Certified Archive.
4. Sealed/final budgets are per-episode; the Certified Archive is
   experiment-global but every entry records its episode_id.
"""
