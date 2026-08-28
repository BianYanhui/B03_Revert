# b03_motivation — B03 Motivation Experiment

Counterfactual instrumentation of the B02 shared-link platform to answer:
*after B02 semantic reduction, how many forwarded updates are actually
decision-irrelevant, is update value skewed, and is value predictable before
transmission?*

Design: `EXPERIMENT_DESIGN.md` (frozen).  Results: `results/` (raw per cell,
aggregates, figures) and `MOTIVATION_REPORT.md` (verdict).

## Files

- `run_b03_motivation.py` — copy-with-instrumentation of B02's
  `run_live_shared_link_v3.py` (provenance in the docstring).  Read-only
  recorders: forwarded-update ledger (captured before the index mutates),
  per-request dispatch snapshots, send-time observable features.  Ports and
  docker resources are B03-scoped (`b03-net`, 9702/9703).
- `counterfactual.py` — OFFLINE World 0 / World 1 evaluation (never touches
  the live run): exact `choose()` replay on recorded snapshots with one
  reverted slot; live-writer rule for superseded updates; oracle values.
- `analyze_b03.py` — RQ1-RQ4 aggregates, Figures 1-5, sanity checks,
  pre-declared verdict logic (conditions A-E).
- `net_b03/` — B03-scoped copy of the B02 networking platform (see its
  README); plus `smoke_b03.sh` end-to-end Phase-1 orchestration.

## Run

```bash
# Phase 1 (smoke: cluster + platform + instrumentation sanity)
bash b03_motivation/smoke_b03.sh

# Phase 2+ (formal presets; requires the platform from smoke to be up)
cd /home/byh/B03/b03_motivation
/home/byh/B02/poc/.venv/bin/python run_b03_motivation.py \
    --preset core --tag b03core --kv-cache-tokens <KV> --repetitions 2
/home/byh/B02/poc/.venv/bin/python analyze_b03.py --run-dir results --tag b03core
```

Outputs: `results/raw/b03_updates_*.csv`, `results/raw/b03_requests_*.csv`,
`results/cells_*.csv`, `results/aggregates/*.csv`,
`results/figures/fig1..fig5_*.png`, `results/aggregates/report_numbers_*.json`
(including the pre-declared A–E conditions and the final verdict).

## Invariants (enforced)

1. The real run always executes the unmodified B02 policy path; B03 never
   suppresses or reorders updates.
2. The live dispatcher index is never mutated by instrumentation; World 0 is
   replayed offline from recorded snapshots on a reverted clone of exactly
   one slot.
3. Features used for prediction are strictly pre-transmission; oracle and
   realized columns are separated from deployable features.
4. All artifacts stay under `/home/byh/B03`; B02 files are never modified.
