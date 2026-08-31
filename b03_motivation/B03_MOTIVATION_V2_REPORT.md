# B03 Motivation V2 Report — Decision Value Retention under Signaling Budget

Experiment: `b03v2` — 65 live cells (5 ideal calibrations + 60 link cells),
10,400 requests, base B02 workload (α=0.55, pool 64, 3-step lineages),
n = 192 requests per cell (160 measured), **5 repetitions**, ρ ∈
{0.5, 0.8, 1.0, 1.2}, policies `agg_static` / `agg_full` (+ `exact_fifo`
as the no-reduction reference), real 4xT4 vLLM + real TCP/tc signaling
path, B03 counterfactual instrumentation (read-only; World-1 replay
reproduces 100% of recorded dispatch decisions in every cell).

Method: every gateway-forwarded update carries its counterfactual dispatch
value V(u) (next-use; horizon-4 for robustness) from the frozen records.
Selection policies rank the SAME per-cell candidate population by
pre-transmission features only; retention is computed per cell and
averaged with equal cell weight (bootstrap 95% CI, 2,000 draws).  Oracle
ranks by the true V(u).  Random averages 20 seeds.  Auxiliary
`LearnedLogistic` = logistic regression trained leave-one-rep-out on the
run's own labels (an optimistic bound for an offline-trained deployable
model).  No signaling delay is subtracted from value (V2 prompt §4).

---

## 1. Fate of every B02-forwarded update (Figure B)

Over all 7,294 forwarded updates of the 60 link cells:

| fate | share |
|---|---|
| A — no future use observed in the window | **70.6%** |
| B — future use, no decision flip | 15.4% |
| C — flip, low value (0 ≤ V ≤ 5 ms) | 0.9% |
| D — flip, meaningful value (V > 5 ms) | **12.9%** |
| E — flip, negative value | 0.2% |

(V1 run for comparison: A 79.7%, B 11.6%, C 2.8%, D 5.9%, E 0.1%; the A
share depends on the observation window, which is why V2 lengthened the
trace to n = 192.)

**Q1 — How much signaling has no actual decision value?**
85.9% of B02-forwarded updates produce no decision effect at all inside
the observed window (70.6% are never followed by any request to their
prefix; 15.4% are observed but change nothing).  Only **12.9%** flip a
decision with meaningful positive value.  Under congestion the "delivered
but never used" share grows (A rises with ρ) — exactly the updates a
budget-constrained sender should not have spent link time on.

## 2. Value Retention vs Signaling Budget (Figure A)

Retention = Σ max(V,0) over the selected top-β / Σ max(V,0) over all
candidates, averaged over 34–52 cells per point (agg_static + agg_full
population; exact_fifo excluded from the headline, shown in the CSVs).

| policy | @10% | @20% | @40% | @100% |
|---|---|---|---|---|
| **Oracle** | **0.266** | **0.505** | **0.855** | 1.000 |
| Freshness (newest first; ties = first-ads in delivery order) | 0.117 | **0.233** | **0.481** | 1.000 |
| Random (20 seeds) | 0.101 | 0.198 | 0.396 | 1.000 |
| FirstLook (simple first-advertisement rule) | 0.055 | 0.151 | 0.409 | 1.000 |
| SimpleCoverageLoad | 0.056 | 0.148 | 0.393 | 1.000 |
| CoverageAdvantage | 0.054 | 0.146 | 0.385 | 1.000 |
| B02Utility (relay τ=30, λ=16, tombstones-first) | 0.063 | 0.146 | 0.360 | 1.000 |
| CoverageDelta | 0.049 | 0.145 | 0.408 | 1.000 |
| DecisionAware (handcrafted, a-priori weights) | 0.050 | 0.137 | 0.377 | 1.000 |
| LearnedLogistic (LORO, auxiliary) | 0.049 | 0.095 | 0.306 | 1.000 |

Horizon-4 values confirm the ordering (Oracle 0.571, Freshness 0.242,
Random 0.197, all coverage-family policies 0.081–0.157 at β = 20%).

**Q2 — What can Oracle retain at 20%?**  **50.5%** of all positive
decision value (85.5% at 40%) — i.e., a perfect selector could halve the
signaling volume while keeping half the value.

**Q3 — Freshness?**  23.3% at 20% — the best deployable policy.  Honest
decomposition: 71% of evaluated updates are first advertisements of their
(instance, digest) pair (no source history → age undefined → tied at 0),
so "newest first" degenerates to "first advertisements first, in delivery
order".  Its strength is the first-advertisement signal, not age
comparison; among age-defined updates freshness ranks them last, which is
correct (re-advertisements are mostly worthless: flip rate 0.093 vs 0.635
for first advertisements).

**Q4 — Coverage?**  14.5–14.8% at 20% (CoverageAdvantage 0.146,
CoverageDelta 0.145, SimpleCoverageLoad 0.148) — **below Random
(0.198)**.  The failure is not an artifact: 37 of 52 cells rank
coverage-advantage below random; it holds at every ρ (Figure D) and in
both V1 and V2 data.  Mechanism: within a cell the largest
coverage-advantage updates are the biggest advertisements of hot lineages,
which are (i) superseded by later extensions before use, or (ii) flip the
decision only laterally between equally-covered replicas (net gain ≈ 0),
while the value concentrates in moderate-coverage first advertisements
that create NEW affinity — a non-monotone relation that a monotone
coverage ranking cannot express.

**Q5 — B02 utility?**  0.146 at 20% — indistinguishable from the other
coverage-based scores and below Random.  B02's frozen utility
(exp(−(age+Dq)/τ)·coverage − λ·64) is a freshness×coverage product; in
this workload its coverage factor inherits the same non-monotone
value relation, and its freshness factor ranks the valuable
first-advertisements LAST (they have no age → utility dominated by the
coverage term).  **B02's existing admission utility does not rank
updates by decision value.**

**Q6 — Coverage ↔ Oracle gap?**  **0.36 at β = 20%** (0.21 at 10%, 0.47
at 40%).  Against the best deployable policy (Freshness) the gap is still
0.27 at 20%.

## 3. Predictability of value (auxiliary; Figure 5-style check)

- Predicting whether an update flips a decision: LORO logistic AUROC
  **0.736** — the decision-irrelevant share is learnable.
- Predicting whether an update carries positive value (V > 1 ms): AUROC
  **0.618**; V > 5 ms: 0.589.  The marginal step from "will it flip" to
  "will the flip be worth it" is where the signal collapses: whether an
  update is live at its next use, and what the loads and tie-breaks are at
  that moment, is determined by FUTURE arrivals that no send-time feature
  observes.
- Consistently, every handcrafted score tested lands at or below Random —
  the learned linear model even below the handcrafted ones (it fits the
  pooled between-cell pattern, which inverts within cells).

## 4. Direct answers (V2 prompt §19)

1. **B02 signaling without decision value:** 85.9% of forwarded updates
   (A + B) change nothing within the window; 70.6% are never used at all.
2. **Oracle at 20%:** 50.5% of positive decision value (57.1% horizon-H4).
3. **Freshness at 20%:** 23.3% — best deployable, but only via the
   first-advertisement structure (see Q3 decomposition).
4. **Coverage at 20%:** 14.6% (advantage variant) — below Random.
5. **B02 utility at 20%:** 14.6% — no better than coverage.
6. **Coverage ↔ Oracle gap:** 0.36 at 20%; 0.47 at 40%; 0.21 at 10%.
7. **Is the gap worth pursuing?**  The *gap* is real and large — but the
   decisive question is Q8.
8. **Do the data support a learned model?**  **No, not yet.**  Every
   decision-aware signal tested (handcrafted scores, FirstLook, and a
   LORO-trained linear model) retains ≤ Random-level value; the
   value-prediction AUROC ceiling measured here is ≈ 0.62, against 0.736
   for mere flip-prediction.  The missing information is the future
   arrival/supersession/load trajectory, which send-time features do not
   carry.

## 5. Go / No-Go verdict

- **NO-GO for "Coverage ≈ Oracle"** — clearly false: the gap is 0.21–0.47
  across budgets.  There IS substantial decision value on the table.
- **NO-GO for a learned predictor now** — the second GO condition
  ("decision-aware features close the gap") failed: no tested
  decision-aware representation beats Random at retention, and the
  measured linear ceiling is AUROC 0.62 on value.  Building a
  Transformer/MLP on today's feature set would fit noise.
- **Verdict: RECONSIDER (conditional GO), with a precise reframe.**  The
  motivation for decision-aware signaling survives (huge gap, huge waste),
  but the *value* of an update is largely decided by what happens AFTER
  transmission (future requests to the prefix, later supersessions, load
  at use time).  Concretely, the next step this data supports is:
  1. **Simple structural rule first** — prioritize first advertisements
     (age-undefined, replica_visible = 0) and suppress re-advertisements
     and replica duplicates; this is what Freshness accidentally does
     (0.233) and it is already the best deployable policy; a
     retention-aware variant of B02's relay is a small, interpretable
     change worth measuring end-to-end.
  2. **Reframe the learned question** — predict the *arrival/supersession
     trajectory* (will this prefix be requested again before a newer
     update lands?) rather than static update value; that is where the
     missing information demonstrably lives.
  3. **Do not** train a static per-update value predictor on the current
     features: the data say it cannot beat Random.

## 6. Sanity checks (all PASS)

1. β = 100% ⇒ retention = 100% for every policy.
2. Oracle ≥ every policy at equal budget in every aggregate.
3. Random retention grows monotonically with β.
4. All policies rank the identical per-cell candidate population.
5. Counterfactual labels are frozen V1/V2 records; selection is a pure
   offline re-scoring (Check 5/6 of the V2 prompt — no live state, no
   dispatch, no signaling was altered).

## 7. Statistics disclosure

- Per-cell retention → equal-cell-weight mean; bootstrap 95% CI over cells
  (2,000 draws) in `value_retention_v2.csv` (`ci95_low/high`).
- Per-cell rows in `value_retention_by_cell_v2.csv`; per-rho curves in
  Figure D (retention@20% is stable across ρ: Freshness 0.22–0.27, Random
  0.19–0.20, coverage family 0.11–0.18, Oracle 0.45–0.54).
- Random = 20 seeds per cell before aggregation.
- 5 repetitions; 34–52 cells contribute per point (cells with < 5
  evaluated updates or zero positive value are excluded from curves and
  counted in the report numbers).

## 8. Threats to validity

- **First-order counterfactual:** V(u) is measured on the all-delivered
  timeline; suppressing a set of updates changes later state (supersession
  chains, load).  The budget simulation composes single-update values and
  ignores these interactions — standard for a motivation stage, but the
  true online retention of any subset may differ.
- **Observation window:** fate class A ("no future use") depends on trace
  length and drain horizon; longer traces shift A → B/D.  The A share is
  reported per run (70.6% at n = 192; 79.7% at n = 128 in V1) and both are
  given.
- **Freshness tie structure:** its lead depends on first-advertisements
  being 71% of the evaluable population and on stable-sort tie behavior;
  the FirstLook policy isolates the same signal explicitly and underperforms
  Freshness because its secondary coverage term reintroduces the
  big-advertisement trap — the decomposition is reported, not hidden.
- **LearnedLogistic is optimistic:** it trains on the same run's labels
  (other reps); a genuinely online model would do worse.  It is an upper
  bound for the linear-deployable class, and it still loses to Random.
- Equal-cell weighting avoids domination by exact_fifo-sized populations
  but gives thin high-ρ cells the same voice as rich ones; per-cell and
  per-rep rows are published for re-weighting.

## 9. Artifacts

- `results/aggregates/value_retention_v2.csv` (headline curves + CI)
- `results/aggregates/value_retention_by_cell_v2.csv` (per cell × policy × β)
- `results/aggregates/oracle_gap_v2.csv`
- `results/aggregates/update_fate_v2.csv` (+ per-update rows)
- `results/figures/figA..figD_v2.png`
- `results/aggregates/v2_report_numbers_v2.json`
- V1-tagged equivalents (`*_v1.csv/png`) for the cross-run comparison
