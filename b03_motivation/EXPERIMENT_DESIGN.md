# B03 Motivation Experiment — EXPERIMENT_DESIGN

Decision-Aware KV-State Signaling for Cache-Aware LLM Dispatch.

This document is the frozen design for the B03 motivation experiment.  It
answers: *is the B03 research question real and large enough to pursue?* —
before any predictor or suppression mechanism is designed.

---

## 1. Hypothesis

> **H:** Not every fresh and semantically valid KV-state update deserves
> signaling resources.  After B02's semantic reduction, a substantial
> fraction of forwarded updates never change a subsequent dispatch decision
> (decision-irrelevant), the flips that do occur are heterogeneous in
> realized value, and update value is concentrated in a small fraction of
> updates — yet value is predictable from pre-transmission features.

B02 answers *"is this update semantically worth keeping?"*; B03 asks *"is it
worth transmitting for the dispatch decision?"*.

## 2. System under study (unchanged B02 platform)

The experiment runs on the B02 hybrid live platform, copied into this
repository (code-only) and never modified semantically:

- 4 x vLLM 0.10.2 (Qwen2.5-1.5B-Instruct, one T4 per instance, ports 8000-8003,
  `--enable-prefix-caching --enable-prompt-tokens-details --max-num-seqs 8
  --enforce-eager`).
- Real kernel signaling path: instance agents → TCP → `b03-gateway` container
  → tc HTB on eth0 egress (shared link, rate = offered/rho) → TCP → harness
  dispatcher endpoint.  64-byte frames (104 B on-wire convention).
- Dispatcher affinity index updates ONLY on frame receipt; delivery delay is
  wall-clock cross-process on one host.
- Policies: `ideal` (bypass; calibration cell), `exact_fifo` (transmission
  reference), `agg_static` (merge + tombstone priority + replica cap 2 +
  gateway global top-K), `agg_full` (agg_static + adaptive utility gate).
- Workload: B02 lineage coverage-growth trace (3-step chains, 2048+512k
  tokens, Zipf popularity, disjoint phase shift at 50%, per-rep deterministic
  trace, cache_salt cell isolation, fixed 4-token outputs, shadow-model
  tombstones with real KV capacity from server logs).

The B03 runner (`run_b03_motivation.py`) is a copy-with-instrumentation of
B02's `run_live_shared_link_v3.py`.  The only functional additions are
read-only recorders (Section 4).  Ports/names are B03-scoped (9702/9703,
`b03-net`) so B02 experiments can run concurrently.

## 3. Counterfactual definition

For a gateway-forwarded update `u = (instance i, digest d, kind, coverage)`
and a later request `r` whose digest is `d`, with dispatcher state `S` at
r's dispatch time:

- **World 1** `D(S, r)` — the REAL recorded decision (native/affinity target
  with its coverage and expected net gain).  No re-execution: the actual
  dispatch of the actual run.
- **World 0** `D(S \ u, r)` — `choose()` replayed offline on the recorded
  snapshot (loads, rr counter, per-instance visible coverage of `d`) with
  EXACTLY ONE slot changed: `(i, d)` reverted to its pre-`u` value
  (`pre_delivery_visible_cov`; 0/absent for tombstones with no prior entry).

`decision_flip(u, r) = 1[D(S \ u, r) != D(S, r)]`, compared on the selected
target instance.  `decision_irrelevant = 1 - decision_flip`.

**Live-writer rule.**  A later write to the same `(i, d)` slot (superseding
upsert or tombstone) erases u's causal effect on `S`; such an update is
recorded with `superseded_before_use = 1` and World 0 ≡ World 1 (flip = 0).
This is semantically correct for a single-update suppression counterfactual:
in a world where u was never sent, the later update would still arrive and
produce the observed state.  The rule is implemented via per-slot writer
sequences captured at delivery and at each dispatch.

**Single-variable guarantee.**  Loads, rr tie-break counter, request bytes,
random seeds, vLLM cache state are recorded once and shared by both worlds;
the reverted slot is the only difference.  The replayed `choose()` is an
exact replica of B02's `Dispatcher.choose` (same j-filter, same tie-breaks
via the recorded rr, same guard rule), and the run-level sanity check
requires the World-1 replay to reproduce 100% of recorded decisions.

**No mutation.**  Instrumentation only appends records; the offline
evaluator never touches the live dispatcher.  Nothing is suppressed at run
time; the policy path is byte-identical to B02.

## 4. Recorded data (per candidate/forwarded update; Section VIII of the task)

Per forwarded update (`results/raw/b03_updates_*.csv`, one row per delivered
frame — the RQ1 population), captured BEFORE the dispatcher index mutates:

- identity: `point_id, cell_id, rep, policy, rho, update_kind, instance,
  digest, seq, writer_seq, cell_tag`
- state transition: `pre_delivery_visible_cov` (dispatcher-visible-before),
  `coverage_after` (dispatcher-visible-after-if-applied)
- timing: `t_send_unix, delivered_at_unix, signaling_delay_s`
- pre-transmission observable features (send time, joined by frame seq):
  `coverage_before_source, source_coverage_delta, update_age_s,
  advertised_before, dispatcher_visible_before, best_visible_cov,
  second_visible_cov, visible_cov_gap, replica_visible_count,
  source_best_cov, digest_req_count_recent (requests to the digest in the
  last 60 s), in_flight_frames (sent - delivered), ewma_delivery_delay_s
  (ack-fed, last 32), supersedes_in_flight, dispatcher_loads, dispatcher_rr,
  rho`
- oracle/realized columns (per future use; see Section 5).

Per request (`results/raw/b03_requests_*.csv`): the full B02 request record
(TTFT, `vllm_cached_tokens`, view-missing, state age, ...) PLUS the dispatch
snapshot `snapshot_rr, snapshot_loads, snapshot_cov_row, snapshot_wseq_row`
and the real decision `world1_*`.

Aggregates: `results/cells_*.csv` (B02 cell metrics, unchanged semantics) and
`results/aggregates/b03_update_counterfactuals_*.csv` (one row per update,
next-use headline) + `b03_update_horizons_*.csv` (per (update, use) rows).

## 5. Oracle value definitions

At each future use `r_k` (Analysis B horizon), with `a0 = World 0` and
`a1 = World 1` choices:

- `reusable_coverage_without / _with` — coverage exploitable relative to the
  least-loaded native instance in each world:
  `max(0, coverage(chosen) - coverage(native))` (0 if the world chose native).
- `estimated_prefill_gain_ms = (cov_with - cov_without) / prefill_tokens_per_ms`
  (**Oracle-A** scale: prefill saving).
- `estimated_queue_delta_ms = penalty(a1) - penalty(a0)` with the dispatcher's
  own `queue_penalty_ms` per unit load difference (**load penalty**).
- `estimated_net_gain_ms = prefill_gain - queue_delta`
  (**Oracle-B**: the dispatcher's own cost model; primary oracle).
- `signaling_cost_ms = measured one-hop delivery delay of u` (includes tc
  queueing at the cell's rho; the per-update byte cost is constant at 104 B
  and does not affect value ranking, so the time cost is the discriminating
  term).
- `oracle_value_ms = estimated_net_gain_ms - signaling_cost_ms` — value used
  for concentration (RQ3).
- Realized anchors: `realized_ttft_ms`, `realized_cached_tokens` of the
  actual (World-1) request.  Per-request causal TTFT deltas are NOT claimed;
  realized columns are used for aggregate cross-checks only (RQ2 context).

These are oracles: they may use future outcomes.  They are never used as
prediction features (Section 7).

## 6. Update–request relationship (Analysis A + B)

Updates are paired with uses by digest over the recorded request stream
(dispatch-time order), restricted to requests strictly after u's delivery:

- **Analysis A (next use)**: the first non-warmup request to the digest after
  u's delivery — the headline RQ1/RQ2 population.
- **Analysis B (horizon)**: first 4 and first 8 uses; per update:
  `impact_count_h{1,4,8}` (number of flips among measured uses) and
  `cumulative_value_h{1,4,8}` (sum of net gains over flips).  Headline
  conclusions must hold across horizons; disagreements are reported.

Updates with no future use of their digest are reported separately
(`evaluated = 0`) and excluded from flip-rate denominators (nothing to
flip); their share is itself a form of waste and is reported.

## 7. RQ4 predictability protocol

- **Labels** (next use): `decision_flip`; `positive_value`
  (`estimated_net_gain_ms > 1 ms`); `high_value_top25pct` (oracle value in
  the top quartile of its policy×rho cell).
- **Features** — only pre-transmission observable columns (Section 4);
  grouped into: `freshness_only` (update age, time since previous update),
  `coverage_only` (coverage, delta, best visible), `fresh_cov` (union),
  `decision_aware` (adds visible gap, replica count, recent digest demand,
  link in-flight, EWMA delay, supersedes-in-flight, rho, kind, load sum).
- **Model**: plain L2-regularized logistic regression (numpy, standardized
  features); no gradient boosting, no oracle features.  Missing feature
  values (e.g., age of a first update) are median-imputed.
- **Protocol**: leave-one-rep-out within each policy×rho group and pooled;
  report AUROC and AUPRC with the class prevalence as the AUPRC baseline and
  0.5 as the AUROC baseline.  Feature timestamps are verified to precede the
  labeled use (sanity check).

## 8. Parameter sweep (points)

Each point = one workload configuration (alpha, overlap, concurrency) × its
rho grid × policies; every point generates its own deterministic per-rep
trace and calibrates the offered signaling rate with its own ideal cell
(offered rate is a workload property).  Presets (`--preset`):

| preset | points | purpose |
|---|---|---|
| `smoke` | base, agg_full, rho {0.8, 1.3}, 1 rep, 48 requests | Phase 1: instrumentation sanity |
| `core` | base (α=0.55, ov=0, c=4), {exact_fifo, agg_static, agg_full} × rho {0.5, 0.8, 1.0, 1.2}, 2 reps | RQ1 gate + primary evidence |
| `alpha` | α {0.2, 0.55, 1.0} at rho 0.8, agg_static/agg_full, 2 reps | popularity skew |
| `overlap` | ov {0, 25, 50, 75}% at rho 0.8, agg_static/agg_full, 2 reps | replica overlap |
| `conc` | c {2, 4, 8} at rho 0.8, agg_static/agg_full, 2 reps | request concurrency |
| `full` | deduplicated union | final evidence run |

Repetitions: 2 minimum (paired reps share the same trace per point);
`exact_fifo` appears only in `core` as the transmission reference.  Within a
rep, the ideal calibration cell runs first; link cells are order-shuffled
with a recorded seed.

## 9. Metrics summary

- RQ1: `decision_flip_rate` / `decision_irrelevant_rate` at next use (and
  h4/h8), overall and by policy × rho; `superseded_before_use` share.
- RQ2: among flips — shares of strictly-positive / near-zero (|Δ| ≤ 1 ms) /
  negative `estimated_net_gain_ms`; distribution percentiles; mean reusable
  coverage gain.
- RQ3: Pareto curve of `oracle_value_ms`; top-10/20/30% share of total
  positive value; Gini.
- RQ4: AUROC/AUPRC per feature set and label (LORO), prevalence baseline.
- Platform (unchanged B02 semantics): TTFT mean/p50/p95, cached tokens,
  stale fallback, state age, view missing, delivery delays, tc backlog,
  relay drop counters.

## 10. Sanity checks (enforced)

1. **Instrumentation is inert**: by construction (recorders only append);
   verified additionally by (a) World-1 replay matching 100% of recorded
   decisions from recorded snapshots, (b) the ledger covering exactly the
   delivered frame count, (c) B02 cell metrics of instrumented runs compared
   against B02's published v3 runs at the same configuration.
2. **World 0 / World 1 single variable**: same recorded loads/rr/seed/cache;
   only slot (i, d) differs (Section 3).
3. **No future leakage**: feature columns are send-time only; a check
   asserts `send_ts_unix < use_dispatch_time_unix` for every evaluated use.
4. **Oracle vs deployable separation**: oracle/realized columns live in
   aggregate files; the RQ4 design matrix is restricted to `FEATURE_SETS`.
5. **Inherited B02 integrity**: byte-identical prompts and fixed output
   tokens across cells (prompt SHA-256), usage telemetry present, per-point
   trace SHA-256 recorded, cache_salt isolation per cell, error-free runs
   required (any request error invalidates the run, as in B02).

## 11. Threats to validity

- **Cost model**: `estimated_net_gain_ms` is the dispatcher's own cost model
  (prefill rate + queue penalty), not a measured TTFT delta per request;
  realized per-request TTFT is confounded by cache state and batching, so
  RQ2 claims heterogeneity of the *modeled* value with realized telemetry as
  aggregate context, not per-request causal TTFT effects.
- **Signaling cost**: measured delivery delay includes queueing shared with
  other updates; treating it as u's marginal cost is an upper bound on
  isolation but the qualitatively correct resource (link time at rho).
- **Shadow-model tombstones** inherit B02's documented approximations
  (per-prefix LRU vs per-block eviction); ground truth for reuse is always
  the physical `vllm_cached_tokens`.
- **Single host** clock for delays; per-rep traces are deterministic, but
  vLLM cache residue across cells is isolated only by cache_salt (inherited
  from B02; identical to the B02 experimental condition).
- **Horizon choice** (1/4/8) may change flip attribution; conclusions are
  required to hold at all three horizons.
- **Superseded updates** count as decision-irrelevant — this is the correct
  single-update counterfactual, but it means gateway merge/drop decisions
  upstream of the counted population also matter; the population is
  explicitly "B02-forwarded", which is the deployment-relevant quantity.

## 12. Success criteria (pre-declared interpretation)

Verdict conditions A–E (thresholds fixed before analysis; see
`CONDITION` in analyze_b03.py):

- **A**: pooled decision-irrelevant rate ≥ 0.50 and ≥ 0.30 in every cell.
- **B**: < 70% of flips strictly positive AND ≥ 20% near-zero/negative.
- **C**: top-20% of updates hold ≥ 50% of total positive value (pooled or
  best cell).
- **D**: |Spearman| of freshness and coverage-delta vs oracle value both
  < 0.30.
- **E**: decision-aware AUROC ≥ 0.65, ≥ freshness-only + 0.05, AUPRC > 1.5 ×
  prevalence, and beats freshness-only on the high-value label.

`STRONGLY SUPPORTED` = all five hold; `SUPPORTED` = ≥ 4; `WEAKLY SUPPORTED`
= 3; otherwise `NOT SUPPORTED`.  If A fails, B03 stops per the phase plan.

### Amendment 1 (pre-declared before the alpha/overlap/conc runs)

Condition A's min-cell guard applies only to cells with ≥ 10 evaluable
updates (`A_min_cell_updates = 10`): the base workload (96 measured requests
over 128 lineages at Zipf α = 0.55) yields a long-tailed number of future
uses per digest, and a 1-update cell (agg_full@ρ1.0 in the first core run)
statistically cannot represent a rate.  The pooled rate and the per-cell
rates remain unfiltered in `rq1_summary`; only the weakest-cell guard is
size-gated.  Additionally, because most updates are delivered late in a
cell (signaling delay 8–33 s at ρ ≥ 0.8 against a ~50 s request stream),
the RQ1/RQ2/RQ3 population is "forwarded updates with ≥ 1 future use of
their digest" — the deployment-relevant notion of a decision-relevant
update — and updates with no future use are reported separately.
