# TODO #1/#2 — frequency-weighted top-k: offline replay

Addresses `TODO.md` #1 (ranking ignores prefix frequency) and #2 (long
background prefixes crowd out short hot ones). Both are the same defect: the
admission score has one dimension, `coverage_tokens`.

This note reports the offline replay only. No gateway change is included, and
no GPU cell was run. The replay answers one question before any GPU time is
spent: **is there enough of a win to justify building it?**

## What the replay does

`experiments/icc_kv/replay_freq_topk.py` reads the per-request CSVs a formal
cell already wrote, replays the top-k admission decision under a candidate
ranking function, and reports how many of the observed routing false negatives
would have been avoided.

The count a worker would advertise is "how many times this prefix has been
served so far". That is the number a real sender can put in the K_UP payload,
which is 32 unused bytes today (`wire.py:frame` pads it with zeros and nothing
reads it), so carrying it costs zero bytes on the wire.

### Only the deterministic part is reported

A digest is **immune** when its score strictly exceeds every noise class in that
cell's mix. `_topk_drop_new` admits anything scoring above the worst retained
entry, and no arrival ever scores above the maximum, so an immune digest can
never be displaced. That is a proof, not a model.

Everything else is left in an **at risk** bucket and not counted. An earlier
draft fitted a displacement probability to the observed FN rates; the control
arm (the `coverage` rule the cells actually ran) then predicted 798 routing FN
against 26 observed. The fitted dynamics were wrong and were discarded. What is
reported here is therefore a **floor**, not an estimate.

## Result

15 cells (main_base, overlap_ov30, overlap_ov60 × 3 seeds), StaticTopK16,
1995 scored requests.

| scoring | immune | at risk | observed FN | real misses | rescued FN | rescued misses | prefill saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `coverage` (shipped rule) | 1197 | 798 | 26 | 16 | 0 | 0 | 0 |
| **`cov_x_n`** | **1719** | **276** | 26 | 16 | **15** | **8** | **12 651** |
| `cov_x_log2` | 1599 | 396 | 26 | 16 | 7 | 3 | 6297 |
| `cov_x_sqrt` | 1575 | 420 | 26 | 16 | 7 | 3 | 6297 |

`coverage` rescues nothing and has zero immune advertisements on ov30/ov60 —
correct, because it is the rule that produced those cells. The 1197 immune ads
in its row all come from `main_base`, whose noise is a single 256-token class
and whose FN is already 0.

`cov_x_n` moves 522 more advertisements into the provably-safe set and halves
the uncertain band. It recovers 8 of the 16 real misses, about 12.7k prefill
tokens. Both compressed variants recover less than half of that, which is
expected: a compressed count grows too slowly to clear a long noise class.

### Why linear wins

A useful prefix has to outscore the longest noise class to become immune, and
the required repeat count is set by the coverage ratio:

| useful coverage | noise 1024 | noise 2048 | noise 4096 |
|---|---:|---:|---:|
| 1024 | n ≥ 2 | n ≥ 3 | **n ≥ 5** |
| 2048 | n ≥ 1 | n ≥ 2 | **n ≥ 3** |
| 4096 | n ≥ 1 | n ≥ 1 | **n ≥ 2** |

Only `coverage × n` grows fast enough to cross that bar within a 600 s cell.
`coverage × log2(1+n)` needs n > 15 for a 1024-token prefix to clear 4096-token
noise, which a Zipf(1.2) pool does not deliver.

## What this does not show

- **No TTFT, no throughput, no end-to-end number.** Only how many FN the
  ranking change would move into the safe set.
- **The 10 rescued requests are not free.** Their prefill saving is the
  `prefill_tokens` those requests actually recorded; the real number after a
  routing change could differ because the request would land on a different
  worker.
- **Cold prefixes stay invisible.** A prefix's first advertisement carries
  n=1, exactly like unique noise, so it cannot be protected. On ov60 all 10
  observed FN of seed 0 were on the 2nd-or-later appearance of their slot, so
  this replay can act on all of them in principle — but a prefix that is only
  ever requested once is indistinguishable from noise and stays that way.
- **The ov30/ov60 noise mixes are synthetic.** The gain depends on the noise
  ceiling being finite and known. A workload whose noise includes longer
  prefixes than any useful prefix would defeat the ranking entirely.

## Verification

```bash
python -m experiments.icc_kv.test_replay_freq_topk     # 28 checks
```

The suite includes the control-arm check (the shipped rule must not claim a
rescue over itself), the per-slot counting check (one prefix recurring must not
raise another's score), and the `rescued_miss` check (an FN that still hit the
cache must not be credited as saved prefill).

## Reproduce

```bash
python -m experiments.icc_kv.replay_freq_topk \
    --summary analysis/icc_kv/formal_A_5d2d300/overlap_ov60/e2e_summary.json \
    --summary analysis/icc_kv/formal_A_5d2d300/overlap_ov30/e2e_summary.json \
    --summary analysis/icc_kv/formal_A_5d2d300/main_base/e2e_summary.json \
    --methods StaticTopK16 \
    --scoring coverage,cov_x_n,cov_x_log2,cov_x_sqrt
```

The `analysis/icc_kv/` tree is gitignored; point the paths at wherever the
formal runs live.

## Next step, if this is worth building

1. `POLICIES["StaticTopK16Freq"] = (7, 1, 1, 0, 2, 16)` — mode 7 is unused, so
   `CFG` needs no change.
2. Gateway reads `data[32:36]` on K_UP, scores `coverage * count`, keeps the
   existing tiebreak. Tombstones stay out of the ranking.
3. Sender puts its per-prefix served count in the payload. Absent field →
   n=0 → clamp to 1 → behaviour is exactly today's StaticTopK16, so old senders
   still work.
4. Replay `overlap_ov60` (largest effect) and `main_base` (regression guard)
   before any full matrix.
