# TODO #1/#2 — frequency-weighted top-k admission

Closes the gateway half of `TODO.md` #1 and #2. The offline replay that
justified this is in `analysis/freq_topk_replay/REPLAY_REPORT.md` and was
merged as PR #3.

## The defect

`StaticTopK16` ranks pending updates by `coverage_tokens` alone. Two
consequences follow from that one choice:

- A long prefix seen once outranks a short prefix served constantly, even
  though the second saves more prefill per cell.
- Background noise whose coverage reaches the short classes evicts them
  outright. On `overlap_ov60` the 1024-token class carries a 8.2% routing FN
  while the 4096-token class carries 1.2%.

Equal coverage is broken by digest order, which is a hash, so two prefixes of
the same length compete by coin flip.

## The change

Rank by `coverage * frequency`, where frequency is how many times the sending
worker has already served that prefix in this cell.

### Where the number comes from

Only the worker knows its own hit count, so it puts the count in the frame.
Every frame already reserves 32 payload bytes and nothing ever read them on
`K_UP`, so the count rides in the first four of those bytes and **the wire
length stays 64**.

```python
FREQ = struct.Struct(">I")
FREQ_MAX = 0xFFFF
```

Why the gateway cannot count for itself: the background stream is 11.3M unique
digests per cell against 142 useful advertisements over 32 digests. A 64K LRU
counter table is swept 172 times per cell, so a useful prefix that recurs every
135 s always finds its counter reset. The signal has to come from the sender.

### Compatibility

A sender that leaves the field zero is ranked exactly as before:

- `frame(..., freq=0)` produces a byte-identical frame to the old `frame(...)`
- `read_freq` returns 0, and `_topk_score` reads 0 as frequency 1
- `_topk_score` only multiplies in `MODE_FREQ`; every other mode returns plain
  coverage, so a running cell cannot change behaviour by upgrading

Both properties are asserted in the tests, including an end-to-end check that a
whole admission sequence under `MODE_STATIC` and `MODE_FREQ` with no-claim
senders produces identical retained sets.

### Policy names

`StaticTopKFreq{4,8,16,32,64}`, encoded as mode 7. Mode 7 was unused, so the
`CFG` struct did not change. `StaticTopK16` and every other existing name is
untouched.

## Why multiplication and not a compressed count

A prefix only becomes un-evictable once it strictly outranks **every** noise
class present. The repeat count needed is set by the coverage ratio:

| useful coverage | noise 1024 | noise 2048 | noise 4096 |
|---|---:|---:|---:|
| 1024 | n ≥ 2 | n ≥ 3 | **n ≥ 5** |
| 2048 | n ≥ 1 | n ≥ 2 | **n ≥ 3** |
| 4096 | n ≥ 1 | n ≥ 1 | **n ≥ 2** |

`cov × log2(1+n)` needs n > 15 for a 1024-token prefix to clear a 4096-token
noise class, which a Zipf(1.2) pool never reaches inside a 600 s cell. The
offline replay measured the same ordering: `cov_x_n` recovered 8 of 16 real
misses, `cov_x_log2` and `cov_x_sqrt` recovered 3.

## What is deliberately not changed

- **Tombstones never enter the ranking.** Invalidation stays absolute and
  unweighted; a frequency term must not be able to delay a tombstone.
- **Merge, replica cap, priority lane, `Dispatcher.choose`** are untouched.
- **`topk_refs`** still counts live queued upserts per digest, and
  `_topk_add` still keeps the best score seen for a digest. A digest's stored
  score only rises while it has queued frames.

## Tests

```bash
python -m experiments.icc_kv.test_harness          # 60 checks
python experiments/4t4/test_policies.py --output /tmp/pc.csv   # 45 checks
```

New coverage:

| check | what it pins |
|---|---|
| `freq=0 is byte-identical to no freq` | an old sender's frames are unchanged |
| `read_freq on an old frame is 0` | absent field reads as "no claim" |
| `freq does not disturb the header` | the four bytes came out of the payload, not the header |
| `static ignores freq` / `bounded ignores freq` | only `MODE_FREQ` multiplies |
| `no-claim senders behave identically` | full admission sequence, both modes, same result |
| `coverage-only loses the short hot prefixes` | the defect reproduces in a unit test |
| `frequency keeps all four` | the fix recovers what the defect lost |
| `a frequency claim is inert outside MODE_FREQ` | gateway-level version of the same guarantee |

## Not verified

No cell has been run. `experiments/4t4/net/gateway_relay_4t4.py` is copied into
the gateway image at build time, so a real run needs `prepare_fixed_gateway`
to rebuild the container and the sha256 pin to be updated. That was
deliberately left alone — the T4s were in use.

The effect size is therefore still the offline replay's lower bound, not a
measured TTFT or FN rate.

## To run it

```bash
python -m experiments.icc_kv.full_queue --confirm --link-bit 10000000 \
    --capacity-file <CAP_FILE> --arrival-rate <AR> \
    --cell-seconds 600 --warmup-requests 10 --burst-mult 1.5 \
    --seeds 2 --blocks freq
```

The `freq` block pairs `StaticTopK16` against `StaticTopKFreq16` and
`StaticTopKFreq8` on `ov60` (where the two must differ) and on `base` (where
they must not).
