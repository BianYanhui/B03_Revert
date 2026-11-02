"""Offline replay of frequency-weighted top-k admission over existing cell traces.

No Docker, no vLLM, no gateway. Reads the per-request CSVs that a formal cell
already wrote and re-scores the top-k admission decision with a candidate
ranking function, so the size of the win is known before any GPU time is
spent.

This is a model of the gateway's admission, not the gateway. It reproduces the
one decision that matters for TODO #1/#2 -- "would this useful digest have
survived the top-k set long enough to be forwarded" -- from the noise mix of
the cell and the queue residence time that cell actually measured. It does not
reproduce frame-level timing, and it cannot show an effect that the original
run's noise stream did not contain.

The two outputs are reported side by side on purpose:

  routing FN    the formal metric, "some worker holds the prefix and the
                dispatcher view does not"
  real miss     of those FN requests, the ones whose cached_tokens really came
                back below the 512-token bar

On ov60 StaticTopK16, 6 of 20 routing FN requests still hit the cache. A replay
that quotes FN alone overstates the benefit.

Usage:
    python -m experiments.icc_kv.replay_freq_topk \\
        --summary analysis/icc_kv/formal_A_5d2d300/overlap_ov60/e2e_summary.json \\
        --methods StaticTopK16,StaticTopK8 \\
        --scoring coverage,cov_x_n,cov_x_log2,cov_x_sqrt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

# Admission bar used by the formal matrix for a "real" reuse. Below this the
# request is a full prefill for our purposes.
CACHE_BAR = 512

# Top-k width of the method being replayed. Kept as a module constant because
# the replay compares scoring functions at one fixed width, not across k.
TOP_K = 16


def parse_mix(text: str) -> list[int]:
    """Expand a coverage mix into the flat table the noise sender walks.

    Same gcd-reduced round-robin as experiments.icc_kv.workload.interleave, so
    the composition here matches the offered stream.
    """
    items: list[tuple[int, int]] = []
    for raw in (text or "").split(","):
        piece = raw.strip()
        if not piece:
            continue
        coverage_s, weight_s = piece.split(":")
        items.append((int(coverage_s), int(weight_s)))
    if not items:
        return [256]
    divisor = 0
    for _coverage, weight in items:
        divisor = math.gcd(divisor, weight)
    counts = [weight // divisor for _coverage, weight in items]
    table: list[int] = []
    while any(counts):
        for index, left in enumerate(counts):
            if left:
                table.append(items[index][0])
                counts[index] -= 1
    return table


def survival_probability(
    score: float,
    noise_table: list[int],
    scoring: str,
) -> float | None:
    """Whether a digest keeping `score` can be displaced by background traffic.

    Returns 1.0 when no arriving noise frame can ever score higher, 0.0 when
    every noise class scores higher, and None when the outcome depends on queue
    dynamics that this offline replay cannot reconstruct.

    Only the two deterministic ends are reported. An earlier draft fitted a
    displacement probability to the observed FN rates; the control arm (the
    `coverage` rule the cells actually ran) predicted 798 routing FN against 26
    observed, so the fitted dynamics were discarded. What remains is a lower
    bound: a digest whose score strictly exceeds every noise class cannot be
    evicted, because `_topk_drop_new` admits anything scoring above the worst
    retained entry and no arrival ever scores above it.
    """
    if not noise_table:
        return 1.0
    noise_scores = [score_of(scoring, coverage, 1) for coverage in noise_table]
    highest = max(noise_scores)
    if score > highest:
        return 1.0
    if score <= min(noise_scores):
        return 0.0
    return None


def score_of(name: str, coverage: int, count: int) -> float:
    """Candidate ranking functions. `count` is 1 for a first advertisement."""
    if name == "coverage":
        return float(coverage)
    if name == "cov_x_n":
        return float(coverage) * count
    if name == "cov_x_log2":
        return float(coverage) * math.log2(1 + count)
    if name == "cov_x_sqrt":
        return float(coverage) * math.sqrt(count)
    raise SystemExit(f"unknown scoring function: {name}")


def replay_cell(
    summary_path: Path,
    method: str,
    scoring: list[str],
) -> list[dict]:
    """One (cell, method) replayed under every scoring function."""
    document = json.loads(summary_path.read_text())
    rows = [row for row in document["rows"] if row["method"] == method]
    out: list[dict] = []
    for row in rows:
        if row["failed_requests"]:
            continue
        forwarded = float(row.get("forwarded_per_s") or 0)
        offered = float(row.get("offered_noise_per_s") or 0)
        maxq = int(row.get("relay_max_queue") or 0)
        if forwarded <= 0 or offered <= 0:
            continue
        noise_table = parse_mix(row.get("noise_coverage_mix", ""))
        request_csv = summary_path.parent / (
            f"requests_{row['scenario']}_{method}_seed{row['seed']}.csv"
        )
        if not request_csv.exists():
            continue
        requests = [
            item
            for item in csv.DictReader(request_csv.open())
            if item["warmup"] == "0"
        ]
        for name in scoring:
            out.append(
                replay_requests(
                    requests, name, noise_table,
                    scenario=row["scenario"], seed=row["seed"], method=method,
                    maxq=maxq,
                )
            )
    return out


def document_row(document: dict, method: str) -> dict:
    for row in document["rows"]:
        if row["method"] == method:
            return row
    raise SystemExit(f"method {method} not in {document}")


def replay_requests(
    requests: list[dict],
    scoring: str,
    noise_table: list[int],
    *,
    scenario: str,
    seed: int,
    method: str,
    maxq: int,
) -> dict:
    """Walk the requests in arrival order, tracking each slot's ad count.

    The count is what a real worker would put in the K_UP payload: how many
    times it has served (and therefore advertised) this prefix so far. A first
    advertisement carries 1, which is exactly what a unique noise frame carries,
    so a cold prefix is indistinguishable from noise -- that is a property of the
    design, not a limitation of this model.
    """
    counts: dict[int, int] = {}
    immune = 0
    at_risk = 0
    rescued_fn = 0
    rescued_miss = 0
    rescued_prefill = 0
    original_fn = 0
    original_miss = 0
    original_prefill = 0
    for item in requests:
        slot = int(item["slot"])
        coverage = int(item["coverage_tokens"])
        counts[slot] = counts.get(slot, 0) + 1
        score = score_of(scoring, coverage, counts[slot])
        state = survival_probability(score, noise_table, scoring)
        immune += int(state == 1.0)
        at_risk += int(state is None)
        fn = item["loose_false_negative"] == "1"
        reused = int(item["reused_tokens"])
        prefill = int(item["prefill_tokens"])
        if fn:
            original_fn += 1
            original_prefill += prefill
            if reused < CACHE_BAR:
                original_miss += 1
            # Only count a rescue when this scoring function makes the digest
            # provably un-evictable. Anything in the uncertain band is left out,
            # so the number is a floor.
            if state == 1.0:
                rescued_fn += 1
                if reused < CACHE_BAR:
                    rescued_miss += 1
                    rescued_prefill += prefill
    return {
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "scoring": scoring,
        "requests": len(requests),
        "digests": len(counts),
        "max_queue": maxq,
        "immune": immune,
        "at_risk": at_risk,
        "original_fn": original_fn,
        "original_miss": original_miss,
        "original_fn_prefill": original_prefill,
        "rescued_fn": rescued_fn,
        "rescued_miss": rescued_miss,
        "rescued_fn_prefill": rescued_prefill,
    }


def report(rows: list[dict]) -> None:
    """Aggregate by scoring function across every cell that was replayed."""
    if not rows:
        raise SystemExit("nothing replayed; check the --summary path and method names")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["scoring"], []).append(row)
    print(f"{'scoring':14s} {'cells':>6s} {'reqs':>6s} {'immune':>7s} {'at risk':>8s} "
          f"{'obs FN':>7s} {'real miss':>10s} {'rescued':>8s} {'resc miss':>10s} "
          f"{'prefill saved':>14s}")
    for name, items in groups.items():
        print(
            f"{name:14s} {len(items):>6d} {sum(r['requests'] for r in items):>6d} "
            f"{sum(r['immune'] for r in items):>7d} "
            f"{sum(r['at_risk'] for r in items):>8d} "
            f"{sum(r['original_fn'] for r in items):>7d} "
            f"{sum(r['original_miss'] for r in items):>10d} "
            f"{sum(r['rescued_fn'] for r in items):>8d} "
            f"{sum(r['rescued_miss'] for r in items):>10d} "
            f"{sum(r['rescued_fn_prefill'] for r in items):>14d}"
        )
    print()
    print("`immune`  = requests whose digest scores strictly above every noise class,")
    print("            so `_topk_drop_new` can never displace it.")
    print("`at risk` = requests in the band where the outcome depends on queue")
    print("            dynamics this replay does not reconstruct. Not counted.")
    print("`rescued` = observed FN that this scoring function places in the immune set.")
    print("            This is a floor: the uncertain band is excluded, not estimated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True,
                        help="e2e_summary.json; repeat for several cells")
    parser.add_argument("--methods", required=True,
                        help="comma-separated methods present in those summaries")
    parser.add_argument("--scoring", default="coverage,cov_x_n",
                        help="comma-separated ranking functions")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    methods = [item for item in args.methods.split(",") if item]
    scoring = [item for item in args.scoring.split(",") if item]
    rows: list[dict] = []
    for summary in args.summary:
        for method in methods:
            rows.extend(replay_cell(summary, method, scoring))
    report(rows)
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
        print()
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
