#!/usr/bin/env python3
"""Offline counterfactual evaluation for the B03 motivation experiment.

This module NEVER touches the live platform.  It consumes the read-only
records produced by run_b03_motivation.py:

  - b03_requests_{cell}.csv   per-request dispatch snapshots:
        snapshot_rr          dispatcher round-robin counter before choose()
        snapshot_loads       dispatcher loads before choose() ("a;b;c;d")
        snapshot_cov_row     visible coverage of the digest per instance
        snapshot_wseq_row    writer sequence of the digest slot per instance
        world1_*             the REAL decision (World 1 anchor)
  - b03_updates_{cell}.csv    one row per gateway-FORWARDED update:
        pre_delivery_visible_cov, writer_seq, signaling_delay_s, send features

Counterfactual semantics (single-update intervention; see
EXPERIMENT_DESIGN.md "Counterfactual definition"):

  World 1  D(S, r)      = the real recorded decision at request r.
  World 0  D(S \\ u, r)  = choose() replayed on the SAME recorded snapshot
                           with only slot (u.instance, u.digest) reverted to
                           its pre-u value.  All other inputs (loads, rr,
                           other slots, request, seeds) are identical by
                           construction, so the only possible difference is
                           the candidate update u.

An update u is "live" at request r iff no later write to the same
(instance, digest) slot happened between u's delivery and r's dispatch
(writer sequence unchanged).  A superseded/overwritten update has no
remaining causal effect on the dispatcher state, so its World 0 equals its
World 1 by definition (recorded as superseded_before_use=1, flip=0).

Dispatcher.choose is replicated EXACTLY (same tie-breaks via the recorded
rr counter, same j-filter, same guard rule) and a self-test
(`--selftest`) verifies that replaying World 1 from the snapshots
reproduces the real recorded decision for every request of every cell.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChooseResult:
    target: int
    raw_fanout: int
    evaluated: int
    affinity: bool
    coverage: int
    expected_net_ms: float


def least_loaded_cf(loads: list[int], candidates: list[int], rr: int) -> tuple[int, int]:
    """Dispatcher._least_loaded on frozen inputs; returns (target, next_rr)."""
    minimum = min(loads[index] for index in candidates)
    ties = [index for index in candidates if loads[index] == minimum]
    return ties[rr % len(ties)], rr + 1


def cf_choose(j: int, prefill_tokens_per_ms: float, queue_penalty_ms: float, guard_ms: float,
              digest: str, cov_row: list[int], loads: list[int], rr0: int) -> ChooseResult:
    """Exact replica of B02 Dispatcher.choose on a frozen state snapshot.

    `cov_row[i]` is the visible coverage of `digest` on instance i; reverting
    one slot of cov_row is the ONLY way World 0 differs from World 1.
    """
    instances = range(len(cov_row))
    native, rr = least_loaded_cf(loads, list(instances), rr0)
    native_coverage = cov_row[native]
    candidates = [(index, coverage) for index, coverage in enumerate(cov_row) if coverage > 0]
    candidates.sort(key=lambda item: (-item[1], loads[item[0]], item[0]))
    raw = len(candidates)
    evaluated = candidates[:j]
    if evaluated:
        best_coverage = evaluated[0][1]
        best = [index for index, coverage in evaluated if coverage == best_coverage]
        target, rr = least_loaded_cf(loads, best, rr)
        incremental_tokens = max(0, best_coverage - native_coverage)
        estimated_net_ms = incremental_tokens / prefill_tokens_per_ms
        estimated_net_ms -= max(0, loads[target] - loads[native]) * queue_penalty_ms
        if estimated_net_ms > guard_ms:
            return ChooseResult(target, raw, len(evaluated), True, best_coverage, estimated_net_ms)
    return ChooseResult(native, raw, len(evaluated), False, native_coverage, 0.0)


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in str(text).split(";") if part != ""] if text not in ("", None) else []


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def evaluate_cell(updates_path: Path, requests_path: Path, j: int, prefill_tokens_per_ms: float,
                  queue_penalty_ms: float, guard_ms: float, horizons: tuple[int, ...] = (1, 4, 8),
                  value_cost_delay_ms: bool = True) -> dict:
    """Evaluate every forwarded update of one cell; returns long-format rows.

    Returns dict with:
      horizon_rows: one row per (update, use_index) for use_index <= max(horizons)
      update_rows:  one row per update (next-use headline; the RQ1 population)
      sanity:       world1 replay exact-match stats for this cell
    """
    requests = load_csv(requests_path)
    updates = load_csv(updates_path)
    for row in requests:
        row["request_id"] = int(row["request_id"])
        row["dispatch_time_unix"] = float(row["dispatch_time_unix"])
    for row in updates:
        row["writer_seq"] = int(row["writer_seq"])
        row["instance"] = int(row["instance"])
        row["delivered_at_unix"] = float(row["delivered_at_unix"])
        row["pre_delivery_visible_cov"] = int(float(row["pre_delivery_visible_cov"]))
    requests.sort(key=lambda row: (row["dispatch_time_unix"], row["request_id"]))
    by_digest: dict[str, list[dict]] = {}
    for row in requests:
        by_digest.setdefault(row["digest"], []).append(row)

    horizon_rows: list[dict] = []
    update_rows: list[dict] = []
    replay_checked = 0
    replay_matched = 0

    # World-1 VALIDATION: replaying the recorded snapshot UNCHANGED must
    # reproduce the real recorded decision for EVERY dispatched request.
    # This certifies the replay machinery before it is used for World 0.
    for snapshot in requests:
        if snapshot.get("snapshot_rr") in ("", None):
            continue
        cov_w1 = parse_int_list(snapshot["snapshot_cov_row"])
        if not cov_w1:
            continue
        replay_checked += 1
        result = cf_choose(j, prefill_tokens_per_ms, queue_penalty_ms, guard_ms, snapshot["digest"],
                           cov_w1, parse_int_list(snapshot["snapshot_loads"]), int(snapshot["snapshot_rr"]))
        if result.target == int(snapshot["world1_target"]) and result.affinity == bool(int(snapshot["world1_affinity"])):
            replay_matched += 1

    # columns copied verbatim from the ledger row into every output row
    # (send-time features for RQ4 + cell context; computed fields below
    # override nothing here because they are added after **base).
    passthrough = [key for key in (updates[0] if updates else {}) if key]

    def replay(snapshot: dict, cov_row: list[int]) -> ChooseResult:
        return cf_choose(j, prefill_tokens_per_ms, queue_penalty_ms, guard_ms, snapshot["digest"],
                         cov_row, parse_int_list(snapshot["snapshot_loads"]), int(snapshot["snapshot_rr"]))

    max_horizon = max(horizons)
    for update in updates:
        digest = update["digest"]
        future = [row for row in by_digest.get(digest, []) if row["dispatch_time_unix"] > update["delivered_at_unix"]]
        ctx_common = {key: update.get(key, "") for key in passthrough}
        base = {
            "update_seq": update["seq"], "update_kind": update["update_kind"],
            "instance": update["instance"], "digest": digest,
            "coverage_after": update.get("coverage_after", ""),
            "pre_delivery_visible_cov": update["pre_delivery_visible_cov"],
            "delivered_at_unix": update["delivered_at_unix"],
            "signaling_delay_s": update.get("signaling_delay_s", ""),
            "writer_seq": update["writer_seq"],
            "future_use_count": len(future),
            **ctx_common,
        }
        if not future:
            update_rows.append({**base, "evaluated": 0, "superseded_before_use": "",
                                "decision_flip": "", "decision_irrelevant": ""})
            continue
        # World 0/1 for the first max_horizon uses of this update.
        flips = 0
        cumulative_net_gain_ms = 0.0
        u_rows: list[dict] = []
        for use_index, snapshot in enumerate(future[:max_horizon], start=1):
            cov_w1 = parse_int_list(snapshot["snapshot_cov_row"])
            wseq = parse_int_list(snapshot["snapshot_wseq_row"])
            live = wseq[update["instance"]] == update["writer_seq"]
            if live:
                cov_w0 = list(cov_w1)
                cov_w0[update["instance"]] = update["pre_delivery_visible_cov"]
                world0 = replay(snapshot, cov_w0)
            else:
                world0 = None  # erased by a later write: worlds identical
            world1_target = int(snapshot["world1_target"])
            world1_affinity = bool(int(snapshot["world1_affinity"]))
            world1_coverage = int(snapshot["world1_coverage"])
            if world0 is None:
                flip, superseded = 0, 1
                w0_target, w0_affinity, w0_coverage = world1_target, world1_affinity, world1_coverage
                native_target = ""
                estimated_prefill_gain_ms = estimated_queue_delta_ms = estimated_net_gain_ms = 0.0
                reusable_without = reusable_with = reusable_gain = 0
                loads_list = parse_int_list(snapshot["snapshot_loads"])
                loads_w0 = loads_w1 = loads_list[world1_target] if loads_list else ""
            else:
                flip, superseded = int(world0.target != world1_target), 0
                w0_target, w0_affinity, w0_coverage = world0.target, world0.affinity, world0.coverage
                loads = parse_int_list(snapshot["snapshot_loads"])
                native_target, _ = least_loaded_cf(loads, list(range(len(cov_w1))), int(snapshot["snapshot_rr"]))
                native_cov = cov_w1[native_target]
                incremental_w1 = max(0, world1_coverage - native_cov) if world1_affinity else 0
                incremental_w0 = max(0, w0_coverage - native_cov) if w0_affinity else 0
                penalty_w1 = max(0, loads[world1_target] - loads[native_target]) * queue_penalty_ms if world1_affinity else 0.0
                penalty_w0 = max(0, loads[w0_target] - loads[native_target]) * queue_penalty_ms if w0_affinity else 0.0
                estimated_prefill_gain_ms = (incremental_w1 - incremental_w0) / prefill_tokens_per_ms
                estimated_queue_delta_ms = penalty_w1 - penalty_w0
                estimated_net_gain_ms = estimated_prefill_gain_ms - estimated_queue_delta_ms
                reusable_without, reusable_with = incremental_w0, incremental_w1
                reusable_gain = incremental_w1 - incremental_w0
                loads_w0, loads_w1 = loads[w0_target], loads[world1_target]
            # realized (World-1) request telemetry, joined from the record
            realized_ttft_ms = to_float(snapshot.get("ttft_ms"))
            realized_cached_tokens = to_float(snapshot.get("vllm_cached_tokens"))
            signaling_cost_ms = (to_float(update.get("signaling_delay_s"), 0.0) or 0.0) * 1000.0
            # Oracle value = Oracle-B (the dispatcher's own net dispatch
            # benefit).  The measured signaling delay is recorded alongside
            # as a net-of-cost variant, but it is NOT subtracted in the
            # headline value: link-time (seconds of queueing shared by all
            # updates) and dispatch benefit (tens of ms) are incommensurable
            # in raw units, and subtracting them would rank updates by their
            # delivery delay instead of by decision value.  See
            # EXPERIMENT_DESIGN.md amendment 2.
            oracle_value_ms = estimated_net_gain_ms
            oracle_value_net_of_cost_ms = estimated_net_gain_ms - signaling_cost_ms
            horizon_rows.append({
                **base, "use_index": use_index, "use_request_id": snapshot["request_id"],
                "use_dispatch_time_unix": snapshot["dispatch_time_unix"],
                "use_discard": snapshot.get("discard", ""),
                "superseded_before_use": superseded,
                "decision_without_update": w0_target, "decision_with_update": world1_target,
                "decision_flip": flip,
                "native_instance_without": native_target,
                "selected_instance_without": w0_target, "selected_instance_with": world1_target,
                "selected_load_without": loads_w0, "selected_load_with": loads_w1,
                "reusable_coverage_without": reusable_without, "reusable_coverage_with": reusable_with,
                "reusable_coverage_gain": reusable_gain,
                "estimated_prefill_gain_ms": estimated_prefill_gain_ms,
                "estimated_queue_delta_ms": estimated_queue_delta_ms,
                "estimated_net_gain_ms": estimated_net_gain_ms,
                "signaling_cost_ms": signaling_cost_ms,
                "oracle_value_ms": oracle_value_ms,
                "oracle_value_net_of_cost_ms": oracle_value_net_of_cost_ms,
                "realized_ttft_ms": realized_ttft_ms if realized_ttft_ms is not None else "",
                "realized_cached_tokens": realized_cached_tokens if realized_cached_tokens is not None else "",
                "world1_expected_net_ms": snapshot.get("world1_expected_net_ms", ""),
            })
            u_rows.append(horizon_rows[-1])
            if flip and not snapshot.get("discard"):
                flips += 1
                cumulative_net_gain_ms += estimated_net_gain_ms
        # next non-discard use = the Analysis-A headline row
        measured_uses = [row for row in u_rows if not row["use_discard"]]
        next_use = measured_uses[0] if measured_uses else (u_rows[0] if u_rows else None)
        if next_use is not None:
            update_rows.append({
                **base, "evaluated": 1,
                "next_use_request_id": next_use["use_request_id"],
                "next_use_distance_requests": next_use["use_request_id"],
                "next_use_time_distance_s": next_use["use_dispatch_time_unix"] - update["delivered_at_unix"],
                "next_use_discard": next_use["use_discard"],
                "superseded_before_use": next_use["superseded_before_use"],
                "decision_without_update": next_use["decision_without_update"],
                "decision_with_update": next_use["decision_with_update"],
                "decision_flip": next_use["decision_flip"],
                "decision_irrelevant": 1 - next_use["decision_flip"],
                "selected_instance_without": next_use["selected_instance_without"],
                "selected_instance_with": next_use["selected_instance_with"],
                "selected_load_without": next_use["selected_load_without"],
                "selected_load_with": next_use["selected_load_with"],
                "reusable_coverage_without": next_use["reusable_coverage_without"],
                "reusable_coverage_with": next_use["reusable_coverage_with"],
                "reusable_coverage_gain": next_use["reusable_coverage_gain"],
                "estimated_prefill_gain_ms": next_use["estimated_prefill_gain_ms"],
                "estimated_queue_delta_ms": next_use["estimated_queue_delta_ms"],
                "estimated_net_gain_ms": next_use["estimated_net_gain_ms"],
                "signaling_cost_ms": next_use["signaling_cost_ms"],
                "oracle_value_ms": next_use["oracle_value_ms"],
                "realized_ttft_ms": next_use["realized_ttft_ms"],
                "realized_cached_tokens": next_use["realized_cached_tokens"],
                **{f"impact_count_h{h}": sum(1 for row in u_rows
                                             if row["use_index"] <= h
                                             and row["decision_flip"] and not row["use_discard"])
                   for h in horizons},
                **{f"cumulative_value_h{h}": sum(row["estimated_net_gain_ms"] for row in u_rows
                                                 if row["use_index"] <= h
                                                 and row["decision_flip"] and not row["use_discard"])
                   for h in horizons},
            })
    sanity = {"replay_checked": replay_checked, "replay_matched": replay_matched,
              "updates_total": len(updates), "updates_with_future_use": len(update_rows)}
    return {"horizon_rows": horizon_rows, "update_rows": update_rows, "sanity": sanity}
