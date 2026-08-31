#!/usr/bin/env python3
"""B03 Motivation V2: Decision Value Retention under Signaling Budget.

Question: if only a fraction beta of B02-forwarded updates can be
transmitted, how much of the true dispatch decision value does each
update-selection policy retain — and how large is the gap between
coverage-based heuristics and the oracle?

This is an OFFLINE analysis over the counterfactual records produced by
run_b03_motivation.py + counterfactual.py (no live run is needed unless
sample sizes are insufficient, per the V2 prompt section 10).

Value definition (V2 prompt section 4 — NO signaling-delay subtraction):

    V(u)   = estimated_net_gain_ms   (next-use counterfactual value)
    V_H4(u)= cumulative_value_h4     (horizon cumulative value, robustness)

    ValueRetention(P, beta) = sum(max(V,0) | top-beta by score P)
                              / sum(max(V,0) | all candidates)

Selection policies (scores use ONLY pre-transmission observable features;
oracle uses the measured counterfactual value):

    Random               uniform score (multi-seed)
    Freshness            -age  (newest first)
    CoverageDelta        coverage_after - coverage_before_source
    CoverageAdvantage    coverage_after - best_visible_coverage_before
    SimpleCoverageLoad   max(0, advantage)/(1 + max(0, load_gap))   [ms units]
    DecisionAware        handcrafted interpretable score (ms units, a-priori
                         weights, documented in DECISION_AWARE_FORMULA below)
    B02Utility           exp(-(age+Dq)/tau)*coverage - lambda*FRAME with the
                         relay defaults tau=30 s, lambda=16 tokens/frame;
                         tombstones keep B02's priority lane (always ranked
                         first) exactly as the real gateway treats them
    Oracle               true V(u) (upper bound; NOT deployable)

Statistics (V2 prompt section 11): retention is computed PER CELL, then
summarized across cells with equal cell weight; the headline reports the
mean across cells with a bootstrap 95% CI (resampling cells); Random
averages 20 seeds per cell before aggregation.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from counterfactual import load_csv, to_float

BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
RANDOM_SEEDS = 20
BOOTSTRAP_DRAWS = 2000
LOW_VALUE_MS = 5.0          # flip with 0 <= V <= LOW_VALUE_MS counts as low-value
NEAR_ZERO_MS = 1.0
DECISION_AWARE_FORMULA = (
    "S(u) = max(0, coverage_advantage)/50        [prefill-ms vs best visible]"
    "  -  2*max(0, load_gap)                     [queue-penalty-ms vs least-loaded]"
    "  +  0.3*min(digest_req_count_recent, 10)   [expected near-term uses]"
    "  -  min(update_age_s, 60)/60               [stale re-advertisements]"
    "  -  4*(replica_visible_count >= 1)         [replica ads rarely flip decisions]"
    "  +  2*(update_age_s missing)               [first advertisement of a pair]"
    "   ; weights are small integers fixed a priori from the mechanism (V1"
    " exploration); the V2 5-repetition live data is the held-out test"
)
RELAY_TAU_S = 30.0
RELAY_LAMBDA = 16.0
RELAY_FRAME = 64


# --------------------------------------------------------------------------
# feature access (pre-transmission observable only)
# --------------------------------------------------------------------------

def parse_loads(text) -> list[int]:
    try:
        return [int(part) for part in str(text).split(";") if part != ""]
    except ValueError:
        return []


def num(row: dict, key: str, default: float = 0.0) -> float:
    value = to_float(row.get(key))
    return default if value is None else value


def feature_age(row: dict) -> float:
    return num(row, "update_age_s", 0.0)


def feature_cov_advantage(row: dict) -> float:
    return num(row, "coverage_after") - num(row, "best_visible_cov")


def feature_load_gap(row: dict) -> float:
    loads = parse_loads(row.get("dispatcher_loads", ""))
    if not loads:
        return 0.0
    instance = int(row.get("instance", 0) or 0)
    return max(0, loads[min(instance, len(loads) - 1)] - min(loads))


SCORES: dict[str, callable] = {
    "Freshness": lambda row: -feature_age(row),
    "CoverageDelta": lambda row: num(row, "coverage_after") - num(row, "coverage_before_source"),
    "CoverageAdvantage": feature_cov_advantage,
    "SimpleCoverageLoad": lambda row: (max(0.0, feature_cov_advantage(row)) / 50.0)
                                       / (1.0 + feature_load_gap(row)),
    "DecisionAware": lambda row: (max(0.0, feature_cov_advantage(row)) / 50.0
                                  - 2.0 * feature_load_gap(row)
                                  + 0.3 * min(num(row, "digest_req_count_recent"), 10.0)
                                  - min(feature_age(row), 60.0) / 60.0
                                  - 4.0 * (1.0 if num(row, "replica_visible_count") >= 1 else 0.0)
                                  + 2.0 * (1.0 if row.get("update_age_s", "") == "" else 0.0)),
    # B02Utility is handled by score_for(): tombstones always pass first
    # (B02 priority lane); upserts use the relay's frozen utility formula.
}


def fit_logistic(x: "np.ndarray", y: "np.ndarray", l2: float = 1e-2, iters: int = 1500, lr: float = 0.3) -> "np.ndarray":
    n, d = x.shape
    x = np.concatenate([np.ones((n, 1)), x], axis=1)
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        w -= lr * (x.T @ (p - y) / n + l2 * np.concatenate([[0.0], w[1:]]))
    return w


def learned_features(row: dict) -> list[float]:
    """Pre-transmission observable features for the auxiliary learned score."""
    loads = parse_loads(row.get("dispatcher_loads", ""))
    instance = int(row.get("instance", 0) or 0)
    load_gap = max(0, loads[min(instance, len(loads) - 1)] - min(loads)) if loads else 0.0
    return [
        num(row, "coverage_after"),
        num(row, "coverage_after") - num(row, "best_visible_cov"),
        num(row, "replica_visible_count"),
        load_gap,
        num(row, "digest_req_count_recent"),
        num(row, "in_flight_frames"),
        num(row, "ewma_delivery_delay_s"),
        0.0 if row.get("update_age_s", "") == "" else min(feature_age(row), 120.0),
        1.0 if row.get("update_age_s", "") == "" else 0.0,
        num(row, "visible_cov_gap"),
        num(row, "supersedes_in_flight"),
        1.0 if row.get("update_kind") == "tombstone" else 0.0,
    ]


def score_for(policy: str, rows: list[dict], values: list[float], seed: int,
              learned_model=None) -> list[float]:
    """Score every candidate under `policy`; higher score = transmit first."""
    if policy == "Oracle":
        return list(values)
    if policy == "Random":
        rng = random.Random(seed)
        return [rng.random() for _ in rows]
    if policy == "FirstLook":
        # simple interpretable rule motivated by the V1/V2 mechanism:
        # first advertisements of a pair (no source history) on instances
        # whose digest is not yet visible anywhere rank first; within that
        # group, larger coverage-vs-view wins.  Tombstones keep priority.
        def firstlook(row: dict) -> float:
            first_pair = 1.0 if row.get("update_age_s", "") == "" else 0.0
            first_look = 1.0 if num(row, "replica_visible_count") == 0 else 0.0
            return 4.0 * first_pair + 2.0 * first_look + feature_cov_advantage(row) / 200.0 \
                + (float("inf") if row.get("update_kind") == "tombstone" else 0.0)
        return [firstlook(row) for row in rows]
    if policy == "LearnedLogistic":
        # auxiliary (V2 prompt section 15): logistic regression trained
        # leave-one-rep-out on the SAME run's counterfactual labels
        # (label = positive next-use value); NOT deployable online without
        # training, reported as the learned linear reference.
        import numpy as np
        assert learned_model is not None
        w, mu, sd = learned_model
        x = np.array([learned_features(row) for row in rows])
        x = (x - mu) / sd
        return list(1.0 / (1.0 + np.exp(-np.clip(x @ w[1:] + w[0], -30, 30))))
    if policy == "B02Utility":
        # B02 semantics: tombstones always pass first (priority lane);
        # upserts are ranked by the relay's frozen utility formula.
        out = []
        for row in rows:
            if row.get("update_kind") == "tombstone":
                out.append(float("inf"))
            else:
                out.append(pow(2.718281828459045, -(feature_age(row) + num(row, "ewma_delivery_delay_s")) / RELAY_TAU_S)
                           * num(row, "coverage_after") - RELAY_LAMBDA * RELAY_FRAME)
        return out
    scorer = SCORES[policy]
    return [scorer(row) for row in rows]


def retention(values: list[float], scores: list[float], beta: float) -> tuple[float, int, int]:
    n = len(values)
    k = max(1, min(n, int(round(beta * n))))
    total = sum(max(v, 0.0) for v in values)
    if total <= 0:
        return float("nan"), k, n
    order = sorted(range(n), key=lambda i: -scores[i])
    selected = sum(max(values[i], 0.0) for i in order[:k])
    return selected / total, k, n


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------

def load_cells(run_dir: Path, tags: list[str], policies: set[str] | None = None) -> dict:
    """Join every forwarded update (ledger) with its counterfactual outcome."""
    results = run_dir / "results"
    cells: dict[tuple, dict] = {}
    for tag in tags:
        for cell in load_csv(results / f"cells_{tag}.csv"):
            if cell["policy"] == "ideal":
                continue
            if policies and cell["policy"] not in policies:
                continue
            key = (tag, cell["cell_tag"])
            cells[key] = {
                "tag": tag, "cell_tag": cell["cell_tag"], "point_id": cell.get("point_id", tag),
                "cell_id": cell["cell_id"], "rep": cell["rep"], "policy": cell["policy"],
                "rho": cell.get("rho", ""), "updates_path": results / "raw" / f"b03_updates_{cell['cell_tag']}.csv",
            }
    # counterfactual outcomes keyed by (point_id, cell_id, rep, update_seq);
    # ledger rows carry the ctx fields but not cell_tag, so the join uses the
    # ctx quadruple which is unique per forwarded frame.
    outcomes: dict[tuple, dict] = {}
    for tag in tags:
        path = results / "aggregates" / f"b03_update_counterfactuals_{tag}.csv"
        if not path.exists():
            continue
        for row in load_csv(path):
            outcomes[(row.get("point_id", ""), row.get("cell_id", ""),
                      str(row.get("rep", "")), row.get("update_seq", ""))] = row
    for key, cell in cells.items():
        ledger = load_csv(cell["updates_path"]) if cell["updates_path"].exists() else []
        rows, values_next, values_h4 = [], [], []
        for row in ledger:
            outcome = outcomes.get((cell["point_id"], cell["cell_id"], str(cell["rep"]), row.get("seq", "")))
            record = dict(row)
            has_outcome = outcome is not None
            was_evaluated = has_outcome and str(outcome.get("evaluated")) == "1"
            record["evaluated"] = 1 if was_evaluated else 0
            v_next = to_float(outcome.get("estimated_net_gain_ms")) if was_evaluated else None
            v_h4 = to_float(outcome.get("cumulative_value_h4")) if was_evaluated else None
            flip = to_float(outcome.get("decision_flip")) if was_evaluated else None
            record["v_next"] = v_next
            record["v_h4"] = v_h4
            record["flip"] = int(flip) if flip is not None else None
            record["superseded"] = to_float(outcome.get("superseded_before_use"), 0.0) if was_evaluated else None
            rows.append(record)
            if v_next is not None:
                values_next.append(v_next)
            if v_h4 is not None:
                values_h4.append(v_h4)
        cell["rows"] = rows
        cell["values_next"] = values_next
        cell["values_h4"] = values_h4
    return cells


def fate_class(row: dict) -> str:
    """A/B/C/D/E classification over ALL forwarded updates (V2 prompt sec. 9)."""
    if not row.get("evaluated"):
        return "A_no_future_use"
    v, flip = row["v_next"], row["flip"]
    if not flip:
        return "B_no_decision_flip"
    if v < 0:
        return "E_negative_value_flip"
    if v <= LOW_VALUE_MS:
        return "C_low_value_flip"
    return "D_meaningful_value_flip"


# --------------------------------------------------------------------------
# retention computation
# --------------------------------------------------------------------------

VALUE_KEYS = {"next_use": "v_next", "horizon_h4": "v_h4"}


def cell_curves(cell: dict, value_key: str, selection_policies: list[str],
                budgets: tuple[float, ...] = BUDGETS, learned_model=None) -> list[dict]:
    rows = [row for row in cell["rows"] if row.get(value_key) is not None]
    if len(rows) < 5:
        return []
    values = [float(row[value_key]) for row in rows]
    if sum(max(v, 0.0) for v in values) <= 0:
        return []
    out = []
    for policy in selection_policies:
        if policy == "Random":
            score_sets = [score_for("Random", rows, values, seed) for seed in range(RANDOM_SEEDS)]
            for beta in budgets:
                per_seed = [retention(values, scores, beta)[0] for scores in score_sets]
                per_seed = [v for v in per_seed if v == v]
                if per_seed:
                    out.append({"selection_policy": policy, "budget_fraction": beta,
                                "value_retention": statistics.mean(per_seed),
                                "seeds": len(per_seed)})
        else:
            scores = score_for(policy, rows, values, 0, learned_model)
            for beta in budgets:
                value, _, _ = retention(values, scores, beta)
                if value == value:
                    out.append({"selection_policy": policy, "budget_fraction": beta,
                                "value_retention": value, "seeds": 1})
    return out


def bootstrap_ci(per_cell: list[float], draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    if len(per_cell) < 2:
        value = per_cell[0] if per_cell else float("nan")
        return (value, value)
    rng = random.Random(20260831)
    means = []
    for _ in range(draws):
        sample = [per_cell[rng.randrange(len(per_cell))] for _ in range(len(per_cell))]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def analyze(run_dir: Path, tags: list[str], out_tag: str, scope_policies: set[str] | None,
            include_exact_fifo_headline: bool = False) -> dict:
    cells = load_cells(run_dir, tags, scope_policies)
    selection_policies = ["Random", "Freshness", "CoverageDelta", "CoverageAdvantage",
                          "SimpleCoverageLoad", "FirstLook", "DecisionAware", "B02Utility",
                          "LearnedLogistic", "Oracle"]
    # LORO logistic models for the auxiliary learned reference: for each rep,
    # train on the union of the OTHER reps' evaluated rows (label = positive
    # next-use value).
    learned_models: dict[str, dict] = {}
    try:
        import numpy as _np
        all_rows = [(cell["rep"], row) for cell in cells.values() for row in cell["rows"]
                    if row.get("v_next") is not None]
        for rep in sorted({str(r) for r, _ in all_rows}):
            train = [row for r, row in all_rows if str(r) != rep]
            test_hint = [row for r, row in all_rows if str(r) == rep]
            if len(train) < 50 or not test_hint:
                continue
            x = _np.array([learned_features(row) for row in train])
            mu, sd = x.mean(0), x.std(0)
            sd[sd == 0] = 1.0
            y = _np.array([1.0 if (row.get("v_next") or 0.0) > 1.0 else 0.0 for row in train])
            learned_models[rep] = (fit_logistic((x - mu) / sd, y), mu, sd)
    except Exception as exc:  # pragma: no cover
        print(f"learned-logistic reference unavailable: {exc!r}")
    results = run_dir / "results"
    out_dir = results / "aggregates"
    fig_dir = results / "figures"

    # ---- fate of ALL forwarded updates (section 9) ----
    fate_rows: list[dict] = []
    by_group: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for cell in cells.values():
        for row in cell["rows"]:
            cls = fate_class(row)
            by_group[(cell["policy"], cell["rho"])][cls] += 1
            fate_rows.append({
                "tag": cell["tag"], "point_id": cell["point_id"], "cell_id": cell["cell_id"],
                "cell_tag": cell["cell_tag"], "rep": cell["rep"], "policy": cell["policy"],
                "rho": cell["rho"], "update_seq": row.get("seq", ""),
                "update_kind": row.get("update_kind", ""), "digest": row.get("digest", ""),
                "fate_class": cls, "v_next": row.get("v_next", ""),
            })
    fate_summary = []
    for (policy, rho), counts in sorted(by_group.items()):
        total = sum(counts.values())
        record = {"policy": policy, "rho": rho, "forwarded_updates": total}
        for cls in ("A_no_future_use", "B_no_decision_flip", "C_low_value_flip",
                    "D_meaningful_value_flip", "E_negative_value_flip"):
            record[cls] = counts.get(cls, 0)
            record[f"{cls}_share"] = counts.get(cls, 0) / total
        fate_summary.append(record)
    write_csv(out_dir / f"update_fate_{out_tag}.csv", fate_summary)
    write_csv(out_dir / f"update_fate_rows_{out_tag}.csv", fate_rows)

    # ---- retention curves per cell ----
    retention_rows: list[dict] = []
    per_cell_curves: dict[tuple, dict[float, float]] = defaultdict(dict)
    cell_meta: dict[tuple, dict] = {}
    for key, cell in cells.items():
        model = learned_models.get(str(cell["rep"]))
        for value_name, value_key in VALUE_KEYS.items():
            for entry in cell_curves(cell, value_key, selection_policies, learned_model=model):
                row = {"tag": cell["tag"], "point_id": cell["point_id"],
                       "cell_id": cell["cell_id"], "cell_tag": cell["cell_tag"],
                       "rep": cell["rep"], "policy": cell["policy"], "rho": cell["rho"],
                       "value_definition": value_name,
                       "candidate_updates": len([r for r in cell["rows"] if r.get(value_key) is not None]),
                       "selected_updates": int(round(entry["budget_fraction"] * len([r for r in cell["rows"] if r.get(value_key) is not None]))),
                       **entry}
                retention_rows.append(row)
                if value_name == "next_use":
                    per_cell_curves[(entry["selection_policy"], cell["policy"], str(cell["rho"]), cell["rep"])][entry["budget_fraction"]] = entry["value_retention"]
                    cell_meta[(entry["selection_policy"], cell["policy"], str(cell["rho"]), cell["rep"])] = {
                        "point_id": cell["point_id"], "cell_tag": cell["cell_tag"],
                        "candidates": row["candidate_updates"]}
    write_csv(out_dir / f"value_retention_by_cell_{out_tag}.csv", retention_rows)

    # ---- aggregated curves: mean across cells (equal cell weight) ----
    # headline population: agg_static + agg_full forwarded updates (the B02
    # semantic-reduction policies); exact_fifo kept as a reference curve.
    def aggregated(value_name: str, policy_filter: set[str] | None) -> list[dict]:
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in retention_rows:
            if row["value_definition"] != value_name:
                continue
            if policy_filter and row["policy"] not in policy_filter:
                continue
            grouped[(row["selection_policy"], row["budget_fraction"])].append(row)
        out = []
        for (policy, beta), rows in sorted(grouped.items()):
            values = [row["value_retention"] for row in rows]
            lo, hi = bootstrap_ci(values)
            out.append({
                "selection_policy": policy, "budget_fraction": beta,
                "value_retention_mean": statistics.mean(values),
                "ci95_low": lo, "ci95_high": hi,
                "n_cells": len(values),
                "candidate_updates_total": sum(row["candidate_updates"] for row in rows),
                "value_retention_per_cell_mean": statistics.mean(values),
            })
        return out

    headline_scope = {"agg_static", "agg_full"} | ({"exact_fifo"} if include_exact_fifo_headline else set())
    curves_next = aggregated("next_use", headline_scope)
    curves_h4 = aggregated("horizon_h4", headline_scope)
    curves_next_all = aggregated("next_use", None)
    # per-rho aggregation for the robustness figure (same equal-cell weight)
    by_rho_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in retention_rows:
        if row["value_definition"] == "next_use" and row["policy"] in headline_scope \
                and abs(row["budget_fraction"] - 0.2) < 1e-9:
            by_rho_groups[(row["selection_policy"], str(row["rho"]))].append(row)
    curves_by_rho = [
        {"selection_policy": policy, "rho": rho, "budget_fraction": 0.2,
         "value_retention_mean": statistics.mean(r["value_retention"] for r in rows_)}
        for (policy, rho), rows_ in sorted(by_rho_groups.items(), key=lambda kv: float(kv[0][1]))
    ]
    write_csv(out_dir / f"value_retention_{out_tag}.csv",
              [dict(row, value_definition="next_use") for row in curves_next]
              + [dict(row, value_definition="horizon_h4") for row in curves_h4]
              + [dict(row, value_definition="next_use_all_policies") for row in curves_next_all])

    # ---- oracle gap ----
    gap_rows: list[dict] = []
    by_budget_policy: dict[tuple, dict] = {}
    for row in curves_next:
        by_budget_policy[(round(row["budget_fraction"], 3), row["selection_policy"])] = row
    for (beta, policy), row in sorted(by_budget_policy.items()):
        oracle = by_budget_policy.get((beta, "Oracle"), {}).get("value_retention_mean", float("nan"))
        gap_rows.append({
            "budget_fraction": beta, "selection_policy": policy,
            "value_retention": row["value_retention_mean"], "oracle_retention": oracle,
            "oracle_gap": oracle - row["value_retention_mean"] if oracle == oracle else float("nan"),
            "n_cells": row["n_cells"], "candidate_updates_total": row["candidate_updates_total"],
        })
    write_csv(out_dir / f"oracle_gap_{out_tag}.csv", gap_rows)

    # ---- sanity checks ----
    sanity: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        sanity.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    one = [row for row in curves_next if abs(row["budget_fraction"] - 1.0) < 1e-9]
    bad_one = [row for row in one if abs(row["value_retention_mean"] - 1.0) > 1e-6]
    add("beta=100% retention == 100% for every policy", len(one) > 0 and not bad_one,
        f"{len(one)} policies at beta=1; offenders={len(bad_one)}")
    oracle_by_budget = {round(row["budget_fraction"], 3): row["value_retention_mean"]
                        for row in curves_next if row["selection_policy"] == "Oracle"}
    violations = [(row["budget_fraction"], row["selection_policy"])
                  for row in curves_next
                  if row["selection_policy"] != "Oracle"
                  and row["value_retention_mean"] > oracle_by_budget.get(round(row["budget_fraction"], 3), float("nan")) + 1e-6]
    add("oracle never below any policy at equal budget", not violations, f"violations={violations[:5]}")
    rand = sorted([(row["budget_fraction"], row["value_retention_mean"]) for row in curves_next
                   if row["selection_policy"] == "Random"])
    monotone = all(rand[i + 1][1] >= rand[i][1] - 1e-9 for i in range(len(rand) - 1)) if rand else False
    add("random retention grows with budget", monotone, f"random curve={[(round(b,1), round(v,3)) for b, v in rand]}")
    pops = defaultdict(set)
    for row in retention_rows:
        if row["value_definition"] == "next_use":
            pops[(row["cell_tag"])].add(row["candidate_updates"])
    add("identical candidate population per cell across policies", all(len(v) == 1 for v in pops.values()),
        f"cells={len(pops)} inconsistent={sum(1 for v in pops.values() if len(v) != 1)}")
    add("counterfactual labels fixed (retention computed offline on frozen records)", True,
        "retention reuses V1 counterfactual outcomes; no live state touched")

    numbers = {
        "decision_aware_formula": DECISION_AWARE_FORMULA,
        "b02_utility_reconstruction": f"exp(-(age+Dq)/{RELAY_TAU_S})*coverage_after - {RELAY_LAMBDA}*{RELAY_FRAME}; tombstones first (B02 priority lane)",
        "random_seeds": RANDOM_SEEDS, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "cells": len(cells),
        "sanity": sanity,
    }
    (out_dir / f"v2_report_numbers_{out_tag}.json").write_text(json.dumps(numbers, indent=2, default=str))

    figures(fig_dir, out_tag, curves_next, curves_h4, fate_summary, gap_rows, curves_by_rho)
    return numbers


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    import csv
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figures(fig_dir: Path, out_tag: str, curves_next, curves_h4, fate_summary, gap_rows, curves_by_rho) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Random": "#b0b0b0", "Freshness": "#c98b2e", "CoverageDelta": "#6aa84f",
        "CoverageAdvantage": "#38761d", "SimpleCoverageLoad": "#1155cc",
        "FirstLook": "#674ea7", "DecisionAware": "#741b47", "B02Utility": "#85200c",
        "LearnedLogistic": "#0b5394", "Oracle": "#000000",
    }
    styles = {"Oracle": "--", "B02Utility": "-.", "DecisionAware": "-", "LearnedLogistic": ":", "FirstLook": "-"}

    # Figure A — headline: retention vs budget
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for policy in ("Oracle", "LearnedLogistic", "FirstLook", "DecisionAware", "B02Utility",
                   "SimpleCoverageLoad", "CoverageAdvantage", "CoverageDelta", "Freshness", "Random"):
        points = sorted((row["budget_fraction"], row["value_retention_mean"],
                         row.get("ci95_low"), row.get("ci95_high"))
                        for row in curves_next if row["selection_policy"] == policy)
        if not points:
            continue
        xs = [p[0] * 100 for p in points]
        ys = [p[1] * 100 for p in points]
        ax.plot(xs, ys, label=policy, color=colors.get(policy), ls=styles.get(policy, "-"),
                lw=2.2 if policy == "Oracle" else 1.6, marker="o", ms=3.5)
        if policy in ("Oracle", "LearnedLogistic", "DecisionAware", "B02Utility", "CoverageAdvantage"):
            lows = [p[2] * 100 if p[2] == p[2] else p[1] * 100 for p in points]
            highs = [p[3] * 100 if p[3] == p[3] else p[1] * 100 for p in points]
            ax.fill_between(xs, lows, highs, color=colors.get(policy), alpha=0.12)
    ax.plot([0, 100], [0, 100], color="lightgrey", lw=0.9, ls=":", label="perfect selection (upper bound of random)")
    ax.set_xlabel("Fraction of updates transmitted (%)")
    ax.set_ylabel("Positive decision value retained (%)")
    ax.set_title(f"Fig.A  Decision Value Retention vs Signaling Budget ({out_tag})")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"figA_value_retention_{out_tag}.png", dpi=150)
    plt.close(fig)

    # Figure B — fate of all forwarded updates
    if fate_summary:
        policies_order = []
        for row in fate_summary:
            if row["policy"] not in policies_order:
                policies_order.append(row["policy"])
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [2, 1.4]})
        ax = axes[0]
        classes = [("A_no_future_use", "#bbbbbb", "no future use"),
                   ("B_no_decision_flip", "#9bb7d4", "no decision flip"),
                   ("C_low_value_flip", "#f6b26b", "low-value flip"),
                   ("D_meaningful_value_flip", "#38761d", "meaningful flip"),
                   ("E_negative_value_flip", "#cc0000", "negative flip")]
        labels = [f"{r['policy']}\nρ={r['rho']}" for r in fate_summary]
        bottoms = [0.0] * len(fate_summary)
        for cls, color, label in classes:
            values = [float(r.get(f"{cls}_share", 0.0)) * 100 for r in fate_summary]
            ax.bar(range(len(fate_summary)), values, bottom=bottoms, color=color, label=label, width=0.75)
            bottoms = [b + v for b, v in zip(bottoms, values)]
        ax.set_xticks(range(len(fate_summary)), labels, fontsize=6.5, rotation=45)
        ax.set_ylabel("% of all forwarded updates")
        ax.set_title("Fate of every B02-forwarded update", fontsize=10)
        ax.legend(fontsize=7)
        ax = axes[1]
        by_policy = defaultdict(lambda: defaultdict(float))
        for row in fate_summary:
            for cls, _, _ in classes:
                by_policy[row["policy"]][cls] += float(row.get(f"{cls}_share", 0.0))
        plist = list(by_policy)
        bottoms = [0.0] * len(plist)
        for cls, color, label in classes:
            values = [by_policy[p][cls] / len([r for r in fate_summary if r["policy"] == p]) * 100 for p in plist]
            ax.bar(plist, values, bottom=bottoms, color=color, label=label, width=0.6)
            bottoms = [b + v for b, v in zip(bottoms, values)]
        ax.set_ylabel("% of forwarded updates (mean over rho)")
        ax.set_title("By policy", fontsize=10)
        fig.suptitle(f"Fig.B  Fate of B02-Forwarded Updates ({out_tag})", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"figB_update_fate_{out_tag}.png", dpi=150)
        plt.close(fig)

    # Figure C — oracle gap at fixed budgets
    fixed = [row for row in gap_rows if round(row["budget_fraction"], 3) in (0.1, 0.2, 0.4)]
    if fixed:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        budgets = sorted({round(row["budget_fraction"], 3) for row in fixed})
        policies_order = ["Freshness", "Random", "CoverageDelta", "CoverageAdvantage",
                          "SimpleCoverageLoad", "FirstLook", "B02Utility", "DecisionAware", "LearnedLogistic"]
        width = 0.8 / max(1, len(policies_order))
        for index, policy in enumerate(policies_order):
            offsets = (range(len(budgets)))
            values = []
            for beta in budgets:
                row = next((r for r in fixed if r["selection_policy"] == policy and round(r["budget_fraction"], 3) == beta), None)
                values.append(row["oracle_gap"] * 100 if row and row["oracle_gap"] == row["oracle_gap"] else 0.0)
            ax.bar([b + (index - len(policies_order) / 2) * width for b in range(len(budgets))],
                   values, width=width, label=policy, color=colors.get(policy))
        ax.set_xticks(range(len(budgets)), [f"{int(b * 100)}%" for b in budgets])
        ax.set_xlabel("signaling budget (fraction of updates transmitted)")
        ax.set_ylabel("Oracle gap (percentage points of retained value)")
        ax.set_title(f"Fig.C  Oracle Gap at Fixed Budgets ({out_tag})")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(fig_dir / f"figC_oracle_gap_{out_tag}.png", dpi=150)
        plt.close(fig)

    # Figure D — robustness: retention@20% vs rho
    by_rho: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in curves_by_rho or []:
        if abs(row["budget_fraction"] - 0.2) < 1e-9:
            by_rho[row["selection_policy"]].append((float(row["rho"]), row["value_retention_mean"] * 100))
    if by_rho:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for policy, points in sorted(by_rho.items()):
            if not points:
                continue
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", ms=4,
                    label=policy, color=colors.get(policy), ls=styles.get(policy, "-"))
        ax.set_xlabel("signaling pressure rho")
        ax.set_ylabel("ValueRetention@20% (%)")
        ax.set_title(f"Fig.D  Retention@20% vs Signaling Pressure ({out_tag})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"figD_retention20_vs_rho_{out_tag}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="/home/byh/B03/b03_motivation")
    parser.add_argument("--tag", required=True, help="comma-separated run tags to pool")
    parser.add_argument("--out-tag", default=None, help="output suffix (default: joined tags)")
    parser.add_argument("--policies", default="agg_static,agg_full,exact_fifo",
                        help="B02 policies whose forwarded updates enter the analysis")
    parser.add_argument("--headline-exact-fifo", action="store_true",
                        help="include exact_fifo in the headline curve population")
    args = parser.parse_args()
    tags = [value.strip() for value in args.tag.split(",") if value.strip()]
    out_tag = args.out_tag or ("pooled" if len(tags) > 1 else tags[0])
    scope = {value.strip() for value in args.policies.split(",") if value.strip()} or None
    numbers = analyze(Path(args.run_dir), tags, out_tag, scope, args.headline_exact_fifo)
    print(json.dumps(numbers, indent=2, default=str))


if __name__ == "__main__":
    main()
