#!/usr/bin/env python3
"""B03 motivation analysis: RQ1-RQ4 aggregates, figures, sanity, verdict.

Consumes the raw artifacts of one or more run tags under
  <run-dir>/results/cells_{tag}.csv
  <run-dir>/results/raw/b03_updates_{cell_tag}.csv
  <run-dir>/results/raw/b03_requests_{cell_tag}.csv

Writes
  results/aggregates/b03_update_counterfactuals_{tag}.csv   (per-update, next use)
  results/aggregates/b03_update_horizons_{tag}.csv          (per (update, use))
  results/aggregates/rq{1,2,3,4}_summary_{tag}.csv
  results/aggregates/sanity_checks_b03_{tag}.csv
  results/aggregates/report_numbers_{tag}.json
  results/figures/fig1..fig5_{tag}.png

All evaluation is OFFLINE (counterfactual.py) and read-only w.r.t. the run.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from counterfactual import evaluate_cell, load_csv, to_float

FIG_DIR_NAME = "figures"
AGG_DIR_NAME = "aggregates"

# Pre-declared interpretation rules for the verdict (documented in
# EXPERIMENT_DESIGN.md "Success criteria"); they translate the qualitative
# conditions A-E into measurable thresholds BEFORE looking at results.
EPS_NET_GAIN_MS = 1.0          # |net gain| below this counts as near-zero
CONDITION = {
    "A_min_irrelevant_rate": 0.50,   # pooled decision-irrelevant fraction
    "A_min_cell_irrelevant": 0.30,   # weakest cell must still be substantial
    "B_max_clean_flip_share": 0.70,  # <70% of flips clearly positive => heterogeneity
    "B_min_neg_or_zero_share": 0.20, # >=20% of flips near-zero/negative
    "C_min_top20_share": 0.50,       # top-20% updates hold >=50% of positive value
    "D_max_heuristic_spearman": 0.30,# freshness/coverage vs value weak
    "E_min_auroc_gain": 0.05,        # decision-aware beats freshness-only
    "E_min_auroc": 0.65,
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


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


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def group_key(row: dict) -> tuple:
    return (str(row.get("point_id", "")), str(row.get("policy", "")), str(row.get("rho", "")))


def fmt(value: float) -> str:
    return f"{value:.4f}"


# --------------------------------------------------------------------------
# RQ4 machinery: numpy logistic regression + rank metrics, leave-one-rep-out
# --------------------------------------------------------------------------

def logistic_fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 800, lr: float = 0.5) -> np.ndarray:
    n, d = x.shape
    x = np.concatenate([np.ones((n, 1)), x], axis=1)
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = x.T @ (p - y) / n + l2 * np.concatenate([[0.0], w[1:]])
        w -= lr * grad
    return w


def logistic_prob(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = cum_pos = 0
    precision_sum = 0.0
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    for i, flag in enumerate(y_sorted, start=1):
        cum_pos += int(flag)
        if flag:
            tp += 1
            precision_sum += tp / i
    return float(precision_sum / total_pos)


FEATURE_SETS: dict[str, list[str]] = {
    "freshness_only": ["update_age_s", "time_since_prev_update_s"],
    "coverage_only": ["coverage_after", "source_coverage_delta", "best_visible_cov"],
    "fresh_cov": ["update_age_s", "time_since_prev_update_s", "coverage_after", "source_coverage_delta", "best_visible_cov"],
    "decision_aware": ["update_age_s", "time_since_prev_update_s", "coverage_after", "source_coverage_delta",
                        "best_visible_cov", "visible_cov_gap", "second_visible_cov", "replica_visible_count",
                        "digest_req_count_recent", "in_flight_frames", "ewma_delivery_delay_s",
                        "supersedes_in_flight", "rho", "is_tombstone", "dispatcher_load_sum"],
}


def design_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    cols = []
    for name in features:
        values = []
        for row in rows:
            value = to_float(row.get(name))
            values.append(value if value is not None else np.nan)
        arr = np.array(values, dtype=float)
        median = np.nanmedian(arr) if not np.all(np.isnan(arr)) else 0.0
        arr = np.nan_to_num(arr, nan=median)
        cols.append(arr)
    x = np.stack(cols, axis=1)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def leave_one_rep_out(rows: list[dict], features: list[str], label_fn) -> tuple[float, float, int]:
    """Mean AUROC/AUPRC over held-out reps (groups with both classes only)."""
    aurocs, auprcs = [], []
    reps = sorted({str(row.get("rep")) for row in rows})
    for rep in reps:
        test = [row for row in rows if str(row.get("rep")) == rep]
        train = [row for row in rows if str(row.get("rep")) != rep]
        y_test = np.array([label_fn(row) for row in test], dtype=int)
        y_train = np.array([label_fn(row) for row in train], dtype=int)
        if len(set(y_test.tolist())) < 2 or len(set(y_train.tolist())) < 2 or len(train) < 20:
            continue
        x_train = design_matrix(train, features)
        x_test = design_matrix(test, features)
        w = logistic_fit(x_train, y_train)
        scores = logistic_prob(w, x_test)
        aurocs.append(auroc(y_test, scores))
        auprcs.append(auprc(y_test, scores))
    if not aurocs:
        return float("nan"), float("nan"), 0
    return float(np.nanmean(aurocs)), float(np.nanmean(auprcs)), len(aurocs)


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------

def analyze(run_dir: Path, tag: str, j: int, prefill_tokens_per_ms: float,
            queue_penalty_ms: float, guard_ms: float) -> dict:
    results = run_dir / "results"
    cells = load_csv(results / f"cells_{tag}.csv")
    link_cells = [row for row in cells if row["policy"] != "ideal"]
    all_update_rows: list[dict] = []
    all_horizon_rows: list[dict] = []
    sanity_rows: list[dict] = []

    def add_sanity(name: str, passed: bool, detail: str) -> None:
        sanity_rows.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    for cell in link_cells:
        cell_tag = cell["cell_tag"]
        updates_path = results / "raw" / f"b03_updates_{cell_tag}.csv"
        requests_path = results / "raw" / f"b03_requests_{cell_tag}.csv"
        if not updates_path.exists() or not requests_path.exists():
            add_sanity(f"{cell_tag}: raw artifacts present", False, f"missing {updates_path.name} / {requests_path.name}")
            continue
        outcome = evaluate_cell(updates_path, requests_path, j, prefill_tokens_per_ms,
                                queue_penalty_ms, guard_ms)
        all_update_rows.extend(outcome["update_rows"])
        all_horizon_rows.extend(outcome["horizon_rows"])
        sanity = outcome["sanity"]
        add_sanity(f"{cell_tag}: World-1 replay reproduces every real decision",
                   sanity["replay_checked"] == sanity["replay_matched"] and sanity["replay_checked"] > 0,
                   f"replayed={sanity['replay_checked']} matched={sanity['replay_matched']}")
        delivered = int(float(cell.get("net_msgs_delivered", 0) or 0))
        add_sanity(f"{cell_tag}: ledger covers exactly the delivered frames",
                   sanity["updates_total"] == delivered,
                   f"ledger={sanity['updates_total']} delivered={delivered}")
        # leakage guard: every used feature timestamp precedes its use
        leaks = sum(1 for row in outcome["horizon_rows"]
                    if to_float(row.get("send_ts_unix")) is not None
                    and to_float(row.get("send_ts_unix")) >= to_float(row.get("use_dispatch_time_unix"), float("inf")))
        add_sanity(f"{cell_tag}: no future information in pre-transmission features", leaks == 0, f"leaking_rows={leaks}")

    run_checks_path = results / f"sanity_checks_{tag}.csv"
    if run_checks_path.exists():
        for row in load_csv(run_checks_path):
            add_sanity(f"run integrity: {row['check_name']}", row["status"] == "PASS", str(row.get("offending_rows", "")))

    aggregate_rows = all_update_rows
    for row in aggregate_rows:
        row["is_tombstone"] = 1 if row.get("update_kind") == "tombstone" else 0
        row["dispatcher_load_sum"] = sum(parse_loads(row.get("dispatcher_loads", "")))
    out = {"aggregate_rows": aggregate_rows, "horizon_rows": all_horizon_rows, "sanity": sanity_rows}

    # ---------------- RQ1 ----------------
    rq1_rows: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        if row.get("evaluated") == 1 and row.get("next_use_discard") in ("False", "0", "", None):
            groups[group_key(row)].append(row)
    for key, rows in sorted(groups.items()):
        reps = sorted({str(row.get("rep")) for row in rows})
        per_rep = []
        for rep in reps:
            sub = [row for row in rows if str(row.get("rep")) == rep]
            flips = sum(int(row["decision_flip"]) for row in sub)
            per_rep.append(1.0 - flips / len(sub))
        flips = sum(int(row["decision_flip"]) for row in rows)
        superseded = sum(int(row.get("superseded_before_use") or 0) for row in rows)
        rq1_rows.append({
            "point_id": key[0], "policy": key[1], "rho": key[2], "n_updates_next_use": len(rows),
            "decision_flip_rate": flips / len(rows),
            "decision_irrelevant_rate": 1.0 - flips / len(rows),
            "irrelevant_rate_mean_across_reps": sum(per_rep) / len(per_rep),
            "irrelevant_rate_min_across_reps": min(per_rep),
            "superseded_before_use_share": superseded / len(rows),
        })
    # ---------------- RQ2 ----------------
    rq2_rows: list[dict] = []
    flip_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        if row.get("evaluated") == 1 and row.get("next_use_discard") in ("False", "0", "", None) and int(row.get("decision_flip") or 0) == 1:
            flip_groups[group_key(row)].append(row)
    for key, rows in sorted(flip_groups.items()):
        gains = [float(row["estimated_net_gain_ms"]) for row in rows if row.get("estimated_net_gain_ms") not in ("", None)]
        positive = [value for value in gains if value > EPS_NET_GAIN_MS]
        negative = [value for value in gains if value < -EPS_NET_GAIN_MS]
        near_zero = len(gains) - len(positive) - len(negative)
        rq2_rows.append({
            "point_id": key[0], "policy": key[1], "rho": key[2], "n_flips": len(rows),
            "share_positive_gain": len(positive) / len(gains) if gains else "",
            "share_near_zero_gain": near_zero / len(gains) if gains else "",
            "share_negative_gain": len(negative) / len(gains) if gains else "",
            "net_gain_mean_ms": sum(gains) / len(gains) if gains else "",
            "net_gain_p10_ms": percentile(gains, 10) if gains else "",
            "net_gain_p90_ms": percentile(gains, 90) if gains else "",
            "reusable_gain_mean_tokens": sum(float(row["reusable_coverage_gain"]) for row in rows) / len(rows),
        })
    # ---------------- RQ3 ----------------
    rq3_rows: list[dict] = []
    pareto_curves: dict[tuple, tuple[list[float], list[float]]] = {}
    for key, rows in sorted(groups.items()):
        values = [float(row["oracle_value_ms"]) for row in rows if row.get("oracle_value_ms") not in ("", None)]
        if not values:
            continue
        positives = sorted((value for value in values if value > 0), reverse=True)
        total_positive = sum(positives)
        ordered_all = sorted(values, reverse=True)
        cumulative = np.cumsum([max(0.0, value) for value in ordered_all])
        curve_y = (cumulative / total_positive).tolist() if total_positive > 0 else [0.0] * len(ordered_all)
        curve_x = [(index + 1) / len(ordered_all) for index in range(len(ordered_all))]
        pareto_curves[key] = (curve_x, curve_y)

        def top_share(fraction: float) -> float:
            head = positives[:max(1, int(round(fraction * len(positives))))]
            return sum(head) / total_positive if total_positive > 0 else float("nan")

        rq3_rows.append({
            "point_id": key[0], "policy": key[1], "rho": key[2], "n_updates": len(values),
            "total_positive_value_ms": total_positive,
            "share_positive_updates": len(positives) / len(values),
            "top10pct_share_of_positive_value": top_share(0.10),
            "top20pct_share_of_positive_value": top_share(0.20),
            "top30pct_share_of_positive_value": top_share(0.30),
            "value_gini": gini(values),
        })
    # ---------------- RQ4 ----------------
    rq4_rows: list[dict] = []
    for key, rows in sorted(groups.items()):
        if len({str(row.get("rep")) for row in rows}) < 2:
            continue
        values = [float(row["oracle_value_ms"]) for row in rows if row.get("oracle_value_ms") not in ("", None)]
        q3 = percentile(values, 75) if values else float("nan")
        labels = {
            "flip_next_use": lambda row: int(row.get("decision_flip") or 0),
            "positive_value": lambda row: int(to_float(row.get("estimated_net_gain_ms"), 0.0) > EPS_NET_GAIN_MS),
            "high_value_top25pct": lambda row: int(to_float(row.get("oracle_value_ms"), float("-inf")) >= q3),
        }
        # pooled across groups for the headline; per-group rows kept too
        for label_name, label_fn in labels.items():
            prevalence = sum(label_fn(row) for row in rows) / len(rows)
            for feature_set, features in FEATURE_SETS.items():
                auroc_value, auprc_value, folds = leave_one_rep_out(rows, features, label_fn)
                rq4_rows.append({
                    "point_id": key[0], "policy": key[1], "rho": key[2], "label": label_name,
                    "feature_set": feature_set, "n_updates": len(rows), "prevalence": prevalence,
                    "auroc_loro_mean": auroc_value, "auprc_loro_mean": auprc_value, "folds": folds,
                })
    pooled_rq4: list[dict] = []
    all_rows = [row for rows in groups.values() for row in rows]
    if len({str(row.get("rep")) for row in all_rows}) >= 2:
        values = [float(row["oracle_value_ms"]) for row in all_rows if row.get("oracle_value_ms") not in ("", None)]
        q3 = percentile(values, 75) if values else float("nan")
        labels = {
            "flip_next_use": lambda row: int(row.get("decision_flip") or 0),
            "positive_value": lambda row: int(to_float(row.get("estimated_net_gain_ms"), 0.0) > EPS_NET_GAIN_MS),
            "high_value_top25pct": lambda row: int(to_float(row.get("oracle_value_ms"), float("-inf")) >= q3),
        }
        for label_name, label_fn in labels.items():
            prevalence = sum(label_fn(row) for row in all_rows) / len(all_rows)
            for feature_set, features in FEATURE_SETS.items():
                auroc_value, auprc_value, folds = leave_one_rep_out(all_rows, features, label_fn)
                pooled_rq4.append({
                    "scope": "pooled", "label": label_name, "feature_set": feature_set,
                    "n_updates": len(all_rows), "prevalence": prevalence,
                    "auroc_loro_mean": auroc_value, "auprc_loro_mean": auprc_value, "folds": folds,
                })

    # ---------------- verdict conditions ----------------
    numbers: dict = {"eps_net_gain_ms": EPS_NET_GAIN_MS, **CONDITION}
    if rq1_rows:
        pooled_irrelevant = 1.0 - sum(int(row.get("decision_flip") or 0) for rows in groups.values() for row in rows) / max(1, sum(len(rows) for rows in groups.values()))
        min_cell = min(row["decision_irrelevant_rate"] for row in rq1_rows)
        numbers["A_pooled_irrelevant_rate"] = pooled_irrelevant
        numbers["A_min_cell_irrelevant_rate"] = min_cell
        numbers["A_hold"] = bool(pooled_irrelevant >= CONDITION["A_min_irrelevant_rate"] and min_cell >= CONDITION["A_min_cell_irrelevant"])
    if rq2_rows:
        all_flips = [row for rows in flip_groups.values() for row in rows]
        gains = [float(row["estimated_net_gain_ms"]) for row in all_flips]
        clean = sum(1 for value in gains if value > EPS_NET_GAIN_MS) / len(gains)
        bad = sum(1 for value in gains if abs(value) <= EPS_NET_GAIN_MS or value < 0) / len(gains)
        p10, p90 = percentile(gains, 10), percentile(gains, 90)
        numbers["B_n_flips"] = len(gains)
        numbers["B_share_strictly_positive"] = clean
        numbers["B_share_near_zero_or_negative"] = bad
        numbers["B_p10_net_gain_ms"] = p10
        numbers["B_p90_net_gain_ms"] = p90
        numbers["B_hold"] = bool(clean < CONDITION["B_max_clean_flip_share"] and bad >= CONDITION["B_min_neg_or_zero_share"])
    if rq3_rows:
        best_top20 = max(row["top20pct_share_of_positive_value"] for row in rq3_rows)
        pooled_values = [float(row["oracle_value_ms"]) for row in all_rows if row.get("oracle_value_ms") not in ("", None)]
        positives = sorted((value for value in pooled_values if value > 0), reverse=True)
        top20 = positives[:max(1, int(round(0.2 * len(positives))))]
        pooled_share = sum(top20) / sum(positives) if positives else float("nan")
        numbers["C_top20_share_best_cell"] = best_top20
        numbers["C_top20_share_pooled"] = pooled_share
        numbers["C_hold"] = bool(max(pooled_share, best_top20) >= CONDITION["C_min_top20_share"])
    # D: freshness/coverage vs value relations
    if all_rows:
        value_pairs = [(row, to_float(row.get("oracle_value_ms"))) for row in all_rows]
        value_pairs = [(row, value) for row, value in value_pairs if value is not None]
        rho_age = spearman([to_float(row.get("update_age_s"), 0.0) or 0.0 for row, _ in value_pairs],
                           [value for _, value in value_pairs])
        rho_cov = spearman([to_float(row.get("source_coverage_delta"), 0.0) or 0.0 for row, _ in value_pairs],
                           [value for _, value in value_pairs])
        numbers["D_spearman_age_vs_value"] = rho_age
        numbers["D_spearman_covdelta_vs_value"] = rho_cov
        numbers["D_hold"] = bool(abs(rho_age) < CONDITION["D_max_heuristic_spearman"] and abs(rho_cov) < CONDITION["D_max_heuristic_spearman"])
    # E: pooled predictability
    if pooled_rq4:
        def metric(label: str, feature_set: str, name: str) -> float:
            row = next((row for row in pooled_rq4 if row["label"] == label and row["feature_set"] == feature_set), {})
            value = row.get(name)
            return float(value) if value not in (None, "") else float("nan")
        da_auroc = metric("flip_next_use", "decision_aware", "auroc_loro_mean")
        fr_auroc = metric("flip_next_use", "freshness_only", "auroc_loro_mean")
        da_auprc = metric("flip_next_use", "decision_aware", "auprc_loro_mean")
        prev = metric("flip_next_use", "decision_aware", "prevalence")
        hv_auroc = metric("high_value_top25pct", "decision_aware", "auroc_loro_mean")
        hv_fr = metric("high_value_top25pct", "freshness_only", "auroc_loro_mean")
        numbers["E_flip_auroc_decision_aware"] = da_auroc
        numbers["E_flip_auroc_freshness_only"] = fr_auroc
        numbers["E_flip_auprc_decision_aware"] = da_auprc
        numbers["E_flip_prevalence"] = prev
        numbers["E_hold"] = bool((not math.isnan(da_auroc)) and da_auroc >= CONDITION["E_min_auroc"]
                                 and da_auroc >= fr_auroc + CONDITION["E_min_auroc_gain"]
                                 and da_auprc > prev * 1.5 and hv_auroc > hv_fr)
    conditions = {key: numbers.get(f"{key}_hold") for key in ("A", "B", "C", "D", "E")}
    holds = [value for value in conditions.values() if value is True]
    evaluated_conditions = [value for value in conditions.values() if value is not None]
    if len(holds) == len(evaluated_conditions) and len(holds) >= 4:
        verdict = "STRONGLY SUPPORTED"
    elif len(holds) >= 4:
        verdict = "SUPPORTED"
    elif len(holds) >= 3:
        verdict = "WEAKLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    numbers["conditions"] = conditions
    numbers["verdict"] = verdict

    # ---------------- outputs ----------------
    out_dir = results / AGG_DIR_NAME
    write_csv(out_dir / f"b03_update_counterfactuals_{tag}.csv", aggregate_rows)
    write_csv(out_dir / f"b03_update_horizons_{tag}.csv", all_horizon_rows)
    write_csv(out_dir / f"rq1_summary_{tag}.csv", rq1_rows)
    write_csv(out_dir / f"rq2_flip_value_{tag}.csv", rq2_rows)
    write_csv(out_dir / f"rq3_concentration_{tag}.csv", rq3_rows)
    write_csv(out_dir / f"rq4_predictability_{tag}.csv", rq4_rows + pooled_rq4)
    write_csv(out_dir / f"sanity_checks_b03_{tag}.csv", sanity_rows)
    (out_dir / f"report_numbers_{tag}.json").write_text(json.dumps(numbers, indent=2, default=str))

    figures(run_dir, tag, rq1_rows, rq2_rows, pareto_curves, all_rows, pooled_rq4)
    return numbers


def parse_loads(text) -> list[int]:
    try:
        return [int(part) for part in str(text).split(";") if part != ""]
    except ValueError:
        return []


def gini(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    if n == 0 or sum(values) == 0:
        return float("nan")
    total = sum((2 * index - n + 1) * value for index, value in enumerate(values, start=1))
    return total / (n * sum(values))


def figures(run_dir: Path, tag: str, rq1_rows, rq2_rows, pareto_curves, update_rows, pooled_rq4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "results" / FIG_DIR_NAME
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1 — decision redundancy by policy x rho
    if rq1_rows:
        labels = [f"{row['policy']}\nρ={row['rho']}" for row in rq1_rows]
        irrelevant = [float(row["decision_irrelevant_rate"]) for row in rq1_rows]
        flipping = [float(row["decision_flip_rate"]) for row in rq1_rows]
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.2))
        x = np.arange(len(labels))
        ax.bar(x, irrelevant, label="decision-irrelevant", color="#9bb7d4")
        ax.bar(x, flipping, bottom=irrelevant, label="decision-flipping", color="#e07b39")
        for index, value in enumerate(irrelevant):
            ax.text(index, value / 2, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("fraction of forwarded updates (next use)")
        ax.set_ylim(0, 1)
        ax.set_title("Fig.1  Decision redundancy of B02-forwarded updates")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig1_decision_redundancy_{tag}.png", dpi=150)
        plt.close(fig)

    # Figure 2 — flip vs actual value (CDF of net gain among flips)
    gains_by_group: dict[tuple, list[float]] = defaultdict(list)
    for row in update_rows:
        if row.get("evaluated") == 1 and int(row.get("decision_flip") or 0) == 1 and row.get("next_use_discard") in ("False", "0", "", None):
            value = to_float(row.get("estimated_net_gain_ms"))
            if value is not None:
                gains_by_group[group_key(row)].append(value)
    if gains_by_group:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, gains in sorted(gains_by_group.items())[:8]:
            ordered = np.sort(gains)
            ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                    label=f"{key[1]} ρ={key[2]}", lw=1.5)
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("estimated net dispatch gain of the flip (ms)")
        ax.set_ylabel("CDF over decision-flipping updates")
        ax.set_title("Fig.2  A decision flip does not imply a useful update")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig2_flip_vs_value_{tag}.png", dpi=150)
        plt.close(fig)

    # Figure 3 — value concentration (Pareto)
    if pareto_curves:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, (curve_x, curve_y) in sorted(pareto_curves.items())[:8]:
            ax.plot(curve_x, curve_y, label=f"{key[1]} ρ={key[2]}", lw=1.5)
        for fraction in (0.1, 0.2, 0.3):
            ax.axvline(fraction, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("fraction of updates (sorted by oracle value, desc)")
        ax.set_ylabel("cumulative share of total positive value")
        ax.set_title("Fig.3  Update value is concentrated (Pareto)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig3_value_concentration_{tag}.png", dpi=150)
        plt.close(fig)

    # Figure 4 — freshness/coverage/margin cannot fully identify value
    usable = [row for row in update_rows
              if row.get("evaluated") == 1 and to_float(row.get("oracle_value_ms")) is not None]
    if usable:
        panels = [("update_age_s", "update age at send (s)"),
                  ("source_coverage_delta", "coverage delta (tokens)"),
                  ("visible_cov_gap", "best-vs-second visible gap (tokens)")]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
        for ax, (feature, xlabel) in zip(axes, panels):
            pairs = [(to_float(row.get(feature)), float(row["oracle_value_ms"])) for row in usable]
            pairs = [(a, b) for a, b in pairs if a is not None]
            if not pairs:
                continue
            pairs.sort()
            values = [pair[0] for pair in pairs]
            targets = [pair[1] for pair in pairs]
            bins = np.quantile(values, np.linspace(0, 1, 9)) if len(set(values)) > 8 else np.linspace(min(values), max(values), 9)
            bins = np.unique(bins)
            centers, means, errs = [], [], []
            for low, high in zip(bins[:-1], bins[1:]):
                selected = [target for value, target in pairs if low <= value <= high]
                if selected:
                    centers.append((low + high) / 2)
                    means.append(float(np.mean(selected)))
                    errs.append(float(np.std(selected) / np.sqrt(len(selected))))
            ax.errorbar(centers, means, yerr=errs, fmt="o-", ms=3, lw=1, color="#2a6f97")
            ax.axhline(0.0, color="k", lw=0.7, ls="--")
            correlation = spearman(values, targets)
            ax.set_title(f"{xlabel}\nSpearman ρ={correlation:.2f}", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
        axes[0].set_ylabel("oracle value (ms)")
        fig.suptitle("Fig.4  Freshness / coverage / margin do not identify update value", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig4_heuristics_insufficient_{tag}.png", dpi=150)
        plt.close(fig)

    # Figure 5 — predictability
    flip_rows = [row for row in pooled_rq4 if row["label"] == "flip_next_use"]
    value_rows = [row for row in pooled_rq4 if row["label"] in ("positive_value", "high_value_top25pct")]
    if flip_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        order = [row for row in flip_rows]
        names = [row["feature_set"] for row in order]
        x = np.arange(len(order))
        axes[0].bar(x - 0.2, [to_float(row.get("auroc_loro_mean")) or 0 for row in order], width=0.4, label="AUROC", color="#2a6f97")
        axes[0].bar(x + 0.2, [to_float(row.get("auprc_loro_mean")) or 0 for row in order], width=0.4, label="AUPRC", color="#e07b39")
        prevalence = to_float(order[0].get("prevalence")) if order else None
        if prevalence is not None:
            axes[0].axhline(prevalence, color="grey", lw=0.8, ls="--", label=f"prevalence={prevalence:.2f}")
        axes[0].axhline(0.5, color="k", lw=0.8, ls=":")
        axes[0].set_xticks(x, names, rotation=20, fontsize=8)
        axes[0].set_title("predict decision_flip (next use)", fontsize=10)
        axes[0].legend(fontsize=8)
        labels_seen = sorted({row["label"] for row in value_rows})
        width = 0.8 / max(1, len(labels_seen))
        for index, label in enumerate(labels_seen):
            rows = [row for row in value_rows if row["label"] == label]
            offsets = (np.arange(len(rows)) - (len(labels_seen) - 1) / 2) * width
            axes[1].bar(offsets, [to_float(row.get("auroc_loro_mean")) or 0 for row in rows],
                        width=width, label=label)
            axes[1].set_xticks(np.arange(len(rows)), [row["feature_set"] for row in rows], rotation=20, fontsize=8)
        axes[1].axhline(0.5, color="k", lw=0.8, ls=":")
        axes[1].set_title("predict value labels (AUROC)", fontsize=10)
        axes[1].legend(fontsize=8)
        fig.suptitle("Fig.5  Pre-transmission features carry predictive signal", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig5_predictability_{tag}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="/home/byh/B03/b03_motivation/results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--j", type=int, default=4)
    parser.add_argument("--prefill-tokens-per-ms", type=float, default=50.0)
    parser.add_argument("--queue-penalty-ms", type=float, default=2.0)
    parser.add_argument("--guard-ms", type=float, default=0.5)
    args = parser.parse_args()
    numbers = analyze(Path(args.run_dir), args.tag, args.j, args.prefill_tokens_per_ms,
                      args.queue_penalty_ms, args.guard_ms)
    print(json.dumps(numbers, indent=2, default=str))


if __name__ == "__main__":
    main()
