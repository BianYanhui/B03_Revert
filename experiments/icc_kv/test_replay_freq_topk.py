"""Offline checks for the frequency-weighted top-k replay. No Docker, no vLLM."""
from __future__ import annotations

from experiments.icc_kv.replay_freq_topk import (
    CACHE_BAR,
    parse_mix,
    replay_requests,
    score_of,
    survival_probability,
)

OV60 = "256:40,1024:20,2048:20,4096:20"
OV30 = "256:70,1024:10,2048:10,4096:10"
BASE = ""


def check(name: str, ok: bool, detail: object = "") -> None:
    if not ok:
        raise SystemExit(f"FAIL {name}: {detail}")
    print(f"PASS {name}")


def test_mix() -> None:
    table = parse_mix(OV60)
    check("ov60 length", len(table) == 5, table)
    check("ov60 composition", table.count(256) == 2 and table.count(4096) == 1, table)
    check("ov30 length", len(parse_mix(OV30)) == 10, parse_mix(OV30))
    check("empty mix falls back to 256", parse_mix(BASE) == [256], parse_mix(BASE))


def test_scoring() -> None:
    check("coverage ignores count", score_of("coverage", 1024, 7) == 1024.0)
    check("cov_x_n grows linearly", score_of("cov_x_n", 1024, 5) == 5120.0)
    check("cov_x_log2 compresses", score_of("cov_x_log2", 1024, 7) == 3072.0)
    check("cov_x_sqrt compresses", abs(score_of("cov_x_sqrt", 1024, 4) - 2048.0) < 1e-9)
    try:
        score_of("nope", 1024, 1)
    except SystemExit:
        check("unknown scoring rejected", True)
    else:
        check("unknown scoring rejected", False)


def test_immunity_thresholds() -> None:
    """A digest is safe only above the longest noise class, and that class has
    to be present in the cell's mix. `coverage` alone never reaches it on
    ov30/ov60, which is the whole point of the change."""
    table = parse_mix(OV60)
    # A 1024 ad only clears the 256 class, so it sits in the uncertain band.
    check("1024 at n=1 is uncertain", survival_probability(1024.0, table, "coverage") is None)
    check("256-class ad is never safe", survival_probability(256.0, table, "coverage") == 0.0)
    check("4096 at n=1 still ties", survival_probability(4096.0, table, "coverage") is None)
    check("1024 at n=5 is immune", survival_probability(5120.0, table, "coverage") == 1.0)
    check("2048 at n=2 still ties", survival_probability(4096.0, table, "coverage") is None)
    check("2048 at n=3 is immune", survival_probability(6144.0, table, "coverage") == 1.0)
    check("4096 at n=2 is immune", survival_probability(8192.0, table, "coverage") == 1.0)
    # The base workload's noise is all 256, so even one ad clears it.
    check(
        "base 1024 is immune at n=1",
        survival_probability(1024.0, parse_mix(BASE), "coverage") == 1.0,
    )


def test_counts_advance_per_slot() -> None:
    """The count is per prefix, not global: slot 0 recurring must not raise
    slot 1's score."""
    requests = [
        {"slot": "0", "coverage_tokens": "1024", "loose_false_negative": "0",
         "reused_tokens": "1056", "prefill_tokens": "32"},
        {"slot": "1", "coverage_tokens": "1024", "loose_false_negative": "0",
         "reused_tokens": "1056", "prefill_tokens": "32"},
        {"slot": "0", "coverage_tokens": "1024", "loose_false_negative": "0",
         "reused_tokens": "1056", "prefill_tokens": "32"},
    ]
    out = replay_requests(
        requests, "cov_x_n", parse_mix(OV60),
        scenario="ultrahigh", seed=0, method="StaticTopK16", maxq=18,
    )
    check("requests counted", out["requests"] == 3, out)
    check("digests counted per slot", out["digests"] == 2, out)
    # n=1 -> 1024, n=2 -> 2048: both below the 4096 noise ceiling.
    check("no early immunity on ov60", out["immune"] == 0, out)
    check("all at risk", out["at_risk"] == 3, out)


def test_rescue_only_counts_fn() -> None:
    """A request that was not a false negative is never reported as rescued."""
    requests = [
        {"slot": "0", "coverage_tokens": "1024", "loose_false_negative": "0",
         "reused_tokens": "1056", "prefill_tokens": "32"}
        for _ in range(6)
    ]
    out = replay_requests(
        requests, "cov_x_n", parse_mix(OV60),
        scenario="ultrahigh", seed=0, method="StaticTopK16", maxq=18,
    )
    check("six ads make it immune", out["immune"] >= 1, out)
    check("no FN means no rescue", out["rescued_fn"] == 0, out)


def test_rescue_counts_real_miss_only() -> None:
    """`rescued_miss` must exclude the FN that still hit the cache, because
    those requests did not actually pay a full prefill."""
    requests = [
        {"slot": "0", "coverage_tokens": "1024", "loose_false_negative": "0",
         "reused_tokens": "1056", "prefill_tokens": "32"}
        for _ in range(5)
    ]
    requests.append({
        "slot": "0", "coverage_tokens": "1024", "loose_false_negative": "1",
        "reused_tokens": str(CACHE_BAR), "prefill_tokens": "32",
    })
    out = replay_requests(
        requests, "cov_x_n", parse_mix(OV60),
        scenario="ultrahigh", seed=0, method="StaticTopK16", maxq=18,
    )
    check("FN rescued", out["rescued_fn"] == 1, out)
    check("but not a real miss", out["rescued_miss"] == 0, out)
    check("so no prefill credited", out["rescued_fn_prefill"] == 0, out)


def test_control_arm_never_rescues() -> None:
    """The rule the cells actually ran must not claim a win over itself."""
    requests = [
        {"slot": str(index % 4), "coverage_tokens": "1024",
         "loose_false_negative": "1" if index % 3 == 0 else "0",
         "reused_tokens": "32", "prefill_tokens": "1000"}
        for index in range(30)
    ]
    out = replay_requests(
        requests, "coverage", parse_mix(OV60),
        scenario="ultrahigh", seed=0, method="StaticTopK16", maxq=18,
    )
    check("control arm rescues nothing", out["rescued_fn"] == 0, out)
    check("control arm has no immune ads", out["immune"] == 0, out)


def main() -> None:
    test_mix()
    test_scoring()
    test_immunity_thresholds()
    test_counts_advance_per_slot()
    test_rescue_only_counts_fn()
    test_rescue_counts_real_miss_only()
    test_control_arm_never_rescues()
    print("all passed")


if __name__ == "__main__":
    main()
