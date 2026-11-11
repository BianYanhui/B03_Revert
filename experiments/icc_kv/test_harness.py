"""Offline checks for the semantic top-k harness. No Docker and no vLLM."""
from __future__ import annotations

import asyncio

from experiments.icc_kv.capacity_window import derive_capacity, window_problems
from experiments.icc_kv.e2e import gather_or_cancel, is_stale_cache_hit, note_worker_dispatch
from experiments.icc_kv.runtime import parse_tc_rate, require_clean_tree
from experiments.icc_kv.split_path import _offered_deadline, _sender_rate
from experiments.icc_kv.wire import (
    CFG, FRAME, POLICIES, POLICY_MAX_QUEUE, TOPK_SWEEP, config_frame, frame, read_freq,
)
from experiments.icc_kv.workload import burst_nominal_rho, interleave, noise_overlap, useful_slots


def check(name: str, ok: bool, detail: object = "") -> None:
    if not ok:
        raise SystemExit(f"FAIL {name}: {detail}")
    print(f"PASS {name}")


def test_policies() -> None:
    check("policy count", len(POLICIES) == 36, sorted(POLICIES))
    check("static alias", POLICIES["StaticTopK"] == POLICIES["StaticTopK16"] == (4, 1, 1, 0, 2, 16))
    check("sweep present", all(f"BoundedSemantic{k}" in POLICIES for k in TOPK_SWEEP))
    check("freq sweep present", all(f"StaticTopKFreq{k}" in POLICIES for k in TOPK_SWEEP))
    check("freq mode is 7", POLICIES["StaticTopKFreq16"][0] == 7, POLICIES["StaticTopKFreq16"])
    check(
        "freq differs from static only in mode",
        POLICIES["StaticTopKFreq16"][1:] == POLICIES["StaticTopK16"][1:],
        (POLICIES["StaticTopKFreq16"], POLICIES["StaticTopK16"]),
    )
    raw = config_frame(1, "BoundedFIFO16", max_queue=4096, max_inflight=0, rate_frames_per_s=0, rate_burst=1)
    decoded = CFG.unpack(raw[32:58])
    check("bounded cap on the wire", decoded[6] == 16 and decoded[0] == 6, decoded)
    check("semantic is mode 6 with merge", POLICIES["BoundedSemantic16"] == (6, 1, 1, 0, 2, 0))
    check("prio cap", POLICY_MAX_QUEUE["BoundedPrio32"] == 32)


def test_mix() -> None:
    slots = useful_slots(6, "1024:1,2048:1,4096:1")
    check("useful round robin", slots == [1024, 2048, 4096, 1024, 2048, 4096], slots)
    ov60 = "256:40,1024:20,2048:20,4096:20"
    table = interleave(ov60)
    check("ov60 reduced length", len(table) == 5 and table.count(256) == 2, table)
    check("ov60 overlap", abs(noise_overlap(ov60, slots) - 0.6) < 1e-9, noise_overlap(ov60, slots))
    check("base overlap is zero", noise_overlap("", [4096] * 16) == 0.0)
    check("burst nominal", abs(burst_nominal_rho(0.9, 5.0) - 1.5) < 1e-9)


def test_tc() -> None:
    sample = "class htb 1:1 root rate 10Mbit ceil 10Mbit burst 1600b cburst 1600b"
    check("tc 10Mbit", parse_tc_rate(sample) == 10_000_000, parse_tc_rate(sample))
    check("tc missing", parse_tc_rate("no class") == 0)


def test_window() -> None:
    rows = []
    for rate, forwarded, queued, cpu, gap in (
        (6000, 6000, 0, 0.4, 0.0),
        (12000, 9000, 100, 0.5, 0.0),
        (13000, 9100, 200, 0.6, 0.0),
        (24000, 6500, 400000, 0.6, 0.0),
    ):
        rows.append({
            "policy": "FullSync", "sent_per_s": rate, "forwarded_per_s": forwarded,
            "queued_end": queued, "gateway_cpu": cpu, "ledger_ingress_gap": gap,
        })
    rows.append({
        "policy": "StaticTopK16", "sent_per_s": 20000, "forwarded_per_s": 9000,
        "queued_end": 16, "gateway_cpu": 0.7, "ledger_ingress_gap": 0.0,
    })
    derived = derive_capacity({"link_bit_s": 10_000_000, "rows": rows})
    check("drain plateau", abs(derived["c_drain_per_s"] - 9050) < 1, derived)
    check("ingress min", derived["c_ingress_min_per_s"] == 20000, derived)
    check(
        "excluded row does not affect c_drain",
        derived["c_drain_rows"] == 2 and abs(derived["c_drain_per_s"] - 9050) < 1,
        derived,
    )
    problems = window_problems(derived, link=10_000_000, capacity=9050, peaks={"ultrahigh": 20000})
    check("window rejects a peak above ingress", bool(problems), problems)


def test_cancel() -> None:
    cancelled = 0

    async def boom() -> None:
        raise RuntimeError("serve failed")

    async def linger() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled += 1
            raise

    async def run() -> None:
        try:
            await gather_or_cancel([boom(), linger(), linger()])
        except RuntimeError:
            return
        raise SystemExit("gather did not surface the failure")

    asyncio.run(run())
    check("cancelled siblings", cancelled == 2, cancelled)


def test_stale_predicate() -> None:
    """Sibling reuse is not leftover KV. The first hit of a version still is."""
    seen: set[tuple[int, int, int]] = set()

    def hit(cached: int, truth: int, version: int, slot: int, worker: int) -> bool:
        earlier = note_worker_dispatch(seen, slot, version, worker)
        return is_stale_cache_hit(cached, truth, version, earlier)

    check("first hit of a version is stale", hit(4128, 0, 2, 3, 1) is True)
    check("later sibling on that worker is not", hit(4128, 0, 2, 3, 1) is False)
    check("same version on another worker still is", hit(512, 0, 2, 3, 0) is True)
    check("other slot on the same worker still is", hit(512, 0, 2, 4, 1) is True)
    check("version 0 is not stale", hit(4128, 0, 0, 3, 1) is False)
    check("routed truth blocks the count", hit(4128, 4096, 1, 5, 2) is False)
    check("cached below 512 is not stale", hit(511, 0, 1, 6, 2) is False)
    check("new version on the same worker counts again", hit(4128, 0, 3, 3, 1) is True)
    check("second request of the new version does not", hit(4128, 0, 3, 3, 1) is False)


def test_offered_deadline() -> None:
    """Absolute offer times track level, including the 25 s / 5 s burst window."""
    t0 = 1000.0
    level, senders = 4000.0, 4
    rate = level / senders
    for sent, offset in ((0, 0.0), (1, 1.0 / rate), (rate, 1.0), (25 * rate, 25.0)):
        got = _offered_deadline(t0, int(sent), level, senders, "xhigh", 5.0)
        check(f"steady deadline sent={sent}", abs(got - (t0 + offset)) < 1e-9, got)
    check(
        "steady rate ignores the burst window",
        _sender_rate(level, senders, "xhigh", 5.0, 27.0) == rate,
    )
    burst_mult = 1.5
    base = _sender_rate(level, senders, "burst", burst_mult, 24.999)
    burst = _sender_rate(level, senders, "burst", burst_mult, 25.0)
    check("burst base window", base == rate and _sender_rate(level, senders, "burst", burst_mult, 0.0) == rate)
    check("burst high window", burst == rate * burst_mult)
    check("burst window repeats", _sender_rate(level, senders, "burst", burst_mult, 30.0) == rate)
    per_cycle = base * 25.0 + burst * 5.0
    end = _offered_deadline(t0, int(per_cycle), level, senders, "burst", burst_mult)
    check("burst cycle ends at 30s", abs(end - (t0 + 30.0)) < 1e-6, end)
    at_25 = _offered_deadline(t0, int(base * 25.0), level, senders, "burst", burst_mult)
    check("burst base chunk ends at 25s", abs(at_25 - (t0 + 25.0)) < 1e-6, at_25)
    one_into_burst = _offered_deadline(t0, int(base * 25.0) + 1, level, senders, "burst", burst_mult)
    check(
        "first burst event is on the multiplied rate",
        abs(one_into_burst - (t0 + 25.0 + 1.0 / burst)) < 1e-6,
        one_into_burst,
    )
    # Oversleep does not move the target for a given count, so the next wait
    # shortens by the time actually lost. 1000 events at `rate` are due at 1s.
    now = t0
    sent = 0
    batch = max(1, int(rate / 200))
    while sent < int(rate):
        target = _offered_deadline(t0, sent, level, senders, "xhigh", 5.0)
        now = target + 0.001 if now < target else now
        sent += batch
        now += 0.0002
    check("oversleep stays on the schedule", now - t0 < 1.05, now - t0)


def test_dirty_allowed() -> None:
    dirty = require_clean_tree(allow_dirty=True)
    check("dirty list", isinstance(dirty, list), dirty)


def test_frame_freq() -> None:
    """The frequency rides in payload bytes that already existed.

    A sender that never sets it must produce a byte-identical frame to the one
    the old code produced, so an old sender is admitted exactly as before.
    """
    import time as _time
    stamp = 1_700_000_000.0
    plain = frame(1, 3, 7, 11, 4096, 0xDEADBEEF, stamp)
    explicit_zero = frame(1, 3, 7, 11, 4096, 0xDEADBEEF, stamp, freq=0)
    check("frame stays 64 bytes", len(plain) == FRAME == 64, len(plain))
    check("freq=0 is byte-identical to no freq", plain == explicit_zero)
    check("read_freq on an old frame is 0", read_freq(plain) == 0, read_freq(plain))
    tagged = frame(1, 3, 7, 11, 4096, 0xDEADBEEF, stamp, freq=38)
    check("freq round-trips", read_freq(tagged) == 38, read_freq(tagged))
    check("payload grew the frame by nothing", len(tagged) == 64, len(tagged))
    check("freq does not disturb the header",
          tagged[:32] == plain[:32])
    capped = frame(1, 3, 7, 11, 4096, 0xDEADBEEF, stamp, freq=10**9)
    check("freq is capped", read_freq(capped) == 0xFFFF, read_freq(capped))


def test_freq_scoring() -> None:
    """MODE_FREQ multiplies; every other mode must ignore the field entirely.

    This is the compatibility guarantee: a cell that is not running MODE_FREQ
    produces the same admission decisions it did before the field existed.
    """
    from types import SimpleNamespace
    gateway = relay_module()
    relay = gateway.Relay(SimpleNamespace(max_queue=4096, gate=2.0, adaptive_queue_gate=8,
                                          tau=30.0, util_lambda=16.0))
    relay.emit_update = lambda *args, **kwargs: None
    relay.current_cell = 1

    relay.mode = gateway.MODE_STATIC
    check("static ignores freq", relay._topk_score(1024, 38) == 1024, relay._topk_score(1024, 38))
    relay.mode = gateway.MODE_BOUNDED
    check("bounded ignores freq", relay._topk_score(1024, 38) == 1024)
    relay.mode = gateway.MODE_FREQ
    check("freq multiplies", relay._topk_score(1024, 38) == 1024 * 38, relay._topk_score(1024, 38))
    check("absent freq reads as 1", relay._topk_score(1024, 0) == 1024, relay._topk_score(1024, 0))

    # The ranking itself: a short hot prefix must beat a long cold one.
    relay.global_topk = 16
    relay.topk_best.clear()
    relay.topk_refs.clear()
    relay._topk_add(1, relay._topk_score(4096, 1))     # long, cold
    relay._topk_add(2, relay._topk_score(1024, 38))    # short, hot
    check("hot short outranks cold long",
          relay.topk_best[2] > relay.topk_best[1],
          dict(relay.topk_best))
    worst = relay._topk_worst()
    check("worst is the cold long prefix", worst[0] == 1, worst)


def test_freq_no_regression_on_static_cells() -> None:
    """With every sender claiming nothing, MODE_FREQ must equal MODE_STATIC.

    This is the strongest compatibility check: turn the feature on and leave the
    senders alone, and the decisions cannot change.
    """
    from types import SimpleNamespace
    gateway = relay_module()
    args = SimpleNamespace(max_queue=4096, gate=2.0, adaptive_queue_gate=8,
                           tau=30.0, util_lambda=16.0)

    def admit_sequence(mode: int) -> list[int]:
        relay = gateway.Relay(args)
        relay.emit_update = lambda *a, **k: None
        relay.current_cell = 1
        relay.mode = mode
        relay.global_topk = 4
        relay.merge = True
        relay.priority = True
        relay.dedup = 2
        kept: list[int] = []
        # A long cold prefix, then several short ones with no frequency claim.
        for index, coverage in enumerate([4096, 1024, 1024, 1024, 1000, 900]):
            digest = 100 + index
            data = gateway.frame(gateway.K_UP, 0, 1, index, coverage, digest,
                                 _now(), freq=0)
            before = relay.drops["global_topk"]
            relay.enqueue(data, gateway.K_UP, 0, index, coverage, digest, _now())
            if relay.drops["global_topk"] == before:
                kept.append(digest)
        return kept

    static_kept = admit_sequence(gateway.MODE_STATIC)
    freq_kept = admit_sequence(gateway.MODE_FREQ)
    check("no-claim senders behave identically", static_kept == freq_kept,
          (static_kept, freq_kept))


def test_freq_recovers_short_hot_prefixes() -> None:
    """The behaviour the whole change exists for.

    A 1024-token prefix served five times must outrank a 4096-token prefix seen
    once. Under MODE_STATIC it does not, so the short hot prefix is evicted by
    long background traffic; under MODE_FREQ it survives.
    """
    gateway = relay_module()
    from types import SimpleNamespace
    args = SimpleNamespace(max_queue=4096, gate=2.0, adaptive_queue_gate=8,
                           tau=30.0, util_lambda=16.0)

    useful = [(1024, 100, 5), (1024, 101, 4), (2048, 102, 3), (4096, 103, 2)]
    items: list[tuple[int, int, int]] = []
    for coverage, digest, reps in useful:
        for rep in range(1, reps + 1):
            items.append((coverage, digest, rep))
    # 48 unique noise frames, 20% of them as long as the longest useful prefix.
    for n in range(48):
        coverage = 4096 if n % 5 == 0 else (2048 if n % 5 == 1 else (1024 if n % 5 == 2 else 256))
        items.append((coverage, 5000 + n, 1))

    def retained(mode: int) -> set[int]:
        relay = gateway.Relay(args)
        relay.emit_update = lambda *a, **k: None
        relay.current_cell = 1
        relay.mode = mode
        relay.global_topk = 16
        relay.merge = True
        relay.priority = True
        relay.dedup = 2
        stamp = _now()
        for index, (coverage, digest, freq) in enumerate(items):
            relay.enqueue(
                gateway.frame(gateway.K_UP, 0, 1, index, coverage, digest, stamp, freq=freq),
                gateway.K_UP, 0, index, coverage, digest, stamp,
            )
        return set(relay.topk_best)

    wanted = {digest for _coverage, digest, _reps in useful}
    static_kept = retained(gateway.MODE_STATIC) & wanted
    freq_kept = retained(gateway.MODE_FREQ) & wanted
    check("coverage-only loses the short hot prefixes",
          static_kept == {102, 103}, sorted(static_kept))
    check("frequency keeps all four",
          freq_kept == wanted, sorted(freq_kept))


def _now() -> float:
    import time as _time
    return _time.time()


def relay_module():
    """The gateway module, loaded by path the way test_policies does it."""
    import importlib.util
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent / "4t4"
    spec = importlib.util.spec_from_file_location(
        "gateway4t4_harness", here / "net" / "gateway_relay_4t4.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    test_policies()
    test_mix()
    test_tc()
    test_window()
    test_cancel()
    test_stale_predicate()
    test_offered_deadline()
    test_dirty_allowed()
    test_frame_freq()
    test_freq_scoring()
    test_freq_no_regression_on_static_cells()
    test_freq_recovers_short_hot_prefixes()
    check("frame size", FRAME == 64)
    print("all passed")


if __name__ == "__main__":
    main()
