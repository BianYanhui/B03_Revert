"""64-byte state frames shared with the frozen 4×T4 gateway relay."""
from __future__ import annotations

import struct
import time

FRAME = 64
HDR = struct.Struct(">BBHIqQd")
CFG = struct.Struct(">BBBBHIIIII")
STATS = struct.Struct(">IIIIIIII")
STATS2 = struct.Struct(">IIIIIIII")
K_UP, K_TOMB, K_RESET, K_STATS_REQ, K_CONFIG, K_ACK, K_STATS, K_RESET_DONE, K_STATS2 = 1, 2, 3, 4, 5, 6, 7, 8, 9
WIRE_BYTES = 104

# A K_UP may carry how many times its sender has already served this prefix in
# this cell, so admission can rank by coverage*frequency instead of coverage
# alone. The field rides in the 32 bytes every frame already reserves and that
# nothing read before, so the wire length stays 64. A sender that leaves it zero
# is ranked exactly as it is today.
FREQ = struct.Struct(">I")
FREQ_MAX = 0xFFFF

# mode, merge, priority, adaptive, dedup, global_topk
# StaticTopK keeps the k highest-coverage digests. Bounded* caps frames.
# "StaticTopK" is the k=16 alias used by the paper's main method.
# StaticTopKFreq ranks by coverage*frequency instead of coverage alone.
TOPK_SWEEP = (4, 8, 16, 32, 64)
POLICIES: dict[str, tuple[int, int, int, int, int, int]] = {
    "FullSync": (0, 0, 0, 0, 0, 0),
    "RateFIFO": (1, 0, 0, 0, 0, 0),
    "StaticSemantic": (4, 1, 1, 0, 2, 0),
    "StaticTopK": (4, 1, 1, 0, 2, 16),
    "StaticTopK16NoMerge": (4, 0, 1, 0, 2, 16),
    "StaticTopK16NoDedup": (4, 1, 1, 0, 0, 16),
    "StaticTopK16NoPriority": (4, 1, 0, 0, 2, 16),
    "Adaptive": (5, 1, 1, 1, 2, 16),
    "AdaptiveNoPriority": (5, 1, 0, 1, 2, 16),
    "Ideal": (0, 0, 0, 0, 0, 0),
    "BoundedFIFO4096": (6, 0, 0, 0, 0, 0),
}
POLICY_MAX_QUEUE: dict[str, int] = {"BoundedFIFO4096": 4096}
for _k in TOPK_SWEEP:
    POLICIES[f"StaticTopK{_k}"] = (4, 1, 1, 0, 2, _k)
    POLICIES[f"StaticTopKFreq{_k}"] = (7, 1, 1, 0, 2, _k)
    POLICIES[f"BoundedFIFO{_k}"] = (6, 0, 0, 0, 0, 0)
    POLICIES[f"BoundedPrio{_k}"] = (6, 0, 1, 0, 0, 0)
    POLICIES[f"BoundedSemantic{_k}"] = (6, 1, 1, 0, 2, 0)
    POLICY_MAX_QUEUE[f"BoundedFIFO{_k}"] = _k
    POLICY_MAX_QUEUE[f"BoundedPrio{_k}"] = _k
    POLICY_MAX_QUEUE[f"BoundedSemantic{_k}"] = _k


def frame(kind: int, instance: int, cell: int, seq: int, coverage: int, digest: int, t: float, payload: bytes = b"", freq: int = 0) -> bytes:
    """One 64-byte frame. `freq` occupies the first four payload bytes.

    It is only meaningful on K_UP; other kinds leave it zero, and a zero reads
    back as "no claim", which admission treats as frequency 1.
    """
    if freq:
        payload = FREQ.pack(min(int(freq), FREQ_MAX)) + payload[:28]
    return HDR.pack(kind, instance & 0xFF, cell & 0xFFFF, seq & 0xFFFFFFFF, coverage, digest, t) + payload.ljust(32, b"\x00")[:32]


def read_freq(data: bytes) -> int:
    """Frequency carried by a received frame, 0 when the sender claimed nothing."""
    if len(data) < 36:
        return 0
    return FREQ.unpack(data[32:36])[0]


def config_frame(cell: int, policy: str, *, max_queue: int, max_inflight: int, rate_frames_per_s: float, rate_burst: int) -> bytes:
    mode, merge, priority, adaptive, dedup, topk = POLICIES[policy]
    max_queue = POLICY_MAX_QUEUE.get(policy, max_queue)
    payload = CFG.pack(
        mode, merge, priority, adaptive, dedup, topk,
        max_queue, max_inflight, int(rate_frames_per_s * 1000), rate_burst,
    )
    return frame(K_CONFIG, 0, cell, 0, 0, 0, time.time(), payload)
