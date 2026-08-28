# b03-net: dedicated B03 networking platform

Copy-with-renames of B02 `shared_link_exp/net/` (provenance: B02_LIS
commit `c917c25`).  The serving/tc design is unchanged; every name is
B03-scoped so the B02 platform (`b02-net`) can run on the same host
without interference.

| resource | B02 | B03 (here) |
|---|---|---|
| docker network | `b02-net` 172.30.0.0/24 | `b03-net` 172.32.0.0/24 |
| image | `b02-gw` | `b03-gw` |
| relay container | `gateway` | `b03-gateway` |
| iperf3 background server | `bgserver` | `b03-bgserver` |
| relay published on host | 127.0.0.1:9700 | 127.0.0.1:9702 |
| dispatcher endpoint | bridge IP:9701 | bridge IP:9703 |
| tc signaling filter | dport 9701 | dport 9703 |

`gateway_relay.py` inside the image is UNCHANGED (it takes `--listen` /
`--downstream` as arguments).  `Dockerfile` is unchanged.

Files:
- `setup_net.sh` — build network/image/containers/tc (idempotent)
- `teardown_net.sh` — remove ONLY b03 resources
- `cell_rate.sh` — per-cell HTB rate (`--sig-bit`)
- `tc_stats.sh` — tc counters snapshot (parsed by the harness)
- `restart_t4_v3_b03.sh` — vLLM cluster restart (T4 x 4; logs + pids under
  `/home/byh/B03/shared_link_exp/server_logs`; vLLM binary re-used READ-ONLY
  from the B02 venv)
