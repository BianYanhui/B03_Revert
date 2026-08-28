#!/usr/bin/env bash
# End-to-end B03 Phase-1 smoke:
#   1. vLLM cluster up at MED KV pressure (restart if unhealthy)
#   2. b03-net docker platform up (network, image, gateway+bgserver, tc)
#   3. run_b03_motivation.py --preset smoke (1 rep x 48 requests; ideal +
#      agg_full x rho {0.8, 1.3}) with built-in instrumentation assertions
#   4. analyze_b03.py on the smoke tag (offline counterfactual evaluation +
#      World-1 replay sanity)
# All artifacts stay under /home/byh/B03/b03_motivation/.
set -euo pipefail

ROOT=/home/byh/B03
HERE="$ROOT/b03_motivation"
PY=/home/byh/B02/poc/.venv/bin/python
SMOKE_TAG="${SMOKE_TAG:-b03smoke-$(date +%Y%m%d%H%M%S)}"

parse_kv_capacity() {
  sed -nE 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/p' \
    "$ROOT/shared_link_exp/server_logs/vllm_0.log" | tail -1 | tr -d ','
}

# 1. vLLM at MED pressure
READY=0
for PORT in 8000 8001 8002 8003; do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && READY=$((READY + 1))
done
if ((READY == 4)); then
  KV=$(parse_kv_capacity)
  echo "vllm already healthy (KV_CACHE_TOKENS=$KV)"
else
  "$HERE/net_b03/restart_t4_v3_b03.sh" MED
  KV=$(parse_kv_capacity)
fi
[ -n "$KV" ] || { echo "could not parse GPU KV cache size" >&2; exit 1; }

# 2. docker + tc platform (b03-net only; b02-net untouched)
"$HERE/net_b03/setup_net.sh"

# 3. instrumented harness smoke (assertions inside; exits nonzero on failure)
cd "$HERE"
"$PY" run_b03_motivation.py \
  --preset smoke --tag "$SMOKE_TAG" --kv-cache-tokens "$KV" \
  --out-dir "$HERE"

# 4. offline counterfactual analysis + replay sanity
"$PY" analyze_b03.py --run-dir "$HERE" --tag "$SMOKE_TAG"

echo "B03 smoke complete: tag=$SMOKE_TAG"
