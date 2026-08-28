#!/usr/bin/env bash
# Tear down ONLY the b03 shared-link platform (our containers + network).
# B03 copy of B02 net/teardown_net.sh; b02-net is never touched.
set -euo pipefail
for c in b03-gateway b03-bgserver; do
  if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
    docker rm -f "$c" >/dev/null
  fi
done
if docker network inspect b03-net >/dev/null 2>&1; then
  docker network rm b03-net >/dev/null
fi
echo "b03-net torn down (b03-gateway, b03-bgserver, network removed; image b03-gw kept)"
