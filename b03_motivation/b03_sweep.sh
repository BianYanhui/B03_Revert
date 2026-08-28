#!/usr/bin/env bash
# Sequential B03 sweep runs (Phase 2/3): alpha, overlap, conc presets.
# One harness at a time (they share the b03-net relay and the vLLM cluster).
# Each preset is analyzed immediately after its run; a pooled analysis over
# all motivation tags runs at the end.
set -euo pipefail

ROOT=/home/byh/B03
HERE="$ROOT/b03_motivation"
PY=/home/byh/B02/poc/.venv/bin/python
KV="${KV:-104544}"
TAGS_RUN=()

run_preset() {
  local preset="$1" tag="$2"; shift 2
  echo "=== B03 preset $preset -> tag $tag ($(date -Is)) ==="
  cd "$HERE"
  "$PY" run_b03_motivation.py --preset "$preset" --tag "$tag" \
      --kv-cache-tokens "$KV" --out-dir "$HERE" "$@"
  "$PY" analyze_b03.py --run-dir "$HERE" --tag "$tag" > "$HERE/results/aggregates/analyze_stdout_$tag.json" 2>&1 || true
  TAGS_RUN+=("$tag")
}

run_preset alpha   b03alpha
run_preset overlap b03overlap
run_preset conc    b03conc

echo "=== pooled analysis over ${TAGS_RUN[*]} + b03core ==="
cd "$HERE"
"$PY" analyze_b03.py --run-dir "$HERE" --tag "b03core,b03alpha,b03overlap,b03conc" \
    > "$HERE/results/aggregates/analyze_stdout_pooled.json" 2>&1 || true
echo "B03 sweep complete ($(date -Is))"
