#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS_DIR=${1:-$(mktemp -d /tmp/flex_local_map_validation.XXXXXX)}
PYTHON_BIN=${PYTHON:-python}
TORCHRUN_BIN=${TORCHRUN:-torchrun}
BASE_REF=${BASE_REF:-origin/main}

mkdir -p "$RESULTS_DIR/source_snapshots"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

run_stage() {
  local name=$1
  shift
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } | tee "$RESULTS_DIR/${name}.command.txt"
  set +e
  "$@" 2>&1 | tee "$RESULTS_DIR/${name}.log"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" > "$RESULTS_DIR/${name}.exit_code"
  return "$status"
}

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  git branch --show-current
  git rev-parse HEAD
  git rev-parse "$BASE_REF" 2>/dev/null || printf 'unavailable: %s\n' "$BASE_REF"
  git status --short
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version())'
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
} > "$RESULTS_DIR/environment.txt"
if git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  git diff --binary "$BASE_REF"...HEAD > "$RESULTS_DIR/committed_vs_base.patch"
fi
git diff --binary > "$RESULTS_DIR/working_tree.patch"
git status --short > "$RESULTS_DIR/git_status.txt"
cp \
  tests/compare_flex_local_map_validation.py \
  examples/example_ds3_local_map.py \
  tests/test_flex_local_map_e2e.py \
  tests/test_optimize_placement.py \
  tests/test_placement_options_utils.py \
  "$RESULTS_DIR/source_snapshots/"
sha256sum "$RESULTS_DIR"/source_snapshots/* > "$RESULTS_DIR/source_checksums.txt"

"$PYTHON_BIN" - "$RESULTS_DIR/execution_config.json" <<'PY'
import json
import sys

config = {
    "world_size": 4,
    "backend": "nccl",
    "pointwise_mesh": [4],
    "moe_mesh": [2, 2],
    "moe_mesh_names": ["dp", "ep"],
    "rng_seed": 1,
    "sequence_length": 2048,
    "local_batch_size": 8,
    "microbatches": 16,
    "dtype": "bfloat16",
    "reduce_dtype": "float32",
    "cases": [
        "plain-sharded",
        "plain-replicated-counts",
        "flex-default",
        "flex-replicated-counts",
    ],
}
with open(sys.argv[1], "w") as output:
    json.dump(config, output, indent=2)
    output.write("\n")
PY

echo "Results directory: $RESULTS_DIR"

run_stage compile \
  "$PYTHON_BIN" -m py_compile \
  tests/test_flex_local_map_e2e.py \
  examples/example_ds3_local_map.py \
  tests/compare_flex_local_map_validation.py \
  tests/test_optimize_placement.py \
  tests/test_placement_options_utils.py

run_stage placement_options \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_placement_options_utils.py::TestFlexLocalMapPlacementOptions

run_stage optimize_placement \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_optimize_placement.py -k flex_local_map

run_stage serialization \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_serialization.py -k flex

run_stage pointwise_4gpu \
  "$PYTHON_BIN" -m pytest -q tests/test_flex_local_map_e2e.py -v

for case in plain-sharded plain-replicated-counts flex-default flex-replicated-counts; do
  run_stage "moe_${case}" \
    "$TORCHRUN_BIN" --standalone --nproc-per-node 4 \
    examples/example_ds3_local_map.py \
    --local-map-case "$case" \
    --rng-seed 1 \
    --logs-dir "$RESULTS_DIR/$case"
done

run_stage numerics \
  "$PYTHON_BIN" tests/compare_flex_local_map_validation.py "$RESULTS_DIR"

echo "$RESULTS_DIR"
