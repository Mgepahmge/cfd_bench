#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <DATASET> [DURATION=10] [OUTPUT_CSV]" >&2
}

if (( $# < 1 || $# > 3 )); then
  usage
  exit 2
fi

DATASET=$1
DURATION=${2:-10}
OUTPUT_CSV=${3:-}

if ! [[ "$DURATION" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || [[ "$DURATION" == "0" ]] || [[ "$DURATION" == "0.0" ]]; then
  echo "Error: duration must be a positive number: $DURATION" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
export CFD_BENCH_DATA_ROOT="$PROJECT_ROOT/data"
mkdir -p "$CFD_BENCH_DATA_ROOT"

CMD=(
  cfd-bench run
  --backend postgresql iotdb tiledb vtk
  --workloads w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11
  --duration "$DURATION"
  --datasets "$DATASET"
)

if [[ -n "$OUTPUT_CSV" ]]; then
  CMD+=(--output "$OUTPUT_CSV")
fi

exec "${CMD[@]}"
