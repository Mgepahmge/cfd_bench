#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <H5_DATA_PATH> <DATASET>" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

H5_DATA_PATH=$1
DATASET=$2

if [[ ! -f "$H5_DATA_PATH" ]]; then
  echo "Error: H5 data file does not exist: $H5_DATA_PATH" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
export CFD_BENCH_DATA_ROOT="$PROJECT_ROOT/data"
mkdir -p "$CFD_BENCH_DATA_ROOT"

exec cfd-bench ingest-h5 \
  --h5 "$H5_DATA_PATH" \
  --datasets "$DATASET" \
  --backends postgresql iotdb tiledb vtk
