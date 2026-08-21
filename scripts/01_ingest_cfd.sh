#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <CFD_DATA_PATH> <DATASET>" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

CFD_DATA_PATH=$1
DATASET=$2

if [[ ! -e "$CFD_DATA_PATH" ]]; then
  echo "Error: CFD data path does not exist: $CFD_DATA_PATH" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
export CFD_BENCH_DATA_ROOT="$PROJECT_ROOT/data"
mkdir -p "$CFD_BENCH_DATA_ROOT"

exec cfd-bench ingest \
  --dat "$CFD_DATA_PATH" \
  --datasets "$DATASET" \
  --backends postgresql iotdb tiledb vtk \
  --iotdb-host "${CFD_BENCH_IOTDB_HOST:-127.0.0.1}" \
  --iotdb-port "${CFD_BENCH_IOTDB_PORT:-6667}" \
  --iotdb-user "${CFD_BENCH_IOTDB_USER:-root}" \
  --iotdb-password "${CFD_BENCH_IOTDB_PASSWORD:-root}"
