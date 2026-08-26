#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage:
  $0 <DATASET> [WORKLOAD ...]

WORKLOAD candidates:
  2 4 5 6 8

Examples:
  $0 test001_default
  $0 test001_default 2
  $0 test001_default 2 5 8
  $0 test001_default w2 w5 w8

If WORKLOAD is omitted, the default is:
  W2 W4 W5 W6 W8
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

DATASET=$1
shift

DURATION=10

# ----------------------------------------------------------------------
# Workload selection
# ----------------------------------------------------------------------

DEFAULT_WORKLOADS=(2 4 5 6 8)
SELECTED_WORKLOADS=()

if [[ $# -eq 0 ]]; then
  SELECTED_WORKLOADS=("${DEFAULT_WORKLOADS[@]}")
else
  for raw in "$@"; do
    # Accept:
    #   2
    #   w2
    #   W2
    workload="${raw#w}"
    workload="${workload#W}"

    case "$workload" in
      2|4|5|6|8)
        ;;
      *)
        echo "Error: invalid workload '$raw'." >&2
        echo "Allowed workloads: 2 4 5 6 8" >&2
        exit 2
        ;;
    esac

    # Avoid duplicate workloads while preserving input order.
    duplicate=false
    for existing in "${SELECTED_WORKLOADS[@]:-}"; do
      if [[ "$existing" == "$workload" ]]; then
        duplicate=true
        break
      fi
    done

    if [[ "$duplicate" == false ]]; then
      SELECTED_WORKLOADS+=("$workload")
    fi
  done
fi

WORKLOAD_ARGS=()
for workload in "${SELECTED_WORKLOADS[@]}"; do
  WORKLOAD_ARGS+=("w${workload}")
done

# ----------------------------------------------------------------------
# Data/cache directory
# ----------------------------------------------------------------------

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

export CFD_BENCH_DATA_ROOT="${CFD_BENCH_DATA_ROOT:-$PROJECT_ROOT/data}"
mkdir -p "$CFD_BENCH_DATA_ROOT"

# ----------------------------------------------------------------------
# Temporary benchmark result
# ----------------------------------------------------------------------

RESULT_CSV=$(mktemp "${TMPDIR:-/tmp}/cfd_bench_perf.XXXXXX.csv")
trap 'rm -f "$RESULT_CSV"' EXIT

echo "[performance] Dataset: $DATASET"
echo "[performance] Workloads: ${WORKLOAD_ARGS[*]}"
echo "[performance] Duration: ${DURATION}s"
echo

# ----------------------------------------------------------------------
# Run benchmark
# ----------------------------------------------------------------------

set +e

PYTHONUNBUFFERED=1 cfd-bench run \
  --backend postgresql iotdb tiledb vtk \
  --workloads "${WORKLOAD_ARGS[@]}" \
  --duration "$DURATION" \
  --datasets "$DATASET" \
  --output "$RESULT_CSV" \
  2>/dev/null |
while IFS= read -r line; do
  case "$line" in
    *"Running w2 "*)
      echo "[performance] Running W2 ..."
      ;;
    *"Running w4 "*)
      echo "[performance] Running W4 ..."
      ;;
    *"Running w5 "*)
      echo "[performance] Running W5 ..."
      ;;
    *"Running w6 "*)
      echo "[performance] Running W6 ..."
      ;;
    *"Running w8 "*)
      echo "[performance] Running W8 ..."
      ;;
  esac
done

RUN_STATUS=${PIPESTATUS[0]}

set -e

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "Error: cfd-bench performance run failed." >&2
  exit "$RUN_STATUS"
fi

# ----------------------------------------------------------------------
# Calculate final TPM result
# ----------------------------------------------------------------------

python3 - "$RESULT_CSV" <<'PY'
import csv
import math
import sys
from collections import defaultdict

path = sys.argv[1]

required_backends = (
    "postgresql",
    "iotdb",
    "tiledb",
    "vtk",
)

# key:
#   (workload, operation, step)
#
# value:
#   {
#       "postgresql": [...],
#       "iotdb": [...],
#       "tiledb": [...],
#       "vtk": [...]
#   }
values = defaultdict(lambda: defaultdict(list))

with open(path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        workload = row.get("workload", "").strip().lower()

        if workload not in {"w2", "w4", "w5", "w6", "w8"}:
            continue

        backend = row.get("backend", "").strip().lower()

        if backend not in required_backends:
            continue

        operation = row.get("operation", "").strip()
        step = row.get("step", "").strip()

        try:
            rate = float(row.get("txns_per_sec", ""))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(rate):
            continue

        values[(workload, operation, step)][backend].append(rate)


if not values:
    print(
        "Error: benchmark CSV contained no usable performance readings.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def mean(items):
    return sum(items) / len(items)


# ------------------------------------------------------------------
# First normalize each workload / operation / step.
#
# This prevents an operation with more CSV samples from receiving
# artificially larger weight in the final average.
# ------------------------------------------------------------------

db_rates = []
vtk_rates = []
missing = []

for key, by_backend in values.items():

    absent = [
        backend
        for backend in required_backends
        if not by_backend.get(backend)
    ]

    if absent:
        missing.append((key, absent))
        continue

    pg_tps = mean(by_backend["postgresql"])
    iotdb_tps = mean(by_backend["iotdb"])
    tiledb_tps = mean(by_backend["tiledb"])
    vtk_tps = mean(by_backend["vtk"])

    # Average of the three database backends.
    db_tps = (
        pg_tps
        + iotdb_tps
        + tiledb_tps
    ) / 3.0

    db_rates.append(db_tps)
    vtk_rates.append(vtk_tps)


if missing:
    for (workload, operation, step), absent in sorted(missing):
        print(
            "Error: missing backend result for "
            f"{workload}/{operation}/step={step or '-'}: "
            + ", ".join(absent),
            file=sys.stderr,
        )

    raise SystemExit(1)


if not db_rates or not vtk_rates:
    print(
        "Error: no complete DB/VTK performance groups found.",
        file=sys.stderr,
    )
    raise SystemExit(1)


# ------------------------------------------------------------------
# Final average throughput
# ------------------------------------------------------------------

db_avg_tps = mean(db_rates)
vtk_avg_tps = mean(vtk_rates)

# TPS -> TPM
db_avg_tpm = db_avg_tps * 60.0
vtk_avg_tpm = vtk_avg_tps * 60.0

ratio = (
    math.inf
    if vtk_avg_tpm == 0.0
    else db_avg_tpm / vtk_avg_tpm
)


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

headers = (
    "db_avg_tpm",
    "vtk_avg_tpm",
    "db_avg_tpm/vtk_avg_tpm=R",
)

values = (
    f"{db_avg_tpm:.6f}",
    f"{vtk_avg_tpm:.6f}",
    "inf" if math.isinf(ratio) else f"{ratio:.6f}",
)

widths = [
    max(len(headers[i]), len(values[i]))
    for i in range(len(headers))
]

print()
print(
    "  ".join(
        headers[i].ljust(widths[i])
        for i in range(len(headers))
    )
)

print(
    "  ".join(
        "-" * widths[i]
        for i in range(len(headers))
    )
)

print(
    "  ".join(
        values[i].ljust(widths[i])
        for i in range(len(values))
    )
)
PY