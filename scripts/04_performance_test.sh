#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <DATASET>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

DATASET=$1
DURATION=10

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
export CFD_BENCH_DATA_ROOT="$PROJECT_ROOT/data"
mkdir -p "$CFD_BENCH_DATA_ROOT"

RESULT_CSV=$(mktemp "${TMPDIR:-/tmp}/cfd_bench_perf.XXXXXX.csv")
trap 'rm -f "$RESULT_CSV"' EXIT

if ! cfd-bench run \
  --backend postgresql iotdb tiledb vtk \
  --workloads w2 w4 w5 w6 w8 \
  --duration "$DURATION" \
  --datasets "$DATASET" \
  --output "$RESULT_CSV" \
  >/dev/null 2>&1; then
  echo "Error: cfd-bench performance run failed." >&2
  exit 1
fi

python3 - "$RESULT_CSV" <<'PY'
import csv
import math
import sys
from collections import defaultdict

path = sys.argv[1]
required_backends = ("postgresql", "iotdb", "tiledb", "vtk")
workload_order = {"w2": 0, "w4": 1, "w5": 2, "w6": 3, "w8": 4}

# key = (workload, operation, step)
values = defaultdict(lambda: defaultdict(list))
with open(path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        workload = row.get("workload", "").strip().lower()
        if workload not in workload_order:
            continue
        backend = row.get("backend", "").strip().lower()
        if backend not in required_backends:
            continue
        operation = row.get("operation", "").strip()
        step = row.get("step", "").strip()
        try:
            rate = float(row.get("txns_per_sec", ""))
        except ValueError:
            continue
        if not math.isfinite(rate):
            continue
        values[(workload, operation, step)][backend].append(rate)

if not values:
    print("Error: benchmark CSV contained no usable performance readings.", file=sys.stderr)
    raise SystemExit(1)

def step_sort_value(step):
    if step == "":
        return (-1, 0)
    try:
        return (0, int(step))
    except ValueError:
        return (1, step)

def mean(items):
    return sum(items) / len(items)

rows = []
missing = []
for key, by_backend in values.items():
    absent = [b for b in required_backends if not by_backend.get(b)]
    if absent:
        missing.append((key, absent))
        continue
    workload, operation, step = key
    pg = mean(by_backend["postgresql"])
    iotdb = mean(by_backend["iotdb"])
    tiledb = mean(by_backend["tiledb"])
    vtk = mean(by_backend["vtk"])
    db_avg = (pg + iotdb + tiledb) / 3.0
    ratio = math.inf if vtk == 0.0 else db_avg / vtk
    rows.append((workload, operation, step, db_avg, vtk, ratio))

if missing:
    for (workload, operation, step), absent in sorted(
        missing,
        key=lambda x: (
            workload_order.get(x[0][0], 999),
            step_sort_value(x[0][2]),
            x[0][1],
        ),
    ):
        step_text = step or "-"
        print(
            f"Error: missing backend result for {workload}/{operation}/step={step_text}: "
            + ", ".join(absent),
            file=sys.stderr,
        )
    raise SystemExit(1)

rows.sort(key=lambda r: (workload_order[r[0]], step_sort_value(r[2]), r[1]))

headers = ("workload", "operation", "step", "db_avg_txns/s", "vtk_txns/s", "db_avg/vtk")
formatted = []
for workload, operation, step, db_avg, vtk, ratio in rows:
    formatted.append(
        (
            workload,
            operation,
            step or "-",
            f"{db_avg:.6g}",
            f"{vtk:.6g}",
            "inf" if math.isinf(ratio) else f"{ratio:.6g}",
        )
    )

widths = [len(h) for h in headers]
for row in formatted:
    for i, value in enumerate(row):
        widths[i] = max(widths[i], len(value))

print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
print("  ".join("-" * widths[i] for i in range(len(headers))))
for row in formatted:
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
PY
