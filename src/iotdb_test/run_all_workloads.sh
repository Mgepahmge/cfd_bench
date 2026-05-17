#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage:"
    echo "  sh run_all_workloads.sh                 # run default workloads: 1 2 3 4 5 7 8"
    echo "  sh run_all_workloads.sh 1 3 5           # run selected workloads"
    echo "  sh run_all_workloads.sh 2 4 8           # run selected workloads"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

declare -A VALID_WORKLOADS=(
    ["1"]="1"
    ["2"]="2"
    ["3"]="3"
    ["4"]="4"
    ["5"]="5"
    ["7"]="7"
    ["8"]="8"
)

if [[ $# -eq 0 ]]; then
    WORKLOADS=("1" "2" "3" "4" "5" "7" "8")
else
    WORKLOADS=("$@")
fi

for w in "${WORKLOADS[@]}"; do
    if [[ -z "${VALID_WORKLOADS[$w]:-}" ]]; then
        echo "ERROR: unsupported workload '$w'"
        usage
        exit 1
    fi
done

for w in "${WORKLOADS[@]}"; do
    echo "============================================================"
    echo "Starting workload w${w}"
    echo "============================================================"

    if [[ "$w" == "1" || "$w" == "3" || "$w" == "5" || "$w" == "7" ]]; then
        bash "$SCRIPT_DIR/runw5.sh" "$w"
    else
        bash "$SCRIPT_DIR/runw24.sh" "$w"
    fi
done

echo "All requested workloads completed."
