#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "INFO: Script will relaunch with sudo to avoid password interruption."
    exec sudo -E bash "$SCRIPT_PATH" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: sh runw24.sh <workload>"
    echo "Supported workloads: 2 4 8"
}

WORKLOAD="${1:-}"
if [[ -z "$WORKLOAD" ]]; then
    usage
    exit 1
fi

declare -A WORKLOAD_SCRIPT=(
    ["2"]="testw2io.py"
    ["4"]="testw4io.py"
    ["8"]="testw81io.py"
)

if [[ -z "${WORKLOAD_SCRIPT[$WORKLOAD]:-}" ]]; then
    usage
    exit 1
fi

PYTHON_SCRIPT="${WORKLOAD_SCRIPT[$WORKLOAD]}"

# 定义所有需要测试的船型列表
SHIP_TYPES=("JBC_615k" "JBC_3843k" "Kvlcc2_351k" "Kvlcc2_3709k" "Suboff_3258k")
VTK_DIR="${VTK_DIR:-$SCRIPT_DIR/../../vtk_dir}"

# 运行时可配置项
PYTHON_BIN="${PYTHON_BIN:-python3}"
IOTDB_HOME="${IOTDB_HOME:-/home/lzhang/IotDB/apache-iotdb-2.0.5-all-bin}"
IOTDB_WAIT_SECONDS="${IOTDB_WAIT_SECONDS:-5}"
DROP_CACHES="${DROP_CACHES:-1}"
PYTHONPATH="${PYTHONPATH:-$SRC_DIR}"
LOG_FILE="benchmark_w${WORKLOAD}.log"

IOTDB_STOP_CMD="$IOTDB_HOME/sbin/stop-standalone.sh"
IOTDB_START_CMD="$IOTDB_HOME/sbin/start-standalone.sh"

if [[ ! -x "$IOTDB_STOP_CMD" || ! -x "$IOTDB_START_CMD" ]]; then
    echo "ERROR: IoTDB startup scripts are missing or not executable."
    echo "Checked: $IOTDB_STOP_CMD and $IOTDB_START_CMD"
    exit 1
fi

restart_iotdb() {
    if ! "$IOTDB_STOP_CMD" >>"$LOG_FILE" 2>&1; then
        echo "WARN: IoTDB stop command returned non-zero, continue." | tee -a "$LOG_FILE"
    fi
    sleep "$IOTDB_WAIT_SECONDS"

    if [[ "$DROP_CACHES" == "1" ]]; then
        sync
        echo 3 | tee /proc/sys/vm/drop_caches >/dev/null
    fi

    "$IOTDB_START_CMD" >>"$LOG_FILE" 2>&1
    sleep "$IOTDB_WAIT_SECONDS"
}

run_python_workload() {
    local ship="$1"

    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        local target_home
        target_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
        sudo -u "$SUDO_USER" env \
            HOME="$target_home" \
            USER="$SUDO_USER" \
            LOGNAME="$SUDO_USER" \
            SHIP_TO_RUN="$ship" \
            PYTHONPATH="$PYTHONPATH" \
            "$PYTHON_BIN" "$PYTHON_SCRIPT" >>"$LOG_FILE" 2>&1
    else
        SHIP_TO_RUN="$ship" PYTHONPATH="$PYTHONPATH" \
            "$PYTHON_BIN" "$PYTHON_SCRIPT" >>"$LOG_FILE" 2>&1
    fi
}

has_vtk_for_step() {
    local ship="$1"
    local step="$2"
    compgen -G "$VTK_DIR/*${ship}*_${step}.vtk" > /dev/null
}

for SHIP in "${SHIP_TYPES[@]}"; do
    # w2/w4/w8 all run per-ship dataset. Each python script handles multi-timestep logic.
    if ! has_vtk_for_step "$SHIP" "200"; then
        echo "SKIP: missing VTK base data for Ship=$SHIP Step=200" | tee -a "$LOG_FILE"
        continue
    fi

    echo "========================================" | tee -a "$LOG_FILE"
    echo "Workload: w${WORKLOAD}, Ship: $SHIP, Script: $PYTHON_SCRIPT" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    restart_iotdb
    run_python_workload "$SHIP"
done
