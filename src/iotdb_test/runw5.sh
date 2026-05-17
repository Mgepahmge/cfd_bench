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
    echo "Usage: sh runw5.sh <workload>"
    echo "Supported workloads: 1 3 5 7"
}

WORKLOAD="${1:-}"
if [[ -z "$WORKLOAD" ]]; then
    usage
    exit 1
fi

declare -A WORKLOAD_SCRIPT=(
    ["1"]="testw1io.py"
    ["3"]="testw3io.py"
    ["5"]="testw5io.py"
    ["7"]="testw7io.py"
)

if [[ -z "${WORKLOAD_SCRIPT[$WORKLOAD]:-}" ]]; then
    usage
    exit 1
fi

PYTHON_SCRIPT="${WORKLOAD_SCRIPT[$WORKLOAD]}"

# 定义船型和时间步
SHIP_TYPES=("JBC_615k" "JBC_3843k" "Kvlcc2_351k" "Kvlcc2_3709k" "Suboff_3258k")
TIME_STEPS=("200" "400" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000")
VTK_DIR="${VTK_DIR:-$SCRIPT_DIR/../../vtk_dir}"

# 运行时可配置项
PYTHON_BIN="${PYTHON_BIN:-python3}"
IOTDB_HOME="${IOTDB_HOME:-/home/lzhang/IotDB/apache-iotdb-2.0.5-all-bin}"
IOTDB_WAIT_SECONDS="${IOTDB_WAIT_SECONDS:-5}"
DROP_CACHES="${DROP_CACHES:-1}"
PYTHONPATH="${PYTHONPATH:-$SRC_DIR}"
LOG_FILE="benchmark_w${WORKLOAD}.log"
RESUME_FROM_LOG="${RESUME_FROM_LOG:-1}"
START_SHIP="${START_SHIP:-}"
START_STEP="${START_STEP:-}"

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
    local step="$2"

    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        local target_home
        target_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
        sudo -u "$SUDO_USER" env \
            HOME="$target_home" \
            USER="$SUDO_USER" \
            LOGNAME="$SUDO_USER" \
            SHIP_TO_RUN="$ship" \
            STEP_TO_RUN="$step" \
            PYTHONPATH="$PYTHONPATH" \
            "$PYTHON_BIN" "$PYTHON_SCRIPT" >>"$LOG_FILE" 2>&1
    else
        SHIP_TO_RUN="$ship" STEP_TO_RUN="$step" PYTHONPATH="$PYTHONPATH" \
            "$PYTHON_BIN" "$PYTHON_SCRIPT" >>"$LOG_FILE" 2>&1
    fi
}

has_vtk_for_step() {
    local ship="$1"
    local step="$2"
    compgen -G "$VTK_DIR/*${ship}*_${step}.vtk" > /dev/null
}

declare -A COMPLETED_CASES=()
if [[ "$RESUME_FROM_LOG" == "1" && -f "$LOG_FILE" ]]; then
    last_case=""
    traceback_after_last_case=0

    while IFS= read -r line; do
        if [[ "$line" =~ Workload:\ w[0-9]+,\ Ship:\ ([^,]+),\ Step:\ ([^,]+),\ Script: ]]; then
            ship="${BASH_REMATCH[1]}"
            step="${BASH_REMATCH[2]}"
            case_key="${ship}|${step}"
            COMPLETED_CASES["$case_key"]=1
            last_case="$case_key"
            traceback_after_last_case=0
            continue
        fi

        if [[ -n "$last_case" && "$line" == Traceback* ]]; then
            traceback_after_last_case=1
        fi
    done < "$LOG_FILE"

    # 最后一个 case 若以 Traceback 结束，则视为失败断点，不计入已完成。
    if [[ -n "$last_case" && "$traceback_after_last_case" == "1" ]]; then
        unset 'COMPLETED_CASES[$last_case]'
    fi
fi

start_filter_active=0
start_reached=0
if [[ -n "$START_SHIP" || -n "$START_STEP" ]]; then
    if [[ -z "$START_SHIP" || -z "$START_STEP" ]]; then
        echo "ERROR: START_SHIP and START_STEP must be set together."
        exit 1
    fi
    start_filter_active=1
fi

for SHIP in "${SHIP_TYPES[@]}"; do
    for STEP in "${TIME_STEPS[@]}"; do
        if [[ "$start_filter_active" == "1" && "$start_reached" == "0" ]]; then
            if [[ "$SHIP" == "$START_SHIP" && "$STEP" == "$START_STEP" ]]; then
                start_reached=1
            else
                continue
            fi
        fi

        case_key="${SHIP}|${STEP}"
        if [[ -n "${COMPLETED_CASES[$case_key]:-}" ]]; then
            echo "RESUME: skip completed Ship=$SHIP Step=$STEP" | tee -a "$LOG_FILE"
            continue
        fi

        if ! has_vtk_for_step "$SHIP" "$STEP"; then
            echo "SKIP: missing VTK data for Ship=$SHIP Step=$STEP" | tee -a "$LOG_FILE"
            continue
        fi

        echo "====================================================" | tee -a "$LOG_FILE"
        echo "Workload: w${WORKLOAD}, Ship: $SHIP, Step: $STEP, Script: $PYTHON_SCRIPT" | tee -a "$LOG_FILE"
        echo "====================================================" | tee -a "$LOG_FILE"

        restart_iotdb

        run_python_workload "$SHIP" "$STEP"
    done
done