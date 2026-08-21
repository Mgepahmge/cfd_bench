#!/usr/bin/env bash
set -euo pipefail

if [[ "${CFD_BENCH_SKIP_SERVICE_WAIT:-0}" != "1" ]]; then
  python /app/docker/wait_for_services.py
fi

exec "$@"
