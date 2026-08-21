#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cfd-bench-docker-bundle.tar.gz>" >&2
  exit 2
fi

command -v docker >/dev/null 2>&1 || {
  echo "Error: docker is required." >&2
  exit 1
}

gzip -dc "$1" | docker load
