#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-cfd-bench-docker-bundle-v22.4.2.tar.gz}

command -v docker >/dev/null 2>&1 || {
  echo "Error: docker is required." >&2
  exit 1
}

docker compose build cfd-bench
docker pull postgis/postgis:16-3.5
docker pull apache/iotdb:2.0.10-standalone

echo "[docker] exporting application + PostgreSQL/PostGIS + IoTDB images -> $OUT"
docker save \
  cfd-bench:v22.4.2 \
  postgis/postgis:16-3.5 \
  apache/iotdb:2.0.10-standalone \
  | gzip -c > "$OUT"

echo "[docker] bundle ready: $OUT"
