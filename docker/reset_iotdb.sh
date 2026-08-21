#!/usr/bin/env bash
set -euo pipefail

# Reset only the Docker-managed IoTDB state. PostgreSQL and host ./data are not
# touched. This is useful after a failed first bootstrap because IoTDB stores
# node identity/address information in /iotdb/data and several of those values
# are immutable after first startup.

command -v docker >/dev/null 2>&1 || {
  echo "Error: docker is required." >&2
  exit 1
}

project=${COMPOSE_PROJECT_NAME:-cfd-bench}

echo "[docker] stopping/removing IoTDB container ..."
docker compose stop iotdb >/dev/null 2>&1 || true
docker compose rm -sf iotdb >/dev/null 2>&1 || true

removed=0
for logical in iotdb-data iotdb-logs; do
  while IFS= read -r volume; do
    [[ -n "$volume" ]] || continue
    echo "[docker] removing volume $volume"
    docker volume rm "$volume" >/dev/null
    removed=1
  done < <(
    docker volume ls -q \
      --filter "label=com.docker.compose.project=${project}" \
      --filter "label=com.docker.compose.volume=${logical}"
  )
done

if [[ $removed -eq 0 ]]; then
  echo "[docker] no IoTDB volumes found for compose project ${project}"
fi

echo "[docker] IoTDB state reset complete. PostgreSQL was not modified."
echo "[docker] next: docker compose up -d iotdb"
