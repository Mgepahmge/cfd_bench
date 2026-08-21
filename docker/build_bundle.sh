#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained CFD-Bench offline deployment bundle.
#
# The bundle contains:
#   - all Docker images required by compose.yaml
#   - an offline-only compose.yaml (no build section, pull_policy=never)
#   - one-click image loader and service start/stop/status helpers
#   - the runtime data/, datasets/ and output/ directories
#   - environment files and a checksum manifest
#
# Usage:
#   ./docker/build_bundle.sh [output.tar.gz]
#
# Optional environment variables:
#   CFD_BENCH_BUNDLE_NAME=<directory-name-inside-archive>
#   CFD_BENCH_BUNDLE_COPY_RUNTIME_DIRS=1   # 1=copy contents, 0=create empty dirs
#   CFD_BENCH_BUNDLE_SKIP_BUILD=0         # 1=use existing app image
#   CFD_BENCH_BUNDLE_SKIP_PULL=0          # 1=use existing dependency images

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
COMPOSE_FILE="$PROJECT_ROOT/compose.yaml"

COPY_RUNTIME_DIRS=${CFD_BENCH_BUNDLE_COPY_RUNTIME_DIRS:-1}
SKIP_BUILD=${CFD_BENCH_BUNDLE_SKIP_BUILD:-0}
SKIP_PULL=${CFD_BENCH_BUNDLE_SKIP_PULL:-0}

log() {
  printf '[bundle] %s\n' "$*"
}

die() {
  printf '[bundle] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

case "$COPY_RUNTIME_DIRS" in
  0|1) ;;
  *) die "CFD_BENCH_BUNDLE_COPY_RUNTIME_DIRS must be 0 or 1" ;;
esac
case "$SKIP_BUILD" in
  0|1) ;;
  *) die "CFD_BENCH_BUNDLE_SKIP_BUILD must be 0 or 1" ;;
esac
case "$SKIP_PULL" in
  0|1) ;;
  *) die "CFD_BENCH_BUNDLE_SKIP_PULL must be 0 or 1" ;;
esac

require_cmd docker
require_cmd tar
require_cmd gzip
require_cmd awk
require_cmd sed
require_cmd grep
require_cmd sort
require_cmd mktemp
require_cmd sha256sum

docker compose version >/dev/null 2>&1 || die "Docker Compose v2 (docker compose) is required"
[[ -f "$COMPOSE_FILE" ]] || die "compose.yaml not found: $COMPOSE_FILE"

# Resolve the image list from Compose so image version changes do not have to be
# duplicated in this script.
mapfile -t IMAGES < <(
  docker compose -f "$COMPOSE_FILE" config --images \
    | sed '/^[[:space:]]*$/d' \
    | sort -u
)

((${#IMAGES[@]} > 0)) || die "no images found in compose.yaml"

APP_IMAGE=''
for image in "${IMAGES[@]}"; do
  if [[ "$image" == cfd-bench:* || "$image" == */cfd-bench:* ]]; then
    APP_IMAGE=$image
    break
  fi
done
[[ -n "$APP_IMAGE" ]] || APP_IMAGE='cfd-bench:unknown'

APP_TAG=${APP_IMAGE##*:}
if [[ "$APP_TAG" == "$APP_IMAGE" || -z "$APP_TAG" ]]; then
  APP_TAG='latest'
fi

BUNDLE_NAME=${CFD_BENCH_BUNDLE_NAME:-cfd-bench-offline-${APP_TAG}}
DEFAULT_OUT="${BUNDLE_NAME}.tar.gz"
OUT=${1:-$DEFAULT_OUT}
if [[ "$OUT" != /* ]]; then
  OUT="$(pwd)/$OUT"
fi

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/cfd-bench-bundle.XXXXXX")
STAGE_ROOT="$TMP_ROOT/$BUNDLE_NAME"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

mkdir -p "$STAGE_ROOT/images" "$STAGE_ROOT/data" "$STAGE_ROOT/datasets" "$STAGE_ROOT/output"

if [[ "$SKIP_BUILD" == 0 ]]; then
  log "building application image"
  docker compose -f "$COMPOSE_FILE" build cfd-bench
else
  log "skipping application build (CFD_BENCH_BUNDLE_SKIP_BUILD=1)"
fi

if [[ "$SKIP_PULL" == 0 ]]; then
  log "pulling PostgreSQL/PostGIS and IoTDB images"
  docker compose -f "$COMPOSE_FILE" pull postgres iotdb
else
  log "skipping dependency pulls (CFD_BENCH_BUNDLE_SKIP_PULL=1)"
fi

# Fail before creating a large archive if any required image is missing.
for image in "${IMAGES[@]}"; do
  docker image inspect "$image" >/dev/null 2>&1 || die "required image is missing: $image"
done

log "creating offline Compose configuration"
# Remove build: from the application service and force Compose never to pull.
# The generated file therefore uses only images loaded from images/*.tar.gz.
awk '
  skip_build {
    if ($0 ~ /^      / || $0 ~ /^[[:space:]]*$/) {
      next
    }
    skip_build = 0
  }
  /^    build:[[:space:]]*$/ {
    skip_build = 1
    next
  }
  {
    print
    if ($0 ~ /^    image:[[:space:]]*/) {
      print "    pull_policy: never"
    }
  }
' "$COMPOSE_FILE" > "$STAGE_ROOT/compose.yaml"

if grep -Eq '^[[:space:]]+build:' "$STAGE_ROOT/compose.yaml"; then
  die "generated offline compose.yaml still contains a build section"
fi

docker compose -f "$STAGE_ROOT/compose.yaml" config --quiet \
  || die "generated offline compose.yaml is invalid"

# Preserve actual runtime environment when present. Otherwise provide a usable
# .env from the project's example/defaults.
if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
  cp "$PROJECT_ROOT/.env.example" "$STAGE_ROOT/.env.example"
fi
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  cp "$PROJECT_ROOT/.env" "$STAGE_ROOT/.env"
elif [[ -f "$PROJECT_ROOT/.env.example" ]]; then
  cp "$PROJECT_ROOT/.env.example" "$STAGE_ROOT/.env"
else
  cat > "$STAGE_ROOT/.env" <<'ENVEOF'
CFD_BENCH_IOTDB_DISK_SPACE_WARNING_THRESHOLD=0.01
ENVEOF
fi

for dir in data datasets output; do
  mkdir -p "$STAGE_ROOT/$dir"
  if [[ "$COPY_RUNTIME_DIRS" == 1 && -d "$PROJECT_ROOT/$dir" ]]; then
    log "copying $dir/ into bundle"
    cp -a "$PROJECT_ROOT/$dir/." "$STAGE_ROOT/$dir/"
  fi
done

cat > "$STAGE_ROOT/load_images.sh" <<'EOF_LOAD'
#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE_ARCHIVE="$ROOT/images/cfd-bench-images.tar.gz"
EXPECTED_IMAGES="$ROOT/images/expected-images.txt"

command -v docker >/dev/null 2>&1 || {
  echo "Error: docker is required." >&2
  exit 1
}
[[ -f "$IMAGE_ARCHIVE" ]] || {
  echo "Error: image archive not found: $IMAGE_ARCHIVE" >&2
  exit 1
}

echo "[offline] loading Docker images ..."
gzip -dc "$IMAGE_ARCHIVE" | docker load

if [[ -f "$EXPECTED_IMAGES" ]]; then
  while IFS= read -r image; do
    [[ -n "$image" ]] || continue
    docker image inspect "$image" >/dev/null 2>&1 || {
      echo "Error: image was not loaded successfully: $image" >&2
      exit 1
    }
  done < "$EXPECTED_IMAGES"
fi

echo "[offline] all Docker images are ready."
EOF_LOAD

cat > "$STAGE_ROOT/start.sh" <<'EOF_START'
#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

command -v docker >/dev/null 2>&1 || {
  echo "Error: docker is required." >&2
  exit 1
}
docker compose version >/dev/null 2>&1 || {
  echo "Error: Docker Compose v2 is required." >&2
  exit 1
}

mkdir -p data datasets output

echo "[offline] starting PostgreSQL/PostGIS and IoTDB ..."
docker compose -f compose.yaml up -d postgres iotdb
docker compose -f compose.yaml ps

echo
printf '%s\n' "[offline] services started." \
  "Run ./cfd-bench.sh --help to verify the application." \
  "Put input datasets under ./datasets and results will be available under ./output."
EOF_START

cat > "$STAGE_ROOT/install.sh" <<'EOF_INSTALL'
#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$ROOT/load_images.sh"
"$ROOT/start.sh"
EOF_INSTALL

cat > "$STAGE_ROOT/stop.sh" <<'EOF_STOP'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
docker compose -f compose.yaml down
EOF_STOP

cat > "$STAGE_ROOT/status.sh" <<'EOF_STATUS'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
docker compose -f compose.yaml ps
EOF_STATUS

cat > "$STAGE_ROOT/cfd-bench.sh" <<'EOF_RUN'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

if [[ $# -eq 0 ]]; then
  set -- --help
fi

exec docker compose -f compose.yaml run --rm cfd-bench cfd-bench "$@"
EOF_RUN

cat > "$STAGE_ROOT/verify.sh" <<'EOF_VERIFY'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
sha256sum -c SHA256SUMS
EOF_VERIFY

chmod +x \
  "$STAGE_ROOT/load_images.sh" \
  "$STAGE_ROOT/start.sh" \
  "$STAGE_ROOT/install.sh" \
  "$STAGE_ROOT/stop.sh" \
  "$STAGE_ROOT/status.sh" \
  "$STAGE_ROOT/cfd-bench.sh" \
  "$STAGE_ROOT/verify.sh"

printf '%s\n' "${IMAGES[@]}" > "$STAGE_ROOT/images/expected-images.txt"

cat > "$STAGE_ROOT/README-OFFLINE.md" <<EOF_README
# CFD-Bench offline bundle

This directory is a self-contained offline runtime package for **${APP_IMAGE}**.
It does not contain source code and does not build or pull images on the target
machine.

## First use on the offline machine

\`\`\`bash
./verify.sh
./install.sh
\`\`\`

\`install.sh\` loads all Docker images and starts PostgreSQL/PostGIS + IoTDB.
If you prefer separate steps:

\`\`\`bash
./load_images.sh
./start.sh
\`\`\`

## Run CFD-Bench

\`\`\`bash
./cfd-bench.sh --help
\`\`\`

Example:

\`\`\`bash
./cfd-bench.sh run \\
  --backend postgresql iotdb tiledb vtk \\
  --workloads w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 \\
  --duration 10 \\
  --datasets Kvlcc_351K_Small \\
  --output /app/output/results.csv
\`\`\`

## Runtime directories

- \`data/\` -> mounted at \`/app/data\`
- \`datasets/\` -> mounted read-only at \`/datasets\`
- \`output/\` -> mounted at \`/app/output\`

## Operations

\`\`\`bash
./status.sh
./stop.sh
\`\`\`

The generated \`compose.yaml\` has no \`build:\` section and uses
\`pull_policy: never\`, so the offline host will not try to build or download
Docker images.
EOF_README

log "exporting ${#IMAGES[@]} Docker images"
docker save "${IMAGES[@]}" | gzip -1 > "$STAGE_ROOT/images/cfd-bench-images.tar.gz"

# Record image metadata for troubleshooting on the target machine.
{
  printf '%-45s %-20s %-14s %s\n' IMAGE ID ARCHITECTURE CREATED
  for image in "${IMAGES[@]}"; do
    docker image inspect \
      --format '{{.RepoTags}} {{.Id}} {{.Architecture}} {{.Created}}' \
      "$image"
  done
} > "$STAGE_ROOT/images/IMAGE_MANIFEST.txt"

log "creating SHA256SUMS"
(
  cd "$STAGE_ROOT"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

mkdir -p "$(dirname -- "$OUT")"
rm -f "$OUT"
log "creating final offline package: $OUT"
tar -C "$TMP_ROOT" -czf "$OUT" "$BUNDLE_NAME"

log "bundle build complete"
log "output: $OUT"
log "target host: tar -xzf $(basename -- "$OUT") && cd $BUNDLE_NAME && ./install.sh"