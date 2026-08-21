FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CFD_BENCH_DATA_ROOT=/app/data

WORKDIR /app

# VTK wheels still rely on a small set of system OpenGL/X11 runtime libraries
# even though CFD-Bench itself uses VTK headlessly.
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY docker ./docker

# Keep the Python client in lock-step with the IoTDB server image selected in
# compose.yaml. The remaining backend libraries are installed from the
# project's all-extra so one image can execute every supported path.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "apache-iotdb==2.0.10" \
    && python -m pip install ".[all]" \
    && python -m pip check \
    && chmod +x /app/scripts/*.sh /app/docker/entrypoint.sh

RUN mkdir -p /app/data /app/output /datasets

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["cfd-bench", "--help"]
