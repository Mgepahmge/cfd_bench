"""Best-effort dataset inventory without changing CFD-Bench storage contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

from cfd_bench.core.paths import resolve_tiledb_root, resolve_vtk_dir

from .state import StateStore


def discover_dataset_sources(store: StateStore) -> Dict[str, Set[str]]:
    """Combine API history with file-backed stores and PostgreSQL discovery.

    IoTDB's current public discovery helper requires candidate dataset names,
    so API-ingested names are used as its registry rather than introducing a
    new wildcard-query contract in the core project.
    """
    found: Dict[str, Set[str]] = {}

    for dataset in store.registered_datasets():
        found.setdefault(dataset, set()).add("api-registry")

    # Existing PostgreSQL discovery can enumerate every dataset directly.
    try:
        from cfd_bench.infra.postgresql.discovery import discover_postgresql_datasets

        for info in discover_postgresql_datasets():
            found.setdefault(info.dataset_key, set()).add("postgresql")
    except Exception:
        pass

    tiledb_root = Path(resolve_tiledb_root())
    if tiledb_root.is_dir():
        for child in tiledb_root.iterdir():
            if child.is_dir():
                found.setdefault(child.name, set()).add("tiledb")

    vtk_root = Path(resolve_vtk_dir())
    if vtk_root.is_dir():
        for child in vtk_root.iterdir():
            manifest = child / "manifest.json"
            if not manifest.is_file():
                continue
            key = child.name
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                key = str(payload.get("dataset_key") or key)
            except Exception:
                pass
            found.setdefault(key, set()).add("vtk")

    return found
