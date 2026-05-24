from __future__ import annotations

import os
from typing import Literal

from .context import DatasetKey

Backend = Literal["iotdb", "tiledb", "postgresql"]


def iotdb_root() -> str:
    return "root.simulation_data"


def iotdb_mesh_static(key: DatasetKey, leaf: str) -> str:
    return f"{iotdb_root()}.mesh_static.{key.dataset_key}.{key.zone}.{leaf}"


def iotdb_cell_vars(key: DatasetKey, leaf: str = "cell_vars") -> str:
    if key.zone in ("1_Hull", "hull"):
        leaf = "cell_vars_hull" if leaf == "cell_vars" else leaf
    return f"{iotdb_root()}.post_processing_management.{key.dataset_key}.step_{int(key.step)}.{leaf}"


def iotdb_derived(key: DatasetKey, leaf: str) -> str:
    return f"{iotdb_root()}.derived.{key.dataset_key}.step_{int(key.step)}.{leaf}"


def iotdb_legacy_elements(key: DatasetKey) -> str:
    return f"{iotdb_root()}.post_processing_management.{key.dataset_key}.Elements"


def iotdb_legacy_variables(key: DatasetKey) -> str:
    return f"{iotdb_root()}.post_processing_management.{key.dataset_key}.step_{int(key.step)}.Variables"


def tiledb_root(root_path: str, key: DatasetKey) -> str:
    return os.path.join(root_path, key.dataset_key)


def tiledb_mesh_static(root_path: str, key: DatasetKey, leaf: str) -> str:
    return os.path.join(tiledb_root(root_path, key), "mesh_static", key.zone, f"{leaf}.tdb")


def tiledb_cell_vars(root_path: str, key: DatasetKey) -> str:
    leaf = "cell_vars_hull.tdb" if key.zone in ("1_Hull", "hull") else "cell_vars.tdb"
    return os.path.join(tiledb_root(root_path, key), "post_processing", f"step_{int(key.step)}", leaf)


def tiledb_derived(root_path: str, key: DatasetKey, leaf: str) -> str:
    return os.path.join(tiledb_root(root_path, key), "derived", f"step_{int(key.step)}", f"{leaf}.tdb")
