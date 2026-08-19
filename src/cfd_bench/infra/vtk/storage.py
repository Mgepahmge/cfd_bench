from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping


def dataset_dir(root: str, dataset_key: str) -> Path:
    return Path(root) / str(dataset_key)


def manifest_path(root: str, dataset_key: str) -> Path:
    return dataset_dir(root, dataset_key) / "manifest.json"


def _safe_zone(zone: str) -> str:
    text = str(zone or "0_Fluid").strip()
    # Keep readable names while preventing accidental nested paths.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text) or "0_Fluid"


def frame_path(root: str, dataset_key: str, zone: str, step: int) -> Path:
    return dataset_dir(root, dataset_key) / "zones" / _safe_zone(zone) / f"step_{int(step)}.vtu"


def relative_frame_path(dataset_key: str, zone: str, step: int) -> str:
    # Relative to dataset directory, not root.
    return str(Path("zones") / _safe_zone(zone) / f"step_{int(step)}.vtu")


def write_manifest(root: str, dataset_key: str, manifest: Mapping[str, Any]) -> str:
    path = manifest_path(root, dataset_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(manifest), f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return str(path)


def read_manifest(root: str, dataset_key: str) -> Dict[str, Any]:
    path = manifest_path(root, dataset_key)
    if not path.is_file():
        raise FileNotFoundError(f"VTK manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid VTK manifest: {path}")
    return data


def reset_dataset(root: str, dataset_key: str) -> Path:
    path = dataset_dir(root, dataset_key)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
