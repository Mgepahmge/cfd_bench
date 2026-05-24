"""VTK file path helpers for baseline backend."""

from __future__ import annotations

import os
from typing import List


def list_vtk_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".vtk")]


def resolve_vtk_file(files: List[str], ship: str, step: int) -> str:
    suffix = f"_{int(step)}.vtk"
    matches = [f for f in files if ship in f and f.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"no VTK for ship={ship} step={step}")
    return matches[0]
