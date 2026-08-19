"""Shared DAT path helpers for ingest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def dat_dir(dat_path: str) -> str:
    path = Path(dat_path)
    if path.is_dir():
        return str(path)
    return str(path.parent)


def iter_dat_files(dat_path: str) -> List[str]:
    path = Path(dat_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(dat_path)
    files = [
        str(path / name)
        for name in os.listdir(path)
        if name.lower().endswith(".dat") and (path / name).is_file()
    ]
    def sort_key(filename: str):
        stem = Path(filename).stem
        token = stem.split("_", 1)[0]
        try:
            return (0, int(token), stem)
        except ValueError:
            return (1, 0, stem)
    files.sort(key=sort_key)
    if not files:
        raise FileNotFoundError(f"no .dat files under {dat_path}")
    return files


def topology_dat_file(dat_path: str) -> str:
    """Pick one DAT file for static mesh topology (lowest step if numeric names)."""
    files = iter_dat_files(dat_path)
    return files[0]
