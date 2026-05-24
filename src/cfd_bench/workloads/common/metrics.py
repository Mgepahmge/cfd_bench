"""Metrics and aggregation for workloads."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


def aggregation(vals: NDArray[np.float64]) -> np.float64:
    operations = ["sum", "mean", "max", "min"]
    op = random.choice(operations)
    if op == "sum":
        return np.sum(vals)
    if op == "mean":
        return np.mean(vals)
    if op == "max":
        return np.max(vals)
    return np.min(vals)


def aggregation_w2(vals: NDArray[np.float64]):
    return np.mean(vals), np.max(vals), np.min(vals)


def calculate_force(normals: NDArray[np.float64], pressures: NDArray[np.float64]) -> NDArray[np.float64]:
    total = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for i in range(len(normals)):
        total += float(pressures[i]) * normals[i]
    return total


def cal_next_point(current: NDArray[np.float64], velocity: NDArray[np.float64], delta_t: float = 0.01):
    return current + velocity * delta_t


@contextmanager
def tpm_timer(label: str) -> Iterator[dict]:
    """Context manager recording elapsed seconds for TPM reporting."""
    record = {"label": label, "elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed"] = time.perf_counter() - t0
