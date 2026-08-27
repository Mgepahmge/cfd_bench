"""Independent HDF5 result writer for structure/CFD coupling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


STATUS_PASS = 0
STATUS_OUTSIDE_MESH = 1
STATUS_NO_CONTAINING_CELL = 2
STATUS_INTERPOLATION_FAILED = 3
STATUS_CODES = {
    STATUS_PASS: "PASS",
    STATUS_OUTSIDE_MESH: "OUTSIDE_MESH",
    STATUS_NO_CONTAINING_CELL: "NO_CONTAINING_CELL",
    STATUS_INTERPOLATION_FAILED: "INTERPOLATION_FAILED",
}


class CouplingH5Writer:
    """Write one complete coupling result without modifying source datasets.

    Data is written to ``<output>.partial`` first and atomically renamed only
    after all structure nodes have been processed.  A killed/failed run therefore
    never exposes a half-written file as the canonical result.
    """

    FORMAT_VERSION = "cfd-bench-structure-cfd-coupling-v1"

    def __init__(
        self,
        output_path: str | os.PathLike,
        *,
        structure_dataset: str,
        structure_zone: str,
        cfd_dataset: str,
        cfd_zone: str,
        cfd_step: int,
        variables: Sequence[str],
        node_ids: np.ndarray,
        source_node_labels: np.ndarray,
        coordinates: np.ndarray,
        batch_size: int,
        diagnostics: bool,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "structure/CFD coupling output requires h5py; "
                "install with: pip install 'cfd_bench[h5]'"
            ) from exc

        self._h5py = h5py
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.output_path.with_name(self.output_path.name + ".partial")
        if self.partial_path.exists():
            self.partial_path.unlink()

        self.node_count = int(len(node_ids))
        self.variables = tuple(str(v).upper() for v in variables)
        self.diagnostics = bool(diagnostics)
        chunk_rows = max(1, min(int(batch_size), max(1, self.node_count)))
        self._fh = h5py.File(self.partial_path, "w")
        self._fh.attrs["format"] = self.FORMAT_VERSION

        meta = self._fh.create_group("metadata")
        meta.attrs["structure_dataset"] = str(structure_dataset)
        meta.attrs["structure_zone"] = str(structure_zone)
        meta.attrs["cfd_dataset"] = str(cfd_dataset)
        meta.attrs["cfd_zone"] = str(cfd_zone)
        meta.attrs["cfd_step"] = int(cfd_step)
        meta.attrs["variables_json"] = json.dumps(list(self.variables), separators=(",", ":"))
        meta.attrs["interpolation_method"] = "piecewise-linear barycentric"
        meta.attrs["vertex_value_source"] = "mean of incident cell-centered CFD values"
        meta.attrs["coordinate_frame"] = "as-ingested"
        meta.attrs["status_codes_json"] = json.dumps(STATUS_CODES, separators=(",", ":"))

        nodes = self._fh.create_group("nodes")
        nodes.create_dataset(
            "node_id",
            data=np.asarray(node_ids, dtype=np.int64),
            chunks=(chunk_rows,),
            compression="lzf",
        )
        nodes.create_dataset(
            "source_node_label",
            data=np.asarray(source_node_labels, dtype=np.int64),
            chunks=(chunk_rows,),
            compression="lzf",
        )
        nodes.create_dataset(
            "coordinates",
            data=np.asarray(coordinates, dtype=np.float64),
            chunks=(chunk_rows, 3),
            compression="lzf",
        )

        values = self._fh.create_group("values")
        self._value_ds = {
            var: values.create_dataset(
                var,
                shape=(self.node_count,),
                dtype="f8",
                chunks=(chunk_rows,),
                compression="lzf",
                fillvalue=np.nan,
            )
            for var in self.variables
        }

        diag = self._fh.create_group("diagnostics")
        self._status_ds = diag.create_dataset(
            "status",
            shape=(self.node_count,),
            dtype="u1",
            chunks=(chunk_rows,),
            compression="lzf",
            fillvalue=STATUS_INTERPOLATION_FAILED,
        )
        self._status_ds.attrs["codes_json"] = json.dumps(STATUS_CODES, separators=(",", ":"))
        self._cell_ds = diag.create_dataset(
            "cfd_cell_id",
            shape=(self.node_count,),
            dtype="i8",
            chunks=(chunk_rows,),
            compression="lzf",
            fillvalue=-1,
        )
        self._error_ds = diag.create_dataset(
            "reconstruction_error",
            shape=(self.node_count,),
            dtype="f8",
            chunks=(chunk_rows,),
            compression="lzf",
            fillvalue=np.nan,
        )
        self._support_ds = None
        self._weight_ds = None
        if self.diagnostics:
            self._support_ds = diag.create_dataset(
                "support_node_ids",
                shape=(self.node_count, 4),
                dtype="i8",
                chunks=(chunk_rows, 4),
                compression="lzf",
                fillvalue=-1,
            )
            self._weight_ds = diag.create_dataset(
                "weights",
                shape=(self.node_count, 4),
                dtype="f8",
                chunks=(chunk_rows, 4),
                compression="lzf",
                fillvalue=np.nan,
            )

    def write_batch(
        self,
        start: int,
        end: int,
        *,
        values: np.ndarray,
        status: np.ndarray,
        cfd_cell_ids: np.ndarray,
        reconstruction_error: np.ndarray,
        support_node_ids: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        a, b = int(start), int(end)
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (b - a, len(self.variables)):
            raise ValueError(
                f"coupling value batch has shape {matrix.shape}; "
                f"expected {(b - a, len(self.variables))}"
            )
        for j, var in enumerate(self.variables):
            self._value_ds[var][a:b] = matrix[:, j]
        self._status_ds[a:b] = np.asarray(status, dtype=np.uint8)
        self._cell_ds[a:b] = np.asarray(cfd_cell_ids, dtype=np.int64)
        self._error_ds[a:b] = np.asarray(reconstruction_error, dtype=np.float64)
        if self._support_ds is not None and support_node_ids is not None:
            self._support_ds[a:b, :] = np.asarray(support_node_ids, dtype=np.int64)
        if self._weight_ds is not None and weights is not None:
            self._weight_ds[a:b, :] = np.asarray(weights, dtype=np.float64)

    def finalize(self, summary: Mapping[str, int]) -> Path:
        meta = self._fh["metadata"]
        for key, value in summary.items():
            meta.attrs[str(key)] = int(value)
        self._fh.flush()
        self._fh.close()
        os.replace(self.partial_path, self.output_path)
        return self.output_path

    def abort(self) -> None:
        try:
            self._fh.close()
        finally:
            try:
                self.partial_path.unlink()
            except FileNotFoundError:
                pass
