#!/usr/bin/env python3
"""Convert a CFD-Bench coupling result H5 into a flat CSV file.

This utility intentionally lives outside the core coupling path.  The canonical
coupling result remains HDF5; CSV is only an optional convenience export.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_STATUS_CODES = {
    0: "PASS",
    1: "OUTSIDE_MESH",
    2: "NO_CONTAINING_CELL",
    3: "INTERPOLATION_FAILED",
}


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_h5", help="Coupling result produced by cfd-bench couple")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default: input basename + .csv)")
    ap.add_argument("--variables", nargs="+", default=None, help="Subset of CFD variables to export")
    ap.add_argument("--chunk-size", type=int, default=100000, help="Rows processed per chunk (default: 100000)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.chunk_size) <= 0:
        raise SystemExit("--chunk-size must be > 0")
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required: pip install h5py") from exc

    source = Path(args.input_h5).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else source.with_suffix(".csv")

    with h5py.File(source, "r") as h5:
        if str(h5.attrs.get("format", "")) != "cfd-bench-structure-cfd-coupling-v1":
            raise SystemExit(f"not a CFD-Bench coupling result: {source}")
        available = list(h5["values"].keys())
        try:
            recorded = json.loads(str(h5["metadata"].attrs.get("variables_json", "[]")))
            if recorded:
                available = [str(v) for v in recorded if str(v) in h5["values"]]
        except Exception:
            pass
        variables = [str(v).upper() for v in args.variables] if args.variables else available
        missing = [v for v in variables if v not in available]
        if missing:
            raise SystemExit(f"variables not present in result: {missing}; available={available}")

        status_codes = dict(DEFAULT_STATUS_CODES)
        raw_codes = h5["diagnostics/status"].attrs.get("codes_json")
        if raw_codes:
            try:
                status_codes.update({int(k): str(v) for k, v in json.loads(str(raw_codes)).items()})
            except Exception:
                pass

        node_id = h5["nodes/node_id"]
        source_label = h5["nodes/source_node_label"]
        coords = h5["nodes/coordinates"]
        coupling_coords = h5["nodes/coupling_coordinates"] if "coupling_coordinates" in h5["nodes"] else None
        status = h5["diagnostics/status"]
        cell_id = h5["diagnostics/cfd_cell_id"]
        error = h5["diagnostics/reconstruction_error"]
        n = int(node_id.shape[0])

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "node_id",
                    "source_node_label",
                    "x",
                    "y",
                    "z",
                    *(["coupling_x", "coupling_y", "coupling_z"] if coupling_coords is not None else []),
                    "status",
                    "cfd_cell_id",
                    "reconstruction_error",
                    *variables,
                ]
            )
            for start in range(0, n, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), n)
                ids = np.asarray(node_id[start:end], dtype=np.int64)
                labels = np.asarray(source_label[start:end], dtype=np.int64)
                xyz = np.asarray(coords[start:end], dtype=np.float64)
                mapped_xyz = (
                    np.asarray(coupling_coords[start:end], dtype=np.float64)
                    if coupling_coords is not None
                    else None
                )
                st = np.asarray(status[start:end], dtype=np.uint8)
                cells = np.asarray(cell_id[start:end], dtype=np.int64)
                errors = np.asarray(error[start:end], dtype=np.float64)
                value_arrays = [np.asarray(h5[f"values/{v}"][start:end], dtype=np.float64) for v in variables]
                for i in range(end - start):
                    writer.writerow(
                        [
                            int(ids[i]),
                            int(labels[i]),
                            f"{xyz[i, 0]:.17g}",
                            f"{xyz[i, 1]:.17g}",
                            f"{xyz[i, 2]:.17g}",
                            *(
                                [
                                    f"{mapped_xyz[i, 0]:.17g}",
                                    f"{mapped_xyz[i, 1]:.17g}",
                                    f"{mapped_xyz[i, 2]:.17g}",
                                ]
                                if mapped_xyz is not None
                                else []
                            ),
                            status_codes.get(int(st[i]), str(int(st[i]))),
                            int(cells[i]),
                            "" if not np.isfinite(errors[i]) else f"{errors[i]:.17g}",
                            *[
                                "" if not np.isfinite(values[i]) else f"{values[i]:.17g}"
                                for values in value_arrays
                            ],
                        ]
                    )

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
