from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np

from cfd_bench.ingest.common.dat_files import iter_dat_files, topology_dat_file
from cfd_bench.ingest.common.topology_export import export_zone_topology
from cfd_bench.ingest.decoder import CAE_Decoder


def normalize_zone_name(name: str, index: int = 0) -> str:
    return str(name or f"Zone_{index}").strip().replace(" ", "_") or f"Zone_{index}"


def step_from_dat_path(path: str) -> int:
    stem = Path(path).stem
    m = re.match(r"^\s*(\d+)", stem)
    if not m:
        raise ValueError(f"cannot infer CFD step from DAT filename: {path}")
    return int(m.group(1))


@dataclass(frozen=True)
class CFDZoneFrame:
    zone_index: int
    zone_name: str
    cell_count: int
    variables: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class CFDFrame:
    step: int
    path: str
    zones: Tuple[CFDZoneFrame, ...]


def load_cfd_topology(dat_path: str, zone_indices: Sequence[int] = (0, 1)) -> Dict[str, dict]:
    """Decode the static mesh once and return backend-neutral zone payloads."""
    topo_path = topology_dat_file(dat_path)
    decoder = CAE_Decoder(3)
    decoder.Decode_dat_file(topo_path, topology=True)
    out: Dict[str, dict] = {}
    for zi in zone_indices:
        if int(zi) >= len(decoder.Zones):
            continue
        zone = decoder.Zones[int(zi)]
        payload = export_zone_topology(zone)
        payload["zone_index"] = int(zi)
        out[payload["zone_name"]] = payload
    if not out:
        raise ValueError(
            f"no requested CFD zones found in {topo_path}; requested={list(zone_indices)} "
            f"available={len(decoder.Zones)}"
        )
    return out


def _zone_variables(decoder: CAE_Decoder, zone_index: int) -> CFDZoneFrame:
    zone = decoder.Zones[int(zone_index)]
    names = [str(v).strip().upper() for v in decoder.Variables[3:]]
    if len(names) != len(zone.Element_Variables):
        raise ValueError(
            f"zone {zone.Zone_name!r}: variable declaration/data mismatch: "
            f"names={len(names)} arrays={len(zone.Element_Variables)}"
        )
    values = {
        name: np.ascontiguousarray(zone.Element_Variables[i], dtype=np.float64)
        for i, name in enumerate(names)
    }
    return CFDZoneFrame(
        zone_index=int(zone_index),
        zone_name=normalize_zone_name(zone.Zone_name, int(zone_index)),
        cell_count=int(zone.Element_count),
        variables=values,
    )


def iter_cfd_frames(dat_path: str, zone_indices: Sequence[int] = (0, 1)) -> Iterator[CFDFrame]:
    """Stream cell-centred result frames without rebuilding face topology."""
    for path in iter_dat_files(dat_path):
        decoder = CAE_Decoder(3)
        decoder.Decode_dat_file(path, topology=False)
        zones = []
        for zi in zone_indices:
            if int(zi) >= len(decoder.Zones):
                continue
            zones.append(_zone_variables(decoder, int(zi)))
        yield CFDFrame(step=step_from_dat_path(path), path=path, zones=tuple(zones))


def validate_frame_topology(frame: CFDFrame, topology: Mapping[str, dict]) -> None:
    for zone in frame.zones:
        topo = topology.get(zone.zone_name)
        if topo is None:
            raise ValueError(
                f"step {frame.step}: result zone {zone.zone_name!r} is absent from static topology"
            )
        expected = int(topo["cell_count"])
        if zone.cell_count != expected:
            raise ValueError(
                f"step {frame.step} zone {zone.zone_name}: cell count changed "
                f"from topology={expected} to frame={zone.cell_count}"
            )
        for var, values in zone.variables.items():
            if len(values) != expected:
                raise ValueError(
                    f"step {frame.step} zone {zone.zone_name} var {var}: "
                    f"expected {expected} cell values, got {len(values)}"
                )


def topology_edges(topo: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for cid, nbrs in enumerate(topo.get("adjacency", ())):
        for nb in nbrs:
            nb = int(nb)
            if nb > cid:
                left.append(int(cid))
                right.append(nb)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def max_neighbor_diffs(
    topo: Mapping[str, object], variables: Mapping[str, np.ndarray]
) -> Dict[str, float]:
    """Compute W3 max-neighbour deltas once per frame, backend-independently."""
    a, b = topology_edges(topo)
    result: Dict[str, float] = {}
    for name, raw in variables.items():
        vals = np.asarray(raw, dtype=np.float64).reshape(-1)
        if a.size == 0 or vals.size == 0:
            result[str(name).upper()] = 0.0
            continue
        result[str(name).upper()] = float(np.max(np.abs(vals[a] - vals[b])))
    return result
