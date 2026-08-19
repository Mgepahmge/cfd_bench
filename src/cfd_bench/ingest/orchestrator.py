"""Unified ingest orchestration for CFD-Bench backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from cfd_bench.ingest.common.dat_files import dat_dir, iter_dat_files, topology_dat_file


@dataclass
class IngestReport:
    success: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _normalize_backends(backends: Sequence[str]) -> List[str]:
    aliases = {
        "pg": "postgresql",
        "postgres": "postgresql",
    }
    out: List[str] = []
    for b in backends:
        key = str(b).strip().lower()
        key = aliases.get(key, key)
        if key not in out:
            out.append(key)
    return out


def _ingest_postgresql(
    dat_path: str,
    ship_type: str,
    scale: str,
    zone_indices: Sequence[int],
    *,
    init_pg_schema: bool,
    build_pg_spatial: bool,
    topology,
) -> None:
    from cfd_bench.ingest.postgresql.pg_io import load_topology_from_dat
    from cfd_bench.ingest.postgresql.build_point_locator_grid import build_point_locator_grid
    from cfd_bench.ingest.postgresql.load_cell_vars import load_cell_vars
    from cfd_bench.ingest.postgresql.schema import apply_pg_schema

    topo_dat = topology_dat_file(dat_path)
    if init_pg_schema:
        apply_pg_schema()

    load_topology_from_dat(topo_dat, ship_type, scale, zone_indices=zone_indices, topology=topology)
    load_cell_vars(dat_dir(dat_path), ship_type, scale, zone_indices=zone_indices, topology=topology)

    if build_pg_spatial:
        for zone_type in topology:
            print(f"[ingest] postgresql: building spatial layer for {zone_type}")
            # Canonical CFD W1 line/plane use cell_bounds directly; no expensive
            # per-cell PostGIS shell is required. Point buckets are still built.
            build_point_locator_grid(ship_type, scale, zone_type)


def _ingest_iotdb(
    dat_path: str,
    ship_type: str,
    scale: str,
    zone_indices: Sequence[int],
    topology,
    **session_kw,
) -> None:
    from cfd_bench.ingest.iotdb.load_cell_vars import load_cell_vars_from_dir
    from cfd_bench.ingest.iotdb.load_topology import load_topology

    topo_dat = topology_dat_file(dat_path)
    load_topology(topo_dat, ship_type, scale, zone_indices=zone_indices, topology=topology, **session_kw)
    load_cell_vars_from_dir(dat_dir(dat_path), ship_type, scale, zone_indices=zone_indices, topology=topology, **session_kw)


def _ingest_tiledb(
    dat_path: str,
    ship_type: str,
    scale: str,
    zone_indices: Sequence[int],
    tiledb_root: str,
    topology,
) -> None:
    from cfd_bench.ingest.tiledb.load_cell_vars import load_cell_vars_from_path
    from cfd_bench.ingest.tiledb.load_topology import load_topology

    topo_dat = topology_dat_file(dat_path)
    load_topology(topo_dat, ship_type, scale, tiledb_root, zone_indices=zone_indices, topology=topology)
    load_cell_vars_from_path(dat_path, ship_type, scale, tiledb_root, zone_indices=zone_indices, topology=topology)


def _ingest_vtk(dat_path: str, ship_type: str, scale: str) -> None:
    from cfd_bench.ingest.vtk.load_vtk import load_vtk_from_dir

    load_vtk_from_dir(dat_dir(dat_path), ship_type, scale)


def ingest_all(
    dat_path: str,
    ship: str,
    backends: Sequence[str],
    *,
    zone_indices: Sequence[int] = (0, 1),
    tiledb_root: str,
    init_pg_schema: bool = True,
    build_pg_spatial: bool = True,
    include_vtk: bool = False,
    iotdb_host: str = "127.0.0.1",
    iotdb_port: str = "6667",
    iotdb_user: str = "root",
    iotdb_password: str = "root",
) -> IngestReport:
    """Ingest one ship dataset into selected backends using the modern stack."""
    if not os.path.isfile(dat_path) and not os.path.isdir(dat_path):
        raise FileNotFoundError(dat_path)

    if "_" in ship:
        ship_type, scale = ship.split("_", 1)
    else:
        ship_type, scale = ship, "default"
    report = IngestReport()
    selected = _normalize_backends(backends)
    if include_vtk and "vtk" not in selected:
        selected.append("vtk")

    session_kw = dict(host=iotdb_host, port=iotdb_port, user=iotdb_user, password=iotdb_password)

    topology = None
    if any(b in selected for b in ("postgresql", "iotdb", "tiledb")):
        from cfd_bench.ingest.cfd.canonical import load_cfd_topology

        print("[ingest] parsing canonical CFD topology once ...")
        topology = load_cfd_topology(dat_path, zone_indices, show_progress=True)
        print(
            "[ingest] canonical topology ready: "
            + ", ".join(
                f"{z}(nodes={t['node_count']}, cells={t['cell_count']})"
                for z, t in topology.items()
            )
        )

    handlers = {
        "postgresql": lambda: _ingest_postgresql(
            dat_path,
            ship_type,
            scale,
            zone_indices,
            init_pg_schema=init_pg_schema,
            build_pg_spatial=build_pg_spatial,
            topology=topology,
        ),
        "iotdb": lambda: _ingest_iotdb(dat_path, ship_type, scale, zone_indices, topology, **session_kw),
        "tiledb": lambda: _ingest_tiledb(dat_path, ship_type, scale, zone_indices, tiledb_root, topology),
        "vtk": lambda: _ingest_vtk(dat_path, ship_type, scale),
    }

    for backend in selected:
        if backend not in handlers:
            report.failed.append((backend, f"unknown backend: {backend}"))
            continue
        try:
            print(f"\n[ingest] === {backend} ===")
            handlers[backend]()
            report.success.append(backend)
            print(f"[ingest] {backend}: OK")
        except Exception as exc:
            report.failed.append((backend, f"{type(exc).__name__}: {exc}"))
            print(f"[ingest] {backend}: FAILED — {exc}")

    return report
