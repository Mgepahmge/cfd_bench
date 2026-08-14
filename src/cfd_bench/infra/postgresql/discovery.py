"""Runtime metadata discovery for PostgreSQL-backed benchmark datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .config import PostgreSQLConfig


@dataclass(frozen=True)
class PostgreSQLDatasetInfo:
    dataset_key: str
    ship_type: str
    scale: str
    zone_type: str
    timesteps: Tuple[int, ...]
    variables: Tuple[str, ...]


def dataset_key(ship_type: str, scale: str) -> str:
    ship_type = str(ship_type)
    scale = str(scale or "")
    return f"{ship_type}_{scale}" if scale else ship_type


def _choose_zone(zones: Sequence[str], preferred_zone: Optional[str]) -> str:
    unique = sorted(set(str(z) for z in zones))
    if not unique:
        return preferred_zone or "0_Fluid"
    if preferred_zone and preferred_zone in unique:
        return preferred_zone
    if "0_Fluid" in unique:
        return "0_Fluid"
    fluid_like = [z for z in unique if "fluid" in z.lower()]
    if fluid_like:
        return fluid_like[0]
    return unique[0]


def group_dataset_rows(
    rows: Iterable[Tuple[str, str, str, int, str]],
    *,
    preferred_zone: Optional[str] = None,
    selected_datasets: Optional[Sequence[str]] = None,
) -> List[PostgreSQLDatasetInfo]:
    """Turn DISTINCT cell_scalar rows into per-dataset runtime metadata.

    Variables are the intersection across selected timesteps, because workloads
    such as W2 choose a variable and then query several timesteps.  Using the
    intersection prevents automatically selecting a variable that is absent in
    one of those frames.
    """

    selected = None if selected_datasets is None else set(selected_datasets)
    grouped: Dict[Tuple[str, str], Dict[str, Dict[int, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for ship, scale, zone, step, var in rows:
        key = dataset_key(ship, scale)
        if selected is not None and key not in selected:
            continue
        grouped[(str(ship), str(scale or ""))][str(zone)][int(step)].add(str(var).upper())

    out: List[PostgreSQLDatasetInfo] = []
    for (ship, scale), zones in sorted(grouped.items(), key=lambda item: dataset_key(*item[0])):
        zone = _choose_zone(list(zones.keys()), preferred_zone)
        by_step = zones[zone]
        timesteps = tuple(sorted(by_step))
        var_sets = [by_step[s] for s in timesteps if by_step[s]]
        common = set.intersection(*var_sets) if var_sets else set()
        # A one-frame dataset naturally uses every variable in that frame.
        variables = tuple(sorted(common))
        out.append(
            PostgreSQLDatasetInfo(
                dataset_key=dataset_key(ship, scale),
                ship_type=ship,
                scale=scale,
                zone_type=zone,
                timesteps=timesteps,
                variables=variables,
            )
        )
    return out


def discover_postgresql_datasets(
    config: Optional[PostgreSQLConfig] = None,
    *,
    preferred_zone: Optional[str] = None,
    selected_datasets: Optional[Sequence[str]] = None,
) -> List[PostgreSQLDatasetInfo]:
    """Discover runnable datasets, steps and variables directly from PostgreSQL."""

    cfg = config or PostgreSQLConfig()
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "PostgreSQL discovery requires psycopg2. Install with: pip install 'cfd_bench[postgresql]'"
        ) from exc

    conn = psycopg2.connect(
        database=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
        host=cfg.db_host,
        port=cfg.db_port,
    )
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT ship_type, scale, zone_type, timestep, var
                FROM cell_scalar
                ORDER BY ship_type, scale, zone_type, timestep, var
                """
            )
            rows = cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()

    return group_dataset_rows(
        rows,
        preferred_zone=preferred_zone,
        selected_datasets=selected_datasets,
    )
