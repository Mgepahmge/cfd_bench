from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from iotdb.Session import Session

from .config import IoTDBConfig


def _field_to_value(field):
    if field is None:
        return None
    text = field.get_string_value()
    if text is None:
        return None
    return text


def _to_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v, default=-1):
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    text = str(v).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(default)


class IoTDBRepository:
    def __init__(self, config: IoTDBConfig):
        self.config = config
        self.session: Optional["Session"] = None
        self._cell_var_path_cache: Dict[Tuple[str, int, str], str] = {}
        self._h5_meta_cache: Dict[str, Dict[str, object]] = {}
        self._h5_steps_cache: Dict[str, List[int]] = {}
        self._cfd_meta_cache: Dict[str, Dict[str, object]] = {}
        self._var_range_cache: Dict[Tuple[str, int, str, str], Tuple[float, float]] = {}

    def open(self):
        if self.session is not None:
            return
        try:
            from iotdb.Session import Session
        except ImportError as exc:
            raise RuntimeError(
                "IoTDB backend requires apache-iotdb. "
                "Install with: pip install 'cfd_bench[iotdb]'"
            ) from exc
        try:
            self.session = Session(
                self.config.host,
                self.config.port,
                self.config.user,
                self.config.password,
                fetch_size=max(1024, int(self.config.query_fetch_size)),
            )
        except TypeError:
            # Older apache-iotdb clients do not expose fetch_size in the
            # constructor. Result-set level set_fetch_size() is still applied
            # by query() below when available.
            self.session = Session(self.config.host, self.config.port, self.config.user, self.config.password)
        self.session.open()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    # -------------------- path helpers --------------------
    def path_cell_vars(self, dataset_key: str, step: int, leaf: str = "cell_vars") -> str:
        return f"{self.config.root_path}.post_processing_management.{dataset_key}.step_{step}.{leaf}"

    @staticmethod
    def cell_vars_leaf_for_zone(zone: str) -> str:
        z = (zone or "0_Fluid").strip().lower()
        if "hull" in z or "wall" in z or z in ("1_hull", "hull"):
            return "cell_vars_hull"
        return "cell_vars"

    def path_cell_vars_for_zone(self, dataset_key: str, step: int, zone: str = "0_Fluid") -> str:
        return self.path_cell_vars(dataset_key, step, self.cell_vars_leaf_for_zone(zone))

    def path_node_vars(self, dataset_key: str, step: int) -> str:
        return f"{self.config.root_path}.post_processing_management.{dataset_key}.step_{step}.node_vars"

    def path_mesh_static(self, dataset_key: str, zone: str, leaf: str) -> str:
        return f"{self.config.root_path}.mesh_static.{dataset_key}.{zone}.{leaf}"

    def path_derived(self, dataset_key: str, step: int, leaf: str) -> str:
        return f"{self.config.root_path}.derived.{dataset_key}.step_{step}.{leaf}"

    def path_h5_metadata(self, dataset_key: str, leaf: str) -> str:
        return f"{self.config.root_path}.h5_metadata.{dataset_key}.{leaf}"

    def path_cfd_metadata(self, dataset_key: str, leaf: str = "dataset_meta") -> str:
        return f"{self.config.root_path}.cfd_metadata.{dataset_key}.{leaf}"

    # -------------------- generic query --------------------
    def query(self, sql: str):
        if self.session is None:
            raise RuntimeError("IoTDB session is not open")
        ds = self.session.execute_query_statement(sql)
        setter = getattr(ds, "set_fetch_size", None)
        if callable(setter):
            try:
                setter(max(1024, int(self.config.query_fetch_size)))
            except Exception:
                pass
        return ds

    @staticmethod
    def _close_dataset(ds) -> None:
        close = getattr(ds, "close_operation_handle", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _frame_to_numeric_arrays(frame, value_count: int):
        """Convert an IoTDB DataFrame batch into timestamp/value NumPy arrays.

        Tree-model SessionDataSet DataFrames normally expose ``Time`` as the
        first column.  Some client versions instead use it as the DataFrame
        index.  Accept both layouts so the fast path remains version-tolerant.
        """
        if frame is None or len(frame) == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, int(value_count)), dtype=np.float64),
            )
        ncols = int(frame.shape[1])
        value_count = int(value_count)
        columns = [str(c).strip().lower() for c in list(frame.columns)]
        time_col = None
        for i, name in enumerate(columns):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"time", "timestamp"}:
                time_col = i
                break

        if time_col is not None:
            timestamps = np.asarray(frame.iloc[:, time_col].to_numpy(), dtype=np.int64)
            value_cols = [i for i in range(ncols) if i != time_col][:value_count]
            values = np.asarray(frame.iloc[:, value_cols].to_numpy(), dtype=np.float64)
        elif ncols == value_count + 1:
            timestamps = np.asarray(frame.iloc[:, 0].to_numpy(), dtype=np.int64)
            values = np.asarray(frame.iloc[:, 1 : 1 + value_count].to_numpy(), dtype=np.float64)
        elif ncols >= value_count:
            timestamps = np.asarray(frame.index.to_numpy(), dtype=np.int64)
            values = np.asarray(frame.iloc[:, :value_count].to_numpy(), dtype=np.float64)
        else:
            raise ValueError(
                f"unexpected IoTDB DataFrame shape {frame.shape} for {value_count} value columns"
            )
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return timestamps, values

    def query_numeric_arrays(self, sql: str, value_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read numeric query results with the fastest client API available.

        IoTDB 2.0.8+ exposes streaming DataFrame batches specifically for
        large query results.  Older clients transparently fall back to the
        historical RowRecord iterator.  Both paths preserve the same query
        semantics and always release the server-side operation handle.
        """
        ds = self.query(sql)
        try:
            has_next_df = getattr(ds, "has_next_df", None)
            next_df = getattr(ds, "next_df", None)
            if callable(has_next_df) and callable(next_df):
                ts_parts: List[np.ndarray] = []
                value_parts: List[np.ndarray] = []
                while has_next_df():
                    frame = next_df()
                    if frame is None or len(frame) == 0:
                        continue
                    ts, vals = self._frame_to_numeric_arrays(frame, value_count)
                    if ts.size:
                        ts_parts.append(ts)
                        value_parts.append(vals)
                if not ts_parts:
                    return (
                        np.zeros((0,), dtype=np.int64),
                        np.zeros((0, int(value_count)), dtype=np.float64),
                    )
                return np.concatenate(ts_parts), np.concatenate(value_parts, axis=0)

            to_df = getattr(ds, "todf", None)
            if callable(to_df) and int(value_count) <= 3:
                frame = to_df()
                return self._frame_to_numeric_arrays(frame, value_count)

            timestamps: List[int] = []
            values: List[List[float]] = []
            while ds.has_next():
                row = ds.next()
                fields = row.get_fields()
                if len(fields) < int(value_count):
                    continue
                timestamps.append(int(row.get_timestamp()))
                values.append([_to_float(_field_to_value(fields[i])) for i in range(int(value_count))])
            if not timestamps:
                return (
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0, int(value_count)), dtype=np.float64),
                )
            return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)
        finally:
            self._close_dataset(ds)

    def query_rows(self, sql: str) -> List[Tuple[int, List[str]]]:
        ds = self.query(sql)
        out: List[Tuple[int, List[str]]] = []
        try:
            while ds.has_next():
                row = ds.next()
                fields = [_field_to_value(x) for x in row.get_fields()]
                out.append((int(row.get_timestamp()), fields))
            return out
        finally:
            # IoTDB query handles are server-side resources.  Leaving them
            # open across many benchmark transactions can progressively slow
            # later workloads in the same session/process.
            self._close_dataset(ds)

    @staticmethod
    def _time_runs(sorted_ids: np.ndarray) -> List[Tuple[int, int]]:
        if sorted_ids.size == 0:
            return []
        cuts = np.flatnonzero(np.diff(sorted_ids) != 1) + 1
        chunks = np.split(sorted_ids, cuts)
        return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size]

    def _selection_predicates(self, cell_ids: Sequence[int]) -> List[Tuple[str, bool]]:
        """Plan compact IoTDB time predicates for arbitrary cell-id selections.

        The bool flag indicates that a predicate intentionally over-reads the
        enclosing time window and therefore needs an exact NumPy membership
        filter afterwards.
        """
        ids = np.unique(np.asarray(list(cell_ids), dtype=np.int64).reshape(-1))
        if ids.size == 0:
            return []
        if ids.size <= 256:
            return [("Time IN (" + ",".join(str(int(x)) for x in ids) + ")", False)]

        runs = self._time_runs(ids)
        if len(runs) <= 128:
            predicates = []
            for start in range(0, len(runs), 64):
                terms = []
                for lo, hi in runs[start : start + 64]:
                    if lo == hi:
                        terms.append(f"Time = {lo}")
                    else:
                        terms.append(f"(Time >= {lo} AND Time <= {hi})")
                predicates.append((" OR ".join(terms), False))
            return predicates

        span = int(ids[-1]) - int(ids[0]) + 1
        density = float(ids.size) / float(max(1, span))
        # Sequential time scans are IoTDB's natural access pattern.  Once the
        # selected IDs are moderately dense, one bounded scan plus a vectorized
        # exact-membership filter is substantially cheaper than parsing a huge
        # Time IN list.
        estimated_in_queries = int((ids.size + 1999) // 2000)
        if density >= 0.02 or (estimated_in_queries > 4 and span <= 5_000_000):
            return [(f"Time >= {int(ids[0])} AND Time <= {int(ids[-1])}", True)]

        predicates: List[Tuple[str, bool]] = []
        chunk = 2000
        for start in range(0, ids.size, chunk):
            part = ids[start : start + chunk]
            predicates.append(("Time IN (" + ",".join(str(int(x)) for x in part) + ")", False))
        return predicates

    @staticmethod
    def _filter_exact_ids(
        timestamps: np.ndarray, values: np.ndarray, wanted_sorted: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if timestamps.size == 0 or wanted_sorted.size == 0:
            return timestamps[:0], values[:0]
        pos = np.searchsorted(wanted_sorted, timestamps)
        mask = pos < wanted_sorted.size
        matched = np.zeros(mask.shape, dtype=bool)
        if np.any(mask):
            matched[mask] = wanted_sorted[pos[mask]] == timestamps[mask]
        return timestamps[matched], values[matched]

    def _fetch_selected_numeric(
        self,
        path: str,
        fields: Sequence[str],
        cell_ids: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        wanted = np.unique(np.asarray(list(cell_ids), dtype=np.int64).reshape(-1))
        if wanted.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, len(fields)), dtype=np.float64),
            )
        ts_parts: List[np.ndarray] = []
        value_parts: List[np.ndarray] = []
        select = ",".join(str(x) for x in fields)
        for predicate, needs_filter in self._selection_predicates(wanted):
            ts, values = self.query_numeric_arrays(
                f"SELECT {select} FROM {path} WHERE {predicate};", len(fields)
            )
            if needs_filter:
                ts, values = self._filter_exact_ids(ts, values, wanted)
            if ts.size:
                ts_parts.append(ts)
                value_parts.append(values)
        if not ts_parts:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, len(fields)), dtype=np.float64),
            )
        ts = np.concatenate(ts_parts)
        values = np.concatenate(value_parts, axis=0)
        order = np.argsort(ts, kind="stable")
        return ts[order], values[order]

    def resolve_cell_var_path(
        self, dataset_key: str, step: int, zone: str = "0_Fluid", probe_var: str = "P"
    ) -> str:
        key = (dataset_key, int(step), zone)
        if key in self._cell_var_path_cache:
            return self._cell_var_path_cache[key]
        path = self.path_cell_vars_for_zone(dataset_key, step, zone)
        try:
            rows = self.query_rows(f"SELECT {probe_var} FROM {path} LIMIT 1;")
            if rows:
                self._cell_var_path_cache[key] = path
                return path
        except Exception:
            pass
        raise RuntimeError(
            f"未找到 IoTDB 变量路径: {path}. "
            "请先运行 cfd-bench ingest 导入 step_xxx.cell_vars。"
        )

    # -------------------- scalar queries --------------------
    def fetch_cell_scalar_map(
        self,
        dataset_key: str,
        step: int,
        var: str,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> Dict[int, float]:
        norm_ids = [] if cell_ids is None else [int(i) for i in list(cell_ids)]
        if not norm_ids:
            return {}
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        ts, values = self._fetch_selected_numeric(path, [var], norm_ids)
        return {int(cid): float(value) for cid, value in zip(ts, values[:, 0])}

    def fetch_cell_scalar_values(
        self,
        dataset_key: str,
        step: int,
        var: str,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> np.ndarray:
        """Return scalar values aligned with the requested cell-id order."""
        ids = np.asarray(list(cell_ids), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.float64)
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        ts, values = self._fetch_selected_numeric(path, [var], ids)
        out = np.full(ids.shape, np.nan, dtype=np.float64)
        if ts.size:
            order = np.argsort(ts, kind="stable")
            sorted_ts = ts[order]
            sorted_values = values[order, 0]
            pos = np.searchsorted(sorted_ts, ids)
            mask = pos < sorted_ts.size
            if np.any(mask):
                exact = np.zeros(mask.shape, dtype=bool)
                exact[mask] = sorted_ts[pos[mask]] == ids[mask]
                out[exact] = sorted_values[pos[exact]]
        return out

    def fetch_cell_scalar_matrix(
        self,
        dataset_key: str,
        step: int,
        variables: Sequence[str],
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> np.ndarray:
        """Return several CFD scalar fields aligned to ``cell_ids`` in one read.

        Coupling can touch every structural point, so issuing one IoTDB query
        per variable (let alone per target point) is unnecessarily expensive.
        This method keeps the existing path-resolution contract but selects all
        requested variables together and aligns the returned rows to the caller's
        cell-id order.
        """

        ids = np.asarray(list(cell_ids), dtype=np.int64).reshape(-1)
        fields = [str(v).upper() for v in variables]
        if ids.size == 0:
            return np.zeros((0, len(fields)), dtype=np.float64)
        if not fields:
            return np.zeros((ids.size, 0), dtype=np.float64)
        path = self.resolve_cell_var_path(
            dataset_key, step, zone=zone, probe_var=fields[0]
        )
        ts, values = self._fetch_selected_numeric(path, fields, ids)
        out = np.full((ids.size, len(fields)), np.nan, dtype=np.float64)
        if ts.size:
            order = np.argsort(ts, kind="stable")
            sorted_ts = ts[order]
            sorted_values = values[order]
            pos = np.searchsorted(sorted_ts, ids)
            mask = pos < sorted_ts.size
            if np.any(mask):
                exact = np.zeros(mask.shape, dtype=bool)
                exact[mask] = sorted_ts[pos[mask]] == ids[mask]
                out[exact] = sorted_values[pos[exact]]
        return out

    def fetch_cell_scalar_values_contiguous(
        self,
        dataset_key: str,
        step: int,
        var: str,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> np.ndarray:
        """Read one contiguous CFD cell interval with ordinary IoTDB SQL.

        This is deliberately narrower than the v19 raw-data fast path.  The
        latter performed badly on real IoTDB servers and could leave later
        workloads under heavy server/client pressure.  W6 surface zones are
        commonly dense 0..N-1 ranges, where one normal SELECT is sufficient.
        Sparse selections fall back to the established point-query path.
        """
        return self.fetch_cell_scalar_values(
            dataset_key, step, var, cell_ids, zone=zone
        )

    def aggregate_cell_scalar_selection(
        self,
        dataset_key: str,
        step: int,
        var: str,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> Tuple[int, float, float, float]:
        """Aggregate a selected CFD cell set inside IoTDB for W2.

        W2 ultimately consumes only mean/max/min across the selected values.
        Pulling every scalar row across the network is unnecessary.  This
        method keeps the same selected cell IDs and performs COUNT/SUM/MIN/MAX
        server-side in exact-ID chunks, then combines the partial aggregates.
        No raw-data API or full-frame scan is used.
        """
        ids = np.asarray(list(cell_ids), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return 0, 0.0, np.nan, np.nan
        unique = np.unique(ids)
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)

        total_count = 0
        total_sum = 0.0
        total_min = np.inf
        total_max = -np.inf
        chunk = 10_000
        for start in range(0, unique.size, chunk):
            part = unique[start:start + chunk]
            if part.size == 0:
                continue
            contiguous = int(part[-1]) - int(part[0]) + 1 == part.size
            if contiguous:
                predicate = f"Time >= {int(part[0])} AND Time <= {int(part[-1])}"
            else:
                idx = ",".join(str(int(i)) for i in part)
                predicate = f"Time IN ({idx})"
            try:
                rows = self.query_rows(
                    f"SELECT COUNT({var}),SUM({var}),MIN_VALUE({var}),MAX_VALUE({var}) "
                    f"FROM {path} WHERE {predicate};"
                )
            except Exception:
                rows = []
            if rows and len(rows[0][1]) >= 4:
                vals = rows[0][1]
                count = _to_int(vals[0], 0)
                subtotal = _to_float(vals[1], 0.0)
                vmin = _to_float(vals[2])
                vmax = _to_float(vals[3])
            else:
                # Compatibility fallback for older IoTDB versions whose
                # aggregation grammar differs.  Keep the fallback scoped to
                # this chunk rather than reverting to a full-frame raw read.
                values = self.fetch_cell_scalar_map(
                    dataset_key, step, var, part.tolist(), zone=zone
                )
                arr = np.asarray(
                    [values.get(int(cid), np.nan) for cid in part],
                    dtype=np.float64,
                )
                arr = arr[np.isfinite(arr)]
                count = int(arr.size)
                subtotal = float(np.sum(arr)) if count else 0.0
                vmin = float(np.min(arr)) if count else np.nan
                vmax = float(np.max(arr)) if count else np.nan
            if count <= 0:
                continue
            total_count += count
            total_sum += subtotal
            if np.isfinite(vmin):
                total_min = min(total_min, vmin)
            if np.isfinite(vmax):
                total_max = max(total_max, vmax)

        if total_count <= 0:
            return 0, 0.0, np.nan, np.nan
        return (
            int(total_count),
            float(total_sum),
            float(total_min),
            float(total_max),
        )

    def fetch_cell_ids_by_var_range(
        self,
        dataset_key: str,
        step: int,
        var: str,
        lower: float,
        upper: float,
        zone: str = "0_Fluid",
    ) -> List[int]:
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        sql = (
            f"SELECT {var} FROM {path} "
            f"WHERE {var} >= {float(lower)} AND {var} <= {float(upper)};"
        )
        timestamps, _ = self.query_numeric_arrays(sql, 1)
        return timestamps.astype(np.int64, copy=False).tolist()

    def fetch_velocity_map(
        self,
        dataset_key: str,
        step: int,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> Dict[int, Tuple[float, float, float]]:
        norm_ids = [int(i) for i in cell_ids]
        if not norm_ids:
            return {}
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var="U")
        ts, values = self._fetch_selected_numeric(path, ["U", "V", "W"], norm_ids)
        return {
            int(cid): (float(row[0]), float(row[1]), float(row[2]))
            for cid, row in zip(ts, values)
        }

    def fetch_velocity_values(
        self,
        dataset_key: str,
        step: int,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> np.ndarray:
        ids = np.asarray(list(cell_ids), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var="U")
        ts, values = self._fetch_selected_numeric(path, ["U", "V", "W"], ids)
        out = np.full((ids.size, 3), np.nan, dtype=np.float64)
        if ts.size:
            order = np.argsort(ts, kind="stable")
            sorted_ts = ts[order]
            sorted_values = values[order]
            pos = np.searchsorted(sorted_ts, ids)
            mask = pos < sorted_ts.size
            if np.any(mask):
                exact = np.zeros(mask.shape, dtype=bool)
                exact[mask] = sorted_ts[pos[mask]] == ids[mask]
                out[exact] = sorted_values[pos[exact]]
        return out

    # -------------------- mesh static --------------------

    def fetch_cells_arrays(self, dataset_key: str, zone: str):
        """Return compact NumPy cell geometry arrays for runtime hot paths."""
        path = self.path_mesh_static(dataset_key, zone, "cells")
        cell_count = None
        try:
            meta_path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
            rows = self.query_rows(f"SELECT cell_count FROM {meta_path} LIMIT 1;")
            if rows and rows[0][1]:
                cell_count = _to_int(rows[0][1][0], -1)
        except Exception:
            cell_count = None

        fields = "cx,cy,cz,xmin,xmax,ymin,ymax,zmin,zmax,cell_type"
        parts_ids = []
        parts_centers = []
        parts_mins = []
        parts_maxs = []
        parts_types = []

        def consume(sql):
            ids, values = self.query_numeric_arrays(sql, 10)
            if ids.size == 0:
                return
            parts_ids.append(ids.astype(np.int64, copy=False))
            parts_centers.append(np.ascontiguousarray(values[:, 0:3], dtype=np.float64))
            parts_mins.append(np.ascontiguousarray(values[:, [3, 5, 7]], dtype=np.float64))
            parts_maxs.append(np.ascontiguousarray(values[:, [4, 6, 8]], dtype=np.float64))
            parts_types.append(np.asarray(values[:, 9], dtype=np.int32))

        if cell_count is None or cell_count <= 0:
            consume(f"SELECT {fields} FROM {path};")
        else:
            chunk = 50000
            for start in range(0, cell_count, chunk):
                end = start + chunk
                consume(f"SELECT {fields} FROM {path} WHERE Time >= {start} AND Time < {end};")

        if not parts_ids:
            return (
                np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
            )
        ids = np.concatenate(parts_ids)
        centers = np.concatenate(parts_centers, axis=0)
        mins = np.concatenate(parts_mins, axis=0)
        maxs = np.concatenate(parts_maxs, axis=0)
        types = np.concatenate(parts_types)
        order = np.argsort(ids, kind="stable")
        return (
            ids[order].astype(np.int32, copy=False),
            np.ascontiguousarray(centers[order], dtype=np.float64),
            np.ascontiguousarray(mins[order], dtype=np.float64),
            np.ascontiguousarray(maxs[order], dtype=np.float64),
            types[order].astype(np.int32, copy=False),
        )

    def fetch_cells(self, dataset_key: str, zone: str) -> Dict[int, Tuple[float, ...]]:
        path = self.path_mesh_static(dataset_key, zone, "cells")
        out: Dict[int, Tuple[float, ...]] = {}
        # Read mesh_meta first to enable chunked loading, avoiding full-table timeout.
        cell_count = None
        try:
            meta_path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
            rows = self.query_rows(f"SELECT cell_count FROM {meta_path} LIMIT 1;")
            if rows and rows[0][1]:
                cell_count = _to_int(rows[0][1][0], -1)
        except Exception:
            cell_count = None

        if cell_count is None or cell_count <= 0:
            sql = (
                "SELECT cx,cy,cz,xmin,xmax,ymin,ymax,zmin,zmax,cell_type "
                f"FROM {path};"
            )
            for cid, vals in self.query_rows(sql):
                if len(vals) < 10:
                    continue
                out[cid] = (
                    _to_float(vals[0]),
                    _to_float(vals[1]),
                    _to_float(vals[2]),
                    _to_float(vals[3]),
                    _to_float(vals[4]),
                    _to_float(vals[5]),
                    _to_float(vals[6]),
                    _to_float(vals[7]),
                    _to_float(vals[8]),
                    _to_int(vals[9], 0),
                )
            return out

        chunk = 50000
        for start in range(0, cell_count, chunk):
            end = start + chunk
            sql = (
                "SELECT cx,cy,cz,xmin,xmax,ymin,ymax,zmin,zmax,cell_type "
                f"FROM {path} WHERE Time >= {start} AND Time < {end};"
            )
            for cid, vals in self.query_rows(sql):
                if len(vals) < 10:
                    continue
                out[cid] = (
                    _to_float(vals[0]),
                    _to_float(vals[1]),
                    _to_float(vals[2]),
                    _to_float(vals[3]),
                    _to_float(vals[4]),
                    _to_float(vals[5]),
                    _to_float(vals[6]),
                    _to_float(vals[7]),
                    _to_float(vals[8]),
                    _to_int(vals[9], 0),
                )
        return out

    def fetch_nodes(self, dataset_key: str, zone: str) -> Dict[int, Tuple[float, float, float]]:
        sql = f"SELECT x,y,z FROM {self.path_mesh_static(dataset_key, zone, 'nodes')};"
        ids, values = self.query_numeric_arrays(sql, 3)
        return {
            int(nid): (float(row[0]), float(row[1]), float(row[2]))
            for nid, row in zip(ids, values)
        }

    def fetch_nodes_arrays(
        self, dataset_key: str, zone: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return sorted dense node ids and XYZ coordinates as NumPy arrays."""

        sql = f"SELECT x,y,z FROM {self.path_mesh_static(dataset_key, zone, 'nodes')};"
        ids, values = self.query_numeric_arrays(sql, 3)
        ids = np.asarray(ids, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        if ids.size <= 1 or np.all(ids[:-1] <= ids[1:]):
            return ids, values
        order = np.argsort(ids, kind="stable")
        return ids[order], values[order]

    def fetch_mesh_meta(self, dataset_key: str, zone: str) -> Dict[str, float]:
        path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
        fields = [
            "node_count", "cell_count", "face_count",
            "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y",
            "bbox_min_z", "bbox_max_z",
        ]
        rows = self.query_rows(f"SELECT {','.join(fields)} FROM {path} LIMIT 1;")
        if not rows or not rows[0][1]:
            return {}
        vals = rows[0][1]
        return {name: _to_float(vals[i]) for i, name in enumerate(fields) if i < len(vals)}

    def fetch_nodes_subset(
        self, dataset_key: str, zone: str, node_ids: Sequence[int]
    ) -> Dict[int, Tuple[float, float, float]]:
        ids = sorted(set(int(x) for x in node_ids if int(x) >= 0))
        if not ids:
            return {}
        path = self.path_mesh_static(dataset_key, zone, "nodes")
        ts, values = self._fetch_selected_numeric(path, ["x", "y", "z"], ids)
        return {
            int(nid): (float(row[0]), float(row[1]), float(row[2]))
            for nid, row in zip(ts, values)
        }

    def fetch_cell_nodes_subset(
        self, dataset_key: str, zone: str, cell_ids: Sequence[int]
    ) -> Dict[int, List[int]]:
        ids = sorted(set(int(x) for x in cell_ids if int(x) >= 0))
        if not ids:
            return {}
        width = self._mesh_meta_int(dataset_key, zone, "max_nodes_per_cell", 16)
        field_names = [f"node_id_{i}" for i in range(width)]
        path = self.path_mesh_static(dataset_key, zone, "cell_nodes")
        ts, values = self._fetch_selected_numeric(path, field_names, ids)
        out = {}
        rows = np.asarray(values, dtype=np.int64)
        for cid, row in zip(ts, rows):
            out[int(cid)] = row[row >= 0].astype(np.int64, copy=False).tolist()
        return out

    def _mesh_meta_int(self, dataset_key: str, zone: str, field: str, default: int) -> int:
        try:
            path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
            rows = self.query_rows(f"SELECT {field} FROM {path} LIMIT 1;")
            if rows and rows[0][1]:
                value = _to_int(rows[0][1][0], default)
                return value if value > 0 else int(default)
        except Exception:
            pass
        return int(default)

    def fetch_cell_nodes(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        width = self._mesh_meta_int(dataset_key, zone, "max_nodes_per_cell", 16)
        fields = [f"node_id_{i}" for i in range(width)]
        sql = f"SELECT {','.join(fields)} FROM {self.path_mesh_static(dataset_key, zone, 'cell_nodes')};"
        out: Dict[int, List[int]] = {}
        ts, values = self.query_numeric_arrays(sql, width)
        rows = np.asarray(values, dtype=np.int64)
        for cid, row in zip(ts, rows):
            out[int(cid)] = row[row >= 0].astype(np.int64, copy=False).tolist()
        return out

    def fetch_cell_nodes_arrays(
        self, dataset_key: str, zone: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return padded connectivity without materialising a Python list per cell."""

        width = self._mesh_meta_int(dataset_key, zone, "max_nodes_per_cell", 16)
        fields = [f"node_id_{i}" for i in range(width)]
        sql = (
            f"SELECT {','.join(fields)} FROM "
            f"{self.path_mesh_static(dataset_key, zone, 'cell_nodes')};"
        )
        ids, values = self.query_numeric_arrays(sql, width)
        ids = np.asarray(ids, dtype=np.int64)
        matrix = np.asarray(values, dtype=np.int64).reshape(-1, width)
        if ids.size <= 1 or np.all(ids[:-1] <= ids[1:]):
            return ids, matrix
        order = np.argsort(ids, kind="stable")
        return ids[order], matrix[order]

    def fetch_cell_adjacency(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        width = self._mesh_meta_int(dataset_key, zone, "max_neighbors_per_cell", 16)
        fields = [f"neighbor_id_{i}" for i in range(width)]
        sql = f"SELECT {','.join(fields)} FROM {self.path_mesh_static(dataset_key, zone, 'cell_adjacency')};"
        out: Dict[int, List[int]] = {}
        try:
            ts, values = self.query_numeric_arrays(sql, width)
        except Exception:
            return out
        rows = np.asarray(values, dtype=np.int64)
        for cid, row in zip(ts, rows):
            out[int(cid)] = row[row >= 0].astype(np.int64, copy=False).tolist()
        return out

    def fetch_face_planes(self, dataset_key: str, zone: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
        sql = f"SELECT neighbor_id,nx,ny,nz,d FROM {self.path_mesh_static(dataset_key, zone, 'face_planes')};"
        out: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        for cid, vals in self.query_rows(sql):
            if len(vals) < 5:
                continue
            out.setdefault(cid, []).append(
                (_to_int(vals[0]), _to_float(vals[1]), _to_float(vals[2]), _to_float(vals[3]), _to_float(vals[4]))
            )
        return out

    def fetch_boundary_faces(
        self, dataset_key: str, zone: str, patch_name: str = "*"
    ) -> List[Tuple[int, float, float, float, float, float, float, float, float]]:
        """Read legacy CFD boundary-face rows.

        ``load_topology.py`` stores boundary faces with the face-row index as
        IoTDB Time and the *owning cell id* in the ``cell_id`` measurement.
        Older code incorrectly treated Time as cell_id and queried a
        non-existent ``patch_name`` field even though ingest writes
        ``patch_code``.  W6 therefore could not use IoTDB-native geometry
        reliably.  Keep the persisted layout unchanged and read it correctly.
        """
        path = self.path_mesh_static(dataset_key, zone, "boundary_faces")
        sql = f"SELECT cell_id,patch_code,nx,ny,nz,area,cx,cy,cz FROM {path};"
        try:
            _face_ids, values = self.query_numeric_arrays(sql, 9)
        except Exception:
            return []
        out: List[Tuple[int, float, float, float, float, float, float, float, float]] = []
        for vals in values:
            cid = int(vals[0])
            patch_code = float(vals[1])
            if cid < 0:
                continue
            if patch_name != "*":
                # Legacy DAT ingest only has a numeric patch code.  Accept
                # either the numeric string or the all-patches wildcard.
                try:
                    if float(patch_name) != patch_code:
                        continue
                except Exception:
                    continue
            out.append(
                (
                    cid,
                    patch_code,
                    float(vals[2]),
                    float(vals[3]),
                    float(vals[4]),
                    float(vals[5]),
                    float(vals[6]),
                    float(vals[7]),
                    float(vals[8]),
                )
            )
        return out

    # -------------------- legacy CFD discovery metadata --------------------
    def cfd_dataset_metadata(self, dataset_key: str) -> Dict[str, object]:
        """Read metadata written by the canonical DAT ingest path.

        This lives under ``cfd_metadata`` so the frozen H5 metadata contract is
        untouched.  Older IoTDB databases simply return an empty mapping and
        retain the historical CLI overrides/defaults.
        """
        if dataset_key in self._cfd_meta_cache:
            return dict(self._cfd_meta_cache[dataset_key])
        path = self.path_cfd_metadata(dataset_key, "dataset_meta")
        fields = [
            "is_cfd", "zone", "zones_csv", "variables_csv",
            "timesteps_csv", "node_count", "cell_count",
        ]
        try:
            rows = self.query_rows(f"SELECT {','.join(fields)} FROM {path} LIMIT 1;")
        except Exception:
            return {}
        if not rows or not rows[0][1]:
            return {}
        vals = list(rows[0][1]) + [None] * len(fields)

        def csv_at(i, *, upper=False):
            text = str(vals[i] or "").strip()
            items = [x.strip() for x in text.split(",") if x.strip()]
            return tuple(x.upper() for x in items) if upper else tuple(items)

        steps = []
        for token in csv_at(4):
            try:
                steps.append(int(token))
            except Exception:
                pass
        result = {
            "is_cfd": _to_bool(vals[0]),
            "zone": str(vals[1] or "0_Fluid"),
            "zones": csv_at(2),
            "variables": csv_at(3, upper=True),
            "timesteps": tuple(sorted(set(steps))),
            "node_count": _to_int(vals[5], 0),
            "cell_count": _to_int(vals[6], 0),
        }
        self._cfd_meta_cache[dataset_key] = dict(result)
        return result

    def fetch_cfd_element_ids_in_coordinate_range(
        self, dataset_key: str, zone: str, lower: Sequence[float], upper: Sequence[float]
    ) -> List[int]:
        """Return implicit one-based Tecplot element IDs by centroid range."""
        lo = [float(x) for x in lower]
        hi = [float(x) for x in upper]
        path = self.path_mesh_static(dataset_key, zone, "cells")
        rows = self.query_rows(
            "SELECT cx FROM " + path +
            f" WHERE cx >= {lo[0]} AND cx <= {hi[0]}"
            f" AND cy >= {lo[1]} AND cy <= {hi[1]}"
            f" AND cz >= {lo[2]} AND cz <= {hi[2]};"
        )
        return sorted(int(cell_id) + 1 for cell_id, _ in rows)

    def fetch_cfd_frame_statistics(
        self,
        dataset_key: str,
        zone: str,
        step: int,
        attribute_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Cell-centered statistics for one legacy CFD frame."""
        meta = self.cfd_dataset_metadata(dataset_key)
        variables = [str(v).upper() for v in meta.get("variables", ())]
        if attribute_name is not None:
            wanted = str(attribute_name).upper()
            variables = [wanted] if wanted in variables else []
        out: Dict[str, Dict[str, object]] = {}
        if not variables:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no CFD values for frame={step}{suffix}")
        path = self.resolve_cell_var_path(
            dataset_key, int(step), zone=zone, probe_var=variables[0]
        )
        for var in variables:
            rows = []
            try:
                rows = self.query_rows(
                    f"SELECT COUNT({var}),MIN_VALUE({var}),MAX_VALUE({var}),"
                    f"AVG({var}),STDDEV_POP({var}) FROM {path};"
                )
            except Exception:
                rows = []
            if rows and len(rows[0][1]) >= 5:
                vals = rows[0][1]
                out[var] = {
                    "position": "cell",
                    "count": _to_int(vals[0], 0),
                    "min": _to_float(vals[1]),
                    "max": _to_float(vals[2]),
                    "mean": _to_float(vals[3]),
                    "stddev": _to_float(vals[4], 0.0),
                }
                continue
            try:
                raw = self.query_rows(f"SELECT {var} FROM {path};")
            except Exception:
                continue
            arr = np.asarray(
                [_to_float(vals[0]) for _, vals in raw if vals], dtype=np.float64
            )
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[var] = {
                    "position": "cell",
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "stddev": float(np.std(arr)),
                }
        if not out:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no CFD values for frame={step}{suffix}")
        return out

    # -------------------- H5 metadata / W9-W11 --------------------
    def h5_dataset_metadata(self, dataset_key: str) -> Dict[str, object]:
        if dataset_key in self._h5_meta_cache:
            return dict(self._h5_meta_cache[dataset_key])
        path = self.path_h5_metadata(dataset_key, "dataset_meta")
        fields = [
            "is_h5", "zone", "part_name", "instance_name", "variables_csv",
            "nodal_variables_csv", "common_variables_csv", "common_nodal_variables_csv",
            "element_types_csv", "node_count", "cell_count",
        ]
        rows = self.query_rows(f"SELECT {','.join(fields)} FROM {path} LIMIT 1;")
        if not rows or not rows[0][1]:
            return {}
        vals = list(rows[0][1]) + [None] * len(fields)
        def csv_at(i):
            text = str(vals[i] or "").strip()
            return tuple(x for x in (p.strip().upper() for p in text.split(",")) if x)
        result = {
            "is_h5": _to_bool(vals[0]),
            "zone": str(vals[1] or "0_Fluid"),
            "part_name": str(vals[2] or ""),
            "instance_name": str(vals[3] or ""),
            "variables": csv_at(4),
            "nodal_variables": csv_at(5),
            "common_variables": csv_at(6),
            "common_nodal_variables": csv_at(7),
            "element_types": tuple(x for x in str(vals[8] or "").split(",") if x),
            "node_count": _to_int(vals[9], 0),
            "cell_count": _to_int(vals[10], 0),
        }
        self._h5_meta_cache[dataset_key] = dict(result)
        return result

    def h5_frame_timesteps(self, dataset_key: str) -> List[int]:
        if dataset_key in self._h5_steps_cache:
            return list(self._h5_steps_cache[dataset_key])
        path = self.path_h5_metadata(dataset_key, "frames")
        rows = self.query_rows(f"SELECT frame_index FROM {path};")
        result = sorted(int(ts) for ts, _ in rows)
        self._h5_steps_cache[dataset_key] = list(result)
        return result

    def is_h5_dataset(self, dataset_key: str) -> bool:
        try:
            return bool(self.h5_dataset_metadata(dataset_key).get("is_h5"))
        except Exception:
            return False

    def fetch_h5_structure_nodes(
        self, dataset_key: str, zone: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return dense ids, original H5 node labels, and XYZ coordinates.

        H5 ingest stores source labels separately from the dense benchmark node
        ids.  Coupling keeps both identifiers so the output can be mapped back
        to the original structural model without reopening the source H5 file.
        """

        meta = self.h5_dataset_metadata(dataset_key)
        if not meta or not bool(meta.get("is_h5")):
            raise ValueError(f"dataset={dataset_key!r} is not an H5 dataset")
        resolved_zone = str(zone or meta.get("zone") or "0_Fluid")
        node_ids, coordinates = self.fetch_nodes_arrays(dataset_key, resolved_zone)
        if node_ids.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float64),
            )

        source_path = self.path_mesh_static(dataset_key, resolved_zone, "node_source")
        source_ids, source_values = self.query_numeric_arrays(
            f"SELECT source_label FROM {source_path};", 1
        )
        source_ids = np.asarray(source_ids, dtype=np.int64)
        source_values = np.asarray(source_values, dtype=np.float64).reshape(-1)
        source_labels = np.full(node_ids.shape, -1, dtype=np.int64)
        if source_ids.size:
            order = np.argsort(source_ids, kind="stable")
            source_ids = source_ids[order]
            source_values = source_values[order]
            pos = np.searchsorted(source_ids, node_ids)
            valid = pos < source_ids.size
            exact = np.zeros(valid.shape, dtype=bool)
            if np.any(valid):
                exact[valid] = source_ids[pos[valid]] == node_ids[valid]
            source_labels[exact] = np.rint(source_values[pos[exact]]).astype(np.int64)

        if np.any(source_labels < 0):
            missing = int(np.count_nonzero(source_labels < 0))
            raise RuntimeError(
                f"H5 dataset={dataset_key!r} zone={resolved_zone!r} is missing "
                f"source labels for {missing} structural nodes"
            )
        return node_ids, source_labels, coordinates

    def fetch_var_value_range(
        self, dataset_key: str, step: int, var: str, zone: str = "0_Fluid"
    ) -> Tuple[float, float]:
        key = (str(dataset_key), int(step), str(zone), str(var).upper())
        cached = self._var_range_cache.get(key)
        if cached is not None:
            return cached
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        result = None
        try:
            rows = self.query_rows(
                f"SELECT MIN_VALUE({var}),MAX_VALUE({var}) FROM {path};"
            )
            if rows and len(rows[0][1]) >= 2:
                result = (_to_float(rows[0][1][0]), _to_float(rows[0][1][1]))
        except Exception:
            pass
        if result is None:
            values = [
                _to_float(vals[0])
                for _, vals in self.query_rows(f"SELECT {var} FROM {path};")
                if vals and np.isfinite(_to_float(vals[0]))
            ]
            if not values:
                raise ValueError(f"no values for {dataset_key} step={step} var={var}")
            result = (float(np.min(values)), float(np.max(values)))
        self._var_range_cache[key] = (float(result[0]), float(result[1]))
        return self._var_range_cache[key]

    def fetch_max_diffs(
        self, dataset_key: str, step: int, variables: Sequence[str]
    ) -> Dict[str, float]:
        vars_ = [str(v).upper() for v in variables]
        if not vars_:
            return {}
        path = self.path_derived(dataset_key, step, "max_diff")
        try:
            rows = self.query_rows(f"SELECT {','.join(vars_)} FROM {path} WHERE Time=0;")
        except Exception:
            return {}
        if not rows:
            return {}
        vals = rows[0][1]
        out = {}
        for var, value in zip(vars_, vals):
            fv = _to_float(value)
            if np.isfinite(fv):
                out[var] = float(fv)
        return out

    def fetch_h5_element_ids_in_coordinate_range(
        self, dataset_key: str, zone: str, lower: Sequence[float], upper: Sequence[float]
    ) -> List[int]:
        lo = [float(x) for x in lower]
        hi = [float(x) for x in upper]
        cells = self.path_mesh_static(dataset_key, zone, "cells")
        rows = self.query_rows(
            "SELECT cx,cy,cz FROM " + cells +
            f" WHERE cx >= {lo[0]} AND cx <= {hi[0]}"
            f" AND cy >= {lo[1]} AND cy <= {hi[1]}"
            f" AND cz >= {lo[2]} AND cz <= {hi[2]};"
        )
        cell_ids = [int(ts) for ts, _ in rows]
        if not cell_ids:
            return []
        src = self.path_mesh_static(dataset_key, zone, "cell_source")
        idx = ",".join(str(x) for x in cell_ids)
        labels = self.query_rows(f"SELECT source_label FROM {src} WHERE Time IN ({idx});")
        return sorted(_to_int(vals[0]) for _, vals in labels if vals)

    def fetch_h5_point_ids(self, dataset_key: str, zone: str) -> List[int]:
        path = self.path_mesh_static(dataset_key, zone, "node_source")
        rows = self.query_rows(f"SELECT source_label FROM {path};")
        return sorted(_to_int(vals[0]) for _, vals in rows if vals)

    def _dense_node_ids_for_source_labels(
        self, dataset_key: str, zone: str, source_labels: Sequence[int]
    ) -> Dict[int, int]:
        wanted = {int(x) for x in source_labels}
        if not wanted:
            return {}
        # H5 source labels need not be dense.  Filter by label in IoTDB so W11
        # does not scan the full node-source mapping on every transaction.
        path = self.path_mesh_static(dataset_key, zone, "node_source")
        labels = ",".join(str(x) for x in sorted(wanted))
        try:
            rows = self.query_rows(
                f"SELECT source_label FROM {path} WHERE source_label IN ({labels});"
            )
        except Exception:
            rows = self.query_rows(f"SELECT source_label FROM {path};")
        out: Dict[int, int] = {}
        for dense_id, vals in rows:
            if not vals:
                continue
            label = _to_int(vals[0])
            if label in wanted:
                out[label] = int(dense_id)
        return out

    def fetch_h5_point_frame_extrema(
        self, dataset_key: str, zone: str, source_labels: Sequence[int], var: str
    ) -> Dict[int, Tuple[float, float]]:
        mapping = self._dense_node_ids_for_source_labels(dataset_key, zone, source_labels)
        if not mapping:
            return {}
        dense_to_source = {dense: src for src, dense in mapping.items()}
        dense_ids = sorted(dense_to_source)
        idx = ",".join(str(x) for x in dense_ids)
        mins = {src: np.inf for src in mapping}
        maxs = {src: -np.inf for src in mapping}
        for step in self.h5_frame_timesteps(dataset_key):
            path = self.path_node_vars(dataset_key, step)
            try:
                rows = self.query_rows(f"SELECT {var} FROM {path} WHERE Time IN ({idx});")
            except Exception:
                continue
            for dense_id, vals in rows:
                if not vals:
                    continue
                value = _to_float(vals[0])
                if not np.isfinite(value) or dense_id not in dense_to_source:
                    continue
                src = dense_to_source[dense_id]
                mins[src] = min(mins[src], value)
                maxs[src] = max(maxs[src], value)
        return {
            src: (float(mins[src]), float(maxs[src]))
            for src in mapping
            if np.isfinite(mins[src]) and np.isfinite(maxs[src])
        }

    def fetch_frame_statistics(
        self, dataset_key: str, zone: str, step: int, attribute_name: Optional[str] = None
    ) -> Dict[str, Dict[str, object]]:
        meta = self.h5_dataset_metadata(dataset_key)
        all_vars = list(meta.get("variables", ()))
        nodal = set(meta.get("nodal_variables", ()))
        if attribute_name is not None:
            wanted = str(attribute_name).upper()
            all_vars = [wanted] if wanted in all_vars else []
        out: Dict[str, Dict[str, object]] = {}
        for var in all_vars:
            position = "node" if var in nodal else "cell"
            path = self.path_node_vars(dataset_key, step) if position == "node" else self.path_cell_vars_for_zone(dataset_key, step, zone)
            rows = []
            try:
                rows = self.query_rows(
                    f"SELECT COUNT({var}),MIN_VALUE({var}),MAX_VALUE({var}),"
                    f"AVG({var}),STDDEV_POP({var}) FROM {path};"
                )
            except Exception:
                rows = []
            if rows and len(rows[0][1]) >= 5:
                vals = rows[0][1]
                out[var] = {
                    "position": position,
                    "count": _to_int(vals[0], 0),
                    "min": _to_float(vals[1]),
                    "max": _to_float(vals[2]),
                    "mean": _to_float(vals[3]),
                    "stddev": _to_float(vals[4], 0.0),
                }
                continue
            # Compatibility fallback for older IoTDB servers lacking
            # STDDEV_POP: fetch this one series and compute the same statistics.
            try:
                raw = self.query_rows(f"SELECT {var} FROM {path};")
            except Exception:
                continue
            arr = np.asarray(
                [_to_float(vals[0]) for _, vals in raw if vals], dtype=np.float64
            )
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[var] = {
                    "position": position,
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "stddev": float(np.std(arr)),
                }
        if not out:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no values for frame={step}{suffix}")
        return out

