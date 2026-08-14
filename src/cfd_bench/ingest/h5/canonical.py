"""Convert HDF5 field output to the cell-centered benchmark representation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model import CanonicalFrame, CanonicalMesh, FrameInfo
from .reader import OdbH5Reader, _string_tuple


BENCHMARK_VECTOR_VARS = ("U", "V", "W")
BENCHMARK_SCALAR_VARS = ("P", "K", "E")


def _component_index(components: Sequence[str], component: str) -> int:
    wanted = str(component).upper()
    for i, name in enumerate(components):
        if str(name).upper() == wanted:
            return i
    raise KeyError(f"component {component!r} not found in {list(components)!r}")


def _ensure_matrix(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _reduce_components(values: np.ndarray, components: Sequence[str], component: Optional[str]) -> np.ndarray:
    values = _ensure_matrix(values)
    if component:
        idx = _component_index(components, component)
        if idx >= values.shape[1]:
            raise ValueError(
                f"component index {idx} exceeds stored value width {values.shape[1]}"
            )
        return values[:, idx]
    if values.shape[1] == 1:
        return values[:, 0]
    # No single component was requested: preserve a scalar by using the
    # Euclidean magnitude.  This is explicit and deterministic for vector or
    # tensor-like source fields and avoids silently selecting component 0.
    return np.linalg.norm(values, axis=1)


def _nodal_to_cell(values: np.ndarray, mesh: CanonicalMesh) -> np.ndarray:
    if len(values) != mesh.node_count:
        raise ValueError(
            f"nodal result has {len(values)} rows but mesh has {mesh.node_count} nodes; "
            "subset nodal output cannot be mapped safely without explicit labels"
        )
    out = np.empty(mesh.cell_count, dtype=np.float64)
    for cid, nodes in enumerate(mesh.cell_node_ids):
        out[cid] = float(np.mean(values[np.asarray(nodes, dtype=np.int64)]))
    return out


def _aggregate_element_real(
    raw: np.ndarray,
    n_elements: int,
    components: Sequence[str],
    component: Optional[str],
) -> np.ndarray:
    matrix = _ensure_matrix(raw)
    if n_elements <= 0:
        return np.zeros((0,), dtype=np.float64)
    if matrix.shape[0] % n_elements != 0:
        raise ValueError(
            f"element result rows={matrix.shape[0]} are not divisible by block size={n_elements}; "
            "cannot infer integration-point ownership safely"
        )
    samples_per_element = matrix.shape[0] // n_elements
    selected = _reduce_components(matrix, components, component)
    return selected.reshape(n_elements, samples_per_element).mean(axis=1)


def _collect_real_datasets(group):
    """Yield all datasets named Real below a result block, preserving paths."""
    h5py = __import__("h5py")
    out = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "Real":
            out.append((name, obj))

    group.visititems(visitor)
    return out


def field_to_nodes(
    reader: OdbH5Reader,
    mesh: CanonicalMesh,
    frame: FrameInfo,
    field_name: str,
    *,
    component: Optional[str] = None,
) -> np.ndarray:
    """Return one genuinely nodal HDF5 field as a scalar per canonical node.

    This intentionally does not interpolate element/integration-point output
    to nodes.  W11 is defined on point IDs, so only source fields that contain
    a direct ``<instance>/Real`` dataset are eligible.
    """
    with reader.open() as h5:
        field_path = f"Steps/{frame.step_name}/Frames/{frame.frame_name}/{field_name}"
        if field_path not in h5:
            raise KeyError(
                f"field {field_name!r} not present in {frame.step_name}/{frame.frame_name}"
            )
        field = h5[field_path]
        if mesh.instance_name not in field:
            raise KeyError(
                f"field {field_name!r} has no results for instance {mesh.instance_name!r}"
            )
        inst = field[mesh.instance_name]
        if "Real" not in inst:
            raise ValueError(f"field {field_name!r} is not a direct nodal field")
        components = _string_tuple(field.attrs.get("ComponentLabels"))
        raw = _ensure_matrix(inst["Real"][...])
        selected = _reduce_components(raw, components, component)
        if len(selected) != mesh.node_count:
            raise ValueError(
                f"nodal result has {len(selected)} rows but mesh has {mesh.node_count} nodes; "
                "subset nodal output cannot be mapped safely without explicit labels"
            )
        return np.asarray(selected, dtype=np.float64)


def field_to_cells(
    reader: OdbH5Reader,
    mesh: CanonicalMesh,
    frame: FrameInfo,
    field_name: str,
    *,
    component: Optional[str] = None,
) -> np.ndarray:
    """Return one source HDF5 field as a scalar per canonical cell.

    Nodal fields are averaged over each element's nodes.  Element/integration
    fields are averaged over inferred samples and section/location groups.
    """
    with reader.open() as h5:
        field_path = f"Steps/{frame.step_name}/Frames/{frame.frame_name}/{field_name}"
        if field_path not in h5:
            raise KeyError(f"field {field_name!r} not present in {frame.step_name}/{frame.frame_name}")
        field = h5[field_path]
        if mesh.instance_name not in field:
            raise KeyError(
                f"field {field_name!r} has no results for instance {mesh.instance_name!r}"
            )
        components = _string_tuple(field.attrs.get("ComponentLabels"))
        inst = field[mesh.instance_name]

        # Nodal-like result: one Real array directly under the instance.
        if "Real" in inst:
            raw = _ensure_matrix(inst["Real"][...])
            selected = _reduce_components(raw, components, component)
            return _nodal_to_cell(selected, mesh)

        # Element-like result: one group per ElementClass, optionally nested
        # under LocationIndex groups.  Each Real array is reduced to one value
        # per source element, then multiple locations are averaged.
        out = np.full(mesh.cell_count, np.nan, dtype=np.float64)
        seen = np.zeros(mesh.cell_count, dtype=np.int32)
        accum = np.zeros(mesh.cell_count, dtype=np.float64)

        for block in mesh.element_blocks:
            if block.name not in inst:
                continue
            result_block = inst[block.name]
            reals = _collect_real_datasets(result_block)
            if not reals:
                continue
            for _, ds in reals:
                vals = _aggregate_element_real(
                    ds[...], block.count, components, component
                )
                accum[block.cell_ids] += vals
                seen[block.cell_ids] += 1

        mask = seen > 0
        out[mask] = accum[mask] / seen[mask]
        if not np.any(mask):
            raise ValueError(
                f"field {field_name!r} contains no mappable Real datasets for {mesh.instance_name}"
            )
        if not np.all(mask):
            missing = np.where(~mask)[0]
            raise ValueError(
                f"field {field_name!r} is missing values for {len(missing)} canonical cells"
            )
        return out


def _choose_vector_field(infos, instance_name: str, preferred: Optional[str]) -> Optional[str]:
    """Choose a 3+ component source field without depending on HDF5 paths."""
    valid = {
        name: info
        for name, info in infos.items()
        if instance_name in info.instances and len(info.components) >= 3
    }
    if not valid:
        return None
    if preferred:
        for name in valid:
            if name.upper() == str(preferred).upper():
                return name
    # Common FE/CFD vector field names first, then a deterministic fallback.
    priorities = ("U", "V", "VELOCITY", "DISPLACEMENT")
    upper = {name.upper(): name for name in valid}
    for candidate in priorities:
        if candidate in upper:
            return upper[candidate]
    # Component labels can identify displacement/velocity even when a producer
    # uses a non-standard group name.  Do not fall back to an arbitrary tensor
    # such as stress/reaction force: silent semantic mistakes are worse than an
    # explicit one-off --vector-field override.
    for name, info in sorted(valid.items()):
        comps = {c.upper() for c in info.components}
        if {"U1", "U2", "U3"}.issubset(comps) or {"V1", "V2", "V3"}.issubset(comps):
            return name
    return None


def available_mapping(
    reader: OdbH5Reader,
    mesh: CanonicalMesh,
    frame: FrameInfo,
    *,
    vector_field: Optional[str] = None,
    scalar_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[str, Optional[str]]]:
    """Infer benchmark mappings from field metadata rather than HDF5 layout.

    The normal path needs no mapping arguments.  A three-component field is
    selected automatically for benchmark U/V/W, and exact P/K/E field names
    are used when present.  Explicit --map entries can later override only the
    exceptional targets without disabling these automatic mappings.
    """
    infos = reader.field_info(frame)
    mapping: Dict[str, Tuple[str, Optional[str]]] = {}

    chosen_vector = _choose_vector_field(infos, mesh.instance_name, vector_field)
    if chosen_vector:
        vinfo = infos[chosen_vector]
        comps = list(vinfo.components)
        # Prefer conventional <field>1/2/3 and X/Y/Z labels, otherwise use the
        # first three advertised components.  Component metadata, not paths,
        # drives the selection.
        upper = {name.upper(): name for name in comps}
        conventional = [
            upper.get(f"{chosen_vector.upper()}{i}") for i in (1, 2, 3)
        ]
        xyz = [upper.get(axis) for axis in ("X", "Y", "Z")]
        resolved = conventional if all(conventional) else xyz if all(xyz) else comps[:3]
        for target, comp in zip(BENCHMARK_VECTOR_VARS, resolved):
            mapping[target] = (chosen_vector, str(comp))

    requested_scalars = tuple(scalar_fields or BENCHMARK_SCALAR_VARS)
    info_by_upper = {name.upper(): name for name in infos}
    for target in requested_scalars:
        target_upper = str(target).upper()
        source_name = info_by_upper.get(target_upper)
        if source_name is None:
            continue
        info = infos[source_name]
        if mesh.instance_name in info.instances:
            mapping[target_upper] = (source_name, None)
    return mapping


def build_canonical_frame(
    reader: OdbH5Reader,
    mesh: CanonicalMesh,
    frame: FrameInfo,
    timestep: int,
    *,
    vector_field: Optional[str] = None,
    scalar_fields: Optional[Sequence[str]] = None,
    explicit_mapping: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
) -> CanonicalFrame:
    # Explicit mappings are overrides/additions, not a replacement for all
    # automatic mappings.  This lets e.g. --map P=S.S11 coexist with inferred
    # U/V/W/E and removes the need to describe the whole source layout.
    mapping = available_mapping(
        reader,
        mesh,
        frame,
        vector_field=vector_field,
        scalar_fields=scalar_fields,
    )
    if explicit_mapping:
        mapping.update({str(k).upper(): v for k, v in explicit_mapping.items()})

    scalars: Dict[str, np.ndarray] = {}
    node_scalars: Dict[str, np.ndarray] = {}
    source_fields: Dict[str, str] = {}
    for target, (source_field, component) in mapping.items():
        target_name = str(target).upper()
        values = field_to_cells(
            reader, mesh, frame, source_field, component=component
        )
        scalars[target_name] = values
        source_fields[target_name] = (
            f"{source_field}.{component}" if component else source_field
        )
        # Preserve real nodal output when available.  Element/integration-point
        # fields remain cell-only rather than being silently interpolated.
        try:
            node_scalars[target_name] = field_to_nodes(
                reader, mesh, frame, source_field, component=component
            )
        except ValueError:
            pass

    result = CanonicalFrame(
        info=frame,
        timestep=int(timestep),
        cell_scalars=scalars,
        node_scalars=node_scalars,
        source_fields=source_fields,
    )
    result.validate(mesh.cell_count, mesh.node_count)
    return result
