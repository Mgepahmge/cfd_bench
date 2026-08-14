"""Reader for ODB-like HDF5 result files.

Only ``.h5`` is treated as the authoritative source.  Auxiliary ``.inp`` or
spreadsheet files are not required by this reader.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model import CanonicalMesh, ElementBlock, FieldInfo, FrameInfo


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "HDF5 ingest requires h5py. Install with: pip install 'cfd_bench[h5]'"
        ) from exc
    return h5py


def _decode_scalar(value, default=""):
    """Convert the scalar/one-element array attributes used by the files."""
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    if isinstance(item, np.generic):
        return item.item()
    return item


def _attr_str(group, name: str, default: str = "") -> str:
    return str(_decode_scalar(group.attrs.get(name), default))


def _attr_int(group, name: str) -> Optional[int]:
    value = _decode_scalar(group.attrs.get(name), None)
    return None if value is None else int(value)


def _attr_float(group, name: str) -> Optional[float]:
    value = _decode_scalar(group.attrs.get(name), None)
    return None if value is None else float(value)


def _string_tuple(value) -> Tuple[str, ...]:
    if value is None:
        return ()
    arr = np.asarray(value).reshape(-1)
    out = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="replace"))
        else:
            out.append(str(item))
    return tuple(out)


def _suffix_int(text: str, prefix: str, fallback: int) -> int:
    if text.startswith(prefix):
        try:
            return int(text[len(prefix) :])
        except ValueError:
            pass
    return int(fallback)


def _is_c3d10(element_type: str) -> bool:
    """Return True for the quadratic 10-node tetrahedron family."""
    return (element_type or "").upper().startswith("C3D10")


# Abaqus C3D10 local-node ordering:
# corners 1..4, then midside nodes 5=(1,2), 6=(2,3), 7=(3,1),
# 8=(1,4), 9=(2,4), 10=(3,4).  Each quadratic triangular face has
# six nodes.  Matching complete face signatures avoids confusing a shared
# quadratic edge (two corners + one midside node) with a shared face.
_C3D10_FACE_LOCAL_IDS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 4, 5, 6),
    (0, 1, 3, 4, 8, 7),
    (1, 2, 3, 5, 9, 8),
    (2, 0, 3, 6, 7, 9),
)


def _build_c3d10_adjacency(cell_node_ids: Sequence[np.ndarray]) -> List[List[int]]:
    """Build exact face adjacency for a pure C3D10 mesh in O(n) faces."""
    face_to_cells: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for cid, nodes in enumerate(cell_node_ids):
        ids = [int(x) for x in np.asarray(nodes).tolist()]
        if len(ids) < 10:
            raise ValueError(f"C3D10 cell {cid} has {len(ids)} nodes; expected at least 10")
        for local_ids in _C3D10_FACE_LOCAL_IDS:
            key = tuple(sorted(ids[i] for i in local_ids))
            face_to_cells[key].append(cid)

    out = [set() for _ in cell_node_ids]
    for owners in face_to_cells.values():
        # A valid conforming volume mesh normally has one or two owners.
        # Connecting every pair also handles non-manifold inputs deterministically.
        unique = sorted(set(owners))
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                out[a].add(b)
                out[b].add(a)
    return [sorted(row) for row in out]


def _adjacency_min_shared_nodes(element_type: str, node_count: int) -> int:
    """Conservative shared-node threshold for non-C3D10 neighbor discovery."""
    et = (element_type or "").upper()
    if et.startswith(("B", "T2D", "T3D", "CONN")):
        return 1
    if et.startswith(("S", "M3D", "CPS", "CPE", "CAX")):
        return 2
    if et.startswith("C3D"):
        return 3
    if node_count <= 2:
        return 1
    if node_count <= 4:
        return 2
    return 3


def _build_adjacency(cell_node_ids: Sequence[np.ndarray], element_types: Sequence[str]) -> List[List[int]]:
    if len(cell_node_ids) != len(element_types):
        raise ValueError("cell_node_ids and element_types must have the same length")
    if cell_node_ids and all(_is_c3d10(t) for t in element_types):
        return _build_c3d10_adjacency(cell_node_ids)

    # Preserve the original generic rule for all existing element families.
    # In mixed meshes, C3D10-C3D10 pairs are handled by exact six-node faces
    # below while cross-family pairs retain the legacy shared-node behavior.
    node_to_cells: Dict[int, List[int]] = defaultdict(list)
    for cid, nodes in enumerate(cell_node_ids):
        for nid in set(int(x) for x in np.asarray(nodes).tolist()):
            node_to_cells[nid].append(cid)

    shared_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for cells in node_to_cells.values():
        unique = sorted(set(cells))
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                shared_counts[(a, b)] += 1

    exact_c3d10 = _build_c3d10_adjacency(
        [nodes for nodes, et in zip(cell_node_ids, element_types) if _is_c3d10(et)]
    ) if any(_is_c3d10(t) for t in element_types) else []
    # Map exact local C3D10 adjacency back to global cell ids.
    c3_global = [cid for cid, et in enumerate(element_types) if _is_c3d10(et)]
    c3_pairs = set()
    for local_a, nbrs in enumerate(exact_c3d10):
        for local_b in nbrs:
            a, b = c3_global[local_a], c3_global[local_b]
            c3_pairs.add((min(a, b), max(a, b)))

    out = [set() for _ in cell_node_ids]
    for (a, b), shared in shared_counts.items():
        if _is_c3d10(element_types[a]) and _is_c3d10(element_types[b]):
            if (a, b) not in c3_pairs:
                continue
        else:
            ta = _adjacency_min_shared_nodes(element_types[a], len(cell_node_ids[a]))
            tb = _adjacency_min_shared_nodes(element_types[b], len(cell_node_ids[b]))
            if shared < min(ta, tb):
                continue
        out[a].add(b)
        out[b].add(a)
    return [sorted(row) for row in out]


class OdbH5Reader:
    """Inspect and decode one ODB-like HDF5 file."""

    def __init__(self, path: str):
        self.path = str(Path(path))
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)

    @contextmanager
    def open(self):
        h5py = _require_h5py()
        with h5py.File(self.path, "r") as h5:
            yield h5

    def list_parts(self) -> List[str]:
        with self.open() as h5:
            return sorted(h5.get("Parts", {}).keys())

    def instance_part_map(self) -> Dict[str, str]:
        with self.open() as h5:
            root = h5.get("Assembly/Instances")
            if root is None:
                return {}
            return {name: _attr_str(group, "PartName") for name, group in root.items()}

    def list_instances(self) -> List[str]:
        return sorted(self.instance_part_map().keys())

    def resolve_instance(self, instance_name: Optional[str] = None) -> Tuple[str, str]:
        mapping = self.instance_part_map()
        if not mapping:
            raise ValueError("HDF5 file contains no Assembly/Instances")
        if instance_name is None:
            if len(mapping) == 1:
                instance_name = next(iter(mapping))
            else:
                # Prefer the unique instance that actually owns field output.
                # This resolves the common case where an assembly contains
                # helper/reference instances without forcing users to know the
                # internal HDF5 instance name.
                counts = {name: 0 for name in mapping}
                for frame in self.iter_frames():
                    for info in self.field_info(frame).values():
                        for name in info.instances:
                            if name in counts:
                                counts[name] += 1
                best = max(counts.values(), default=0)
                candidates = [name for name, count in counts.items() if count == best and count > 0]
                if len(candidates) == 1:
                    instance_name = candidates[0]
                else:
                    detail = ", ".join(f"{name}({counts[name]} fields)" for name in sorted(mapping))
                    raise ValueError(
                        "HDF5 contains multiple result-bearing instances and cannot be "
                        f"selected safely: {detail}. Use --instance only to resolve this ambiguity."
                    )
        if instance_name not in mapping:
            raise KeyError(
                f"instance {instance_name!r} not found; available: {', '.join(sorted(mapping))}"
            )
        part_name = mapping[instance_name]
        if not part_name:
            raise ValueError(f"instance {instance_name!r} has no PartName attribute")
        return instance_name, part_name

    def load_mesh(self, instance_name: Optional[str] = None) -> CanonicalMesh:
        instance_name, part_name = self.resolve_instance(instance_name)
        with self.open() as h5:
            part_path = f"Parts/{part_name}"
            if part_path not in h5:
                raise KeyError(f"part {part_name!r} referenced by {instance_name!r} is missing")
            part = h5[part_path]
            nodes = part.get("Nodes")
            if nodes is None or "Labels" not in nodes or "Coordinates" not in nodes:
                raise ValueError(f"{part_path} has no complete Nodes/Labels + Coordinates")

            source_node_labels = np.asarray(nodes["Labels"][...], dtype=np.int64).reshape(-1)
            coords = np.asarray(nodes["Coordinates"][...], dtype=np.float64)
            if coords.ndim != 2 or coords.shape[0] != len(source_node_labels):
                raise ValueError(
                    f"invalid node coordinate shape {coords.shape} for {len(source_node_labels)} labels"
                )
            if coords.shape[1] < 3:
                padded = np.zeros((coords.shape[0], 3), dtype=np.float64)
                padded[:, : coords.shape[1]] = coords
                coords = padded
            elif coords.shape[1] > 3:
                coords = coords[:, :3]

            node_ids = np.arange(len(source_node_labels), dtype=np.int32)
            elements_root = part.get("Elements")
            if elements_root is None:
                raise ValueError(f"{part_path} has no Elements group")

            blocks: List[ElementBlock] = []
            source_cell_labels: List[int] = []
            cell_node_ids: List[np.ndarray] = []
            cell_element_types: List[str] = []
            block_cell_ids: Dict[str, np.ndarray] = {}
            next_cell_id = 0

            block_names = sorted(
                elements_root.keys(),
                key=lambda n: _suffix_int(n, "ElementClass:", 10**9),
            )
            for block_name in block_names:
                group = elements_root[block_name]
                if "Labels" not in group or "Connectivities" not in group:
                    continue
                labels = np.asarray(group["Labels"][...], dtype=np.int64).reshape(-1)
                conn = np.asarray(group["Connectivities"][...], dtype=np.int64)
                if conn.ndim == 1:
                    conn = conn.reshape(len(labels), -1)
                if conn.ndim != 2 or conn.shape[0] != len(labels):
                    raise ValueError(
                        f"{part_path}/Elements/{block_name}: labels={len(labels)} "
                        f"but connectivity shape={conn.shape}"
                    )
                # In these HDF5 files connectivity values are row offsets into
                # Nodes/Labels, not the source node labels themselves.
                if conn.size and (int(conn.min()) < 0 or int(conn.max()) >= len(node_ids)):
                    raise ValueError(
                        f"{part_path}/Elements/{block_name}: connectivity is outside "
                        f"node row range [0,{len(node_ids) - 1}]"
                    )
                element_type = _attr_str(group, "ElementType", block_name)
                section_category = _attr_str(group, "SectionCategory", "")
                cids = np.arange(next_cell_id, next_cell_id + len(labels), dtype=np.int32)
                block = ElementBlock(
                    name=block_name,
                    element_type=element_type,
                    source_labels=labels,
                    connectivity=conn.astype(np.int32, copy=False),
                    cell_ids=cids,
                    section_category=section_category,
                )
                blocks.append(block)
                block_cell_ids[block_name] = cids
                source_cell_labels.extend(int(x) for x in labels)
                for row in conn:
                    cell_node_ids.append(np.asarray(row, dtype=np.int32))
                    cell_element_types.append(element_type)
                next_cell_id += len(labels)

            if not blocks:
                raise ValueError(f"{part_path} contains no readable element blocks")

            cell_ids = np.arange(next_cell_id, dtype=np.int32)
            centroids = np.empty((next_cell_id, 3), dtype=np.float64)
            for cid, nids in enumerate(cell_node_ids):
                centroids[cid] = np.mean(coords[nids], axis=0)
            adjacency = _build_adjacency(cell_node_ids, cell_element_types)

            return CanonicalMesh(
                part_name=part_name,
                instance_name=instance_name,
                node_ids=node_ids,
                source_node_labels=source_node_labels,
                node_coordinates=coords,
                element_blocks=blocks,
                cell_ids=cell_ids,
                source_cell_labels=np.asarray(source_cell_labels, dtype=np.int64),
                cell_node_ids=cell_node_ids,
                cell_element_types=cell_element_types,
                cell_centroids=centroids,
                cell_adjacency=adjacency,
                block_cell_ids=block_cell_ids,
            )

    def iter_frames(self, step_names: Optional[Sequence[str]] = None) -> Iterator[FrameInfo]:
        selected = None if step_names is None else set(step_names)
        with self.open() as h5:
            steps = h5.get("Steps")
            if steps is None:
                return
            ordered_steps = sorted(
                steps.keys(),
                key=lambda n: _attr_int(steps[n], "Index") or 10**9,
            )
            for step_ordinal, step_name in enumerate(ordered_steps):
                if selected is not None and step_name not in selected:
                    continue
                step = steps[step_name]
                frames = step.get("Frames")
                if frames is None:
                    continue
                frame_names = sorted(
                    frames.keys(),
                    key=lambda n: _suffix_int(n, "Frame:", 10**9),
                )
                for frame_ordinal, frame_name in enumerate(frame_names):
                    frame = frames[frame_name]
                    yield FrameInfo(
                        step_name=step_name,
                        step_index=_attr_int(step, "Index") or (step_ordinal + 1),
                        step_domain=_attr_int(step, "Domain"),
                        frame_name=frame_name,
                        frame_index=_suffix_int(frame_name, "Frame:", frame_ordinal),
                        inc_or_mode=_attr_int(frame, "Inc/Mode"),
                        time_or_frequency=_attr_float(frame, "Time/Freq"),
                        description=_attr_str(frame, "Description", ""),
                        load_case=_attr_str(frame, "LoadCase", ""),
                    )

    def field_info(self, frame: FrameInfo) -> Dict[str, FieldInfo]:
        with self.open() as h5:
            frame_group = h5[f"Steps/{frame.step_name}/Frames/{frame.frame_name}"]
            out: Dict[str, FieldInfo] = {}
            for name, group in frame_group.items():
                # Field output groups carry ComponentLabels/Position/Type.
                if not hasattr(group, "attrs") or not hasattr(group, "keys"):
                    continue
                instances = tuple(sorted(k for k in group.keys()))
                if not instances:
                    continue
                out[name] = FieldInfo(
                    name=name,
                    components=_string_tuple(group.attrs.get("ComponentLabels")),
                    position_code=_attr_int(group, "Position"),
                    type_code=_attr_int(group, "Type"),
                    instances=instances,
                )
            return out

    def inspect(self) -> Dict[str, object]:
        instance_map = self.instance_part_map()
        frames = list(self.iter_frames())
        frame_summary = []
        for frame in frames:
            fields = self.field_info(frame)
            frame_summary.append(
                {
                    "step": frame.step_name,
                    "frame": frame.frame_name,
                    "frame_index": frame.frame_index,
                    "inc_or_mode": frame.inc_or_mode,
                    "time_or_frequency": frame.time_or_frequency,
                    "fields": {
                        name: {
                            "components": list(info.components),
                            "position_code": info.position_code,
                            "instances": list(info.instances),
                        }
                        for name, info in fields.items()
                    },
                }
            )
        return {
            "path": self.path,
            "parts": self.list_parts(),
            "instances": instance_map,
            "frames": frame_summary,
        }
