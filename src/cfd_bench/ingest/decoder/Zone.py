from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


class Zone_3D:
    """Tecplot FE zone used by the legacy CFD ingest path.

    The original project assumed one very specific text layout and an
    eight-node hexahedral mesh.  Real Tecplot FEPolyhedron files are less
    rigid: whitespace varies, cells may have a different number of nodes and
    boundary faces are represented by a zero left/right element id.  This
    decoder keeps the public attributes used by the rest of CFD-Bench but
    validates and normalises the input instead of relying on those historical
    assumptions.
    """

    _SECTION_RE = re.compile(r"(?im)^\s*#\s*([^\n\r]+)\s*$")

    def __init__(self, raw_content: str, var_count: int, dim: int, variables: Sequence[str], *, parse_topology: bool = True):
        self.Zone_name = ""
        self.Zone_type = ""
        self.Element_count = 0
        self.Face_count = 0
        self.Node_count = 0
        self.Variables = list(variables)
        self.Node_Coordinates: List[np.ndarray] = []
        self.Element_Variables: List[np.ndarray] = []
        self.Element_Coordinates: List[np.ndarray] = []
        self.NCPF = np.zeros((0,), dtype=np.int64)
        self.FN: List[np.ndarray] = []
        self.LE = np.zeros((0,), dtype=np.int64)
        self.RE = np.zeros((0,), dtype=np.int64)
        self.EN: Dict[int, np.ndarray] = {}
        self.EF: Dict[int, np.ndarray] = {}
        self.DIMENSION = int(dim)

        header, payload = self._split_header_payload(raw_content)
        self._parse_header(header)
        self._parse_values(payload, int(var_count))
        if parse_topology:
            self._parse_topology(payload)
            self.EN, self.EF = self.construct_element_face_and_nodes()
            self.decode_element_centroids_using_element_nodes()

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _split_header_payload(raw_content: str) -> Tuple[str, str]:
        # DT may contain FLOAT/DOUBLE and arbitrary whitespace.  It marks the
        # beginning of the block-packed numeric payload.
        m = re.search(r"(?is)\bDT\s*=\s*\([^)]*\)", raw_content)
        if not m:
            raise ValueError("Tecplot zone is missing DT=(...) declaration")
        return raw_content[: m.start()], raw_content[m.end() :]

    @staticmethod
    def _first_int(header: str, patterns: Sequence[str], field: str) -> int:
        for pattern in patterns:
            m = re.search(pattern, header, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                return int(m.group(1))
        raise ValueError(f"Tecplot zone is missing {field} count")

    def _parse_header(self, header: str) -> None:
        # The zone block starts immediately after ``ZONE T=``.  The first
        # quoted value is therefore the zone title.
        m = re.search(r'^\s*"([^"]+)"', header)
        if m:
            self.Zone_name = m.group(1).strip()
        else:
            first = header.strip().splitlines()[0] if header.strip() else "Zone_0"
            self.Zone_name = first.strip().strip('"') or "Zone_0"

        # Support both compact Tecplot aliases (N/E/F) and verbose
        # Nodes/Elements/Faces.  Prefer verbose forms when both are present.
        self.Node_count = self._first_int(
            header,
            [r"\bNodes\s*=\s*(\d+)", r"(?:^|[,\s])N\s*=\s*(\d+)"],
            "node",
        )
        self.Element_count = self._first_int(
            header,
            [r"\bElements\s*=\s*(\d+)", r"(?:^|[,\s])E\s*=\s*(\d+)"],
            "element",
        )
        self.Face_count = self._first_int(
            header,
            [r"\bFaces\s*=\s*(\d+)", r"(?:^|[,\s])F\s*=\s*(\d+)"],
            "face",
        )
        zt = re.search(r"\bZONETYPE\s*=\s*([A-Za-z0-9_]+)", header, flags=re.IGNORECASE)
        self.Zone_type = zt.group(1) if zt else "FEPolyhedron"

        if self.Node_count <= 0 or self.Element_count <= 0:
            raise ValueError(
                f"invalid zone counts for {self.Zone_name!r}: "
                f"nodes={self.Node_count} elements={self.Element_count}"
            )

    def _parse_values(self, payload: str, var_count: int) -> None:
        numeric_text = payload.split("#", 1)[0]
        values = np.fromstring(numeric_text, dtype=np.float64, sep=" ")
        nodal_vars = min(self.DIMENSION, var_count)
        expected = nodal_vars * self.Node_count + max(0, var_count - nodal_vars) * self.Element_count
        if values.size < expected:
            raise ValueError(
                f"zone {self.Zone_name!r}: numeric block too short: "
                f"expected at least {expected} values, got {values.size}"
            )
        # Ignore any trailing numeric tokens before the first comment only if
        # present; normal Tecplot files have an exact match.  An exact slicing
        # boundary prevents topology integers from leaking into field values.
        values = values[:expected]
        pos = 0
        for vi in range(var_count):
            count = self.Node_count if vi < nodal_vars else self.Element_count
            arr = np.ascontiguousarray(values[pos : pos + count], dtype=np.float64)
            pos += count
            if vi < nodal_vars:
                self.Node_Coordinates.append(arr)
            else:
                self.Element_Variables.append(arr)

        if len(self.Node_Coordinates) < 3:
            raise ValueError(f"zone {self.Zone_name!r}: expected X/Y/Z nodal coordinates")

    @classmethod
    def _named_sections(cls, payload: str) -> Dict[str, str]:
        matches = list(cls._SECTION_RE.finditer(payload))
        out: Dict[str, str] = {}
        for i, match in enumerate(matches):
            label = re.sub(r"\s+", " ", match.group(1).strip().lower())
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(payload)
            out[label] = payload[start:end]
        return out

    @staticmethod
    def _ints(text: str, *, offset: int = 0) -> np.ndarray:
        vals = np.fromstring(text, dtype=np.int64, sep=" ")
        if offset:
            vals = vals + int(offset)
        return vals

    def _parse_topology(self, payload: str) -> None:
        sections = self._named_sections(payload)

        def section(*names: str) -> str:
            for name in names:
                key = re.sub(r"\s+", " ", name.strip().lower())
                if key in sections:
                    return sections[key]
            return ""

        ncpf_text = section("node count per face", "node counts per face")
        if ncpf_text:
            ncpf = self._ints(ncpf_text)
            if ncpf.size != self.Face_count:
                raise ValueError(
                    f"zone {self.Zone_name!r}: expected {self.Face_count} node-count-per-face values, "
                    f"got {ncpf.size}"
                )
            self.NCPF = ncpf
        else:
            # Historical 2-D polygon fallback.  FEPolyhedron data should
            # normally contain node-count-per-face explicitly.
            self.NCPF = np.full(self.Face_count, 2, dtype=np.int64)

        face_nodes_text = section("face nodes", "face node")
        left_text = section("left elements", "left element")
        right_text = section("right elements", "right element")
        if self.Face_count and (not face_nodes_text or not left_text or not right_text):
            raise ValueError(
                f"zone {self.Zone_name!r}: incomplete face topology; required sections are "
                "# node count per face, # face nodes, # left elements, # right elements"
            )

        # Tecplot connectivity is one-based.  Element id 0 denotes the
        # exterior boundary, hence offset -1 maps it to our -1 sentinel.
        face_nodes = self._ints(face_nodes_text, offset=-1)
        expected_face_nodes = int(np.sum(self.NCPF, dtype=np.int64))
        if face_nodes.size != expected_face_nodes:
            raise ValueError(
                f"zone {self.Zone_name!r}: expected {expected_face_nodes} face-node ids, "
                f"got {face_nodes.size}"
            )
        self.FN = self.decode_face_node_array(face_nodes, self.Face_count)

        self.LE = self._ints(left_text, offset=-1)
        self.RE = self._ints(right_text, offset=-1)
        if self.LE.size != self.Face_count or self.RE.size != self.Face_count:
            raise ValueError(
                f"zone {self.Zone_name!r}: expected {self.Face_count} left/right element ids, "
                f"got left={self.LE.size} right={self.RE.size}"
            )
        for name, arr in (("left", self.LE), ("right", self.RE)):
            bad = arr[(arr < -1) | (arr >= self.Element_count)]
            if bad.size:
                raise ValueError(
                    f"zone {self.Zone_name!r}: {name} element id out of range after zero-based conversion: "
                    f"{int(bad[0])}"
                )
        if face_nodes.size:
            bad_nodes = face_nodes[(face_nodes < 0) | (face_nodes >= self.Node_count)]
            if bad_nodes.size:
                raise ValueError(
                    f"zone {self.Zone_name!r}: face node id out of range after zero-based conversion: "
                    f"{int(bad_nodes[0])}"
                )

    # ------------------------------------------------------------------
    # compatibility helpers used elsewhere in the project
    # ------------------------------------------------------------------
    def decode_regular_part(self, array_in_text, N, offset):
        result = self._ints(str(array_in_text).split("\n", 1)[-1], offset=offset)
        if N != -1 and result.size != N:
            raise ValueError(f"expected {N} values, got {result.size}")
        return result.tolist()

    def decode_face_node_array(self, face_node_array, N):
        result: List[np.ndarray] = []
        offsets = np.concatenate(([0], np.cumsum(self.NCPF, dtype=np.int64)))
        if offsets[-1] != len(face_node_array):
            raise ValueError(
                f"face-node payload mismatch: expected {int(offsets[-1])}, got {len(face_node_array)}"
            )
        for i in range(N):
            result.append(np.asarray(face_node_array[offsets[i] : offsets[i + 1]], dtype=np.int64))
        return result

    def construct_element_face_and_nodes(self):
        element_nodes = [set() for _ in range(self.Element_count)]
        element_faces = [set() for _ in range(self.Element_count)]
        for f in range(self.Face_count):
            nodes = self.FN[f]
            for cid in (int(self.LE[f]), int(self.RE[f])):
                if cid < 0 or cid >= self.Element_count:
                    continue
                element_faces[cid].add(int(f))
                element_nodes[cid].update(int(n) for n in nodes)
        en = {
            cid: np.asarray(sorted(nodes), dtype=np.int64)
            for cid, nodes in enumerate(element_nodes)
        }
        ef = {
            cid: np.asarray(sorted(faces), dtype=np.int64)
            for cid, faces in enumerate(element_faces)
        }
        return en, ef

    def decode_element_centroids_using_element_nodes(self):
        xyz = np.column_stack(self.Node_Coordinates[:3]).astype(np.float64, copy=False)
        centers = np.empty((self.Element_count, 3), dtype=np.float64)
        for cid in range(self.Element_count):
            nodes = np.asarray(self.EN.get(cid, ()), dtype=np.int64)
            if nodes.size == 0:
                raise ValueError(f"zone {self.Zone_name!r}: element {cid} has no nodes")
            centers[cid] = np.mean(xyz[nodes], axis=0)
        self.Element_Coordinates = [
            np.ascontiguousarray(centers[:, 0]),
            np.ascontiguousarray(centers[:, 1]),
            np.ascontiguousarray(centers[:, 2]),
        ]

    def construct_element_adjacency(self):
        adjacency = [set() for _ in range(self.Element_count)]
        for le_raw, re_raw in zip(self.LE, self.RE):
            le, re = int(le_raw), int(re_raw)
            if 0 <= le < self.Element_count and 0 <= re < self.Element_count and le != re:
                adjacency[le].add(re)
                adjacency[re].add(le)
        return [sorted(row) for row in adjacency]
