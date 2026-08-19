from __future__ import annotations

import re
from typing import Callable, List, Optional

from . import Zone


class CAE_Decoder:
    """Decode Tecplot ASCII FE ``.dat`` files used by the CFD benchmark.

    Public attributes intentionally match the historical decoder.  Parsing is
    whitespace-tolerant and all ZONE blocks are retained instead of silently
    truncating after the first two.
    """

    _ZONE_RE = re.compile(r"(?im)^\s*ZONE\s+T\s*=\s*")

    def __init__(self, dim: int):
        self.Title = ""
        self.Variables: List[str] = []
        self.Var_count = -1
        self.Zones: List[Zone.Zone_3D] = []
        self.N_DIM = int(dim)

    @staticmethod
    def _parse_variables(preamble: str) -> List[str]:
        m = re.search(
            r"(?is)\bVARIABLES\s*=\s*(.*?)(?=^\s*ZONE\b)",
            preamble,
            flags=re.MULTILINE,
        )
        text = m.group(1) if m else ""
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return [x.strip() for x in quoted if x.strip()]
        # Fallback for unquoted/simple files.
        return [x for x in re.split(r"[\s,]+", text.strip()) if x]

    def Decode_dat_file(
        self,
        path: str,
        *,
        topology: bool = True,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        if progress is not None:
            progress("read DAT file", 0, 1)
        with open(path, "r", encoding="UTF-8", errors="replace") as fh:
            raw_content = fh.read()
        if progress is not None:
            progress("read DAT file", 1, 1)

        zone_matches = list(self._ZONE_RE.finditer(raw_content))
        preamble = raw_content[: zone_matches[0].start()] if zone_matches else raw_content

        title = re.search(r'(?im)^\s*TITLE\s*=\s*"?([^"\n\r]+)"?', preamble)
        self.Title = title.group(1).strip() if title else ""
        self.Variables = self._parse_variables(raw_content if zone_matches else preamble)
        self.Var_count = len(self.Variables)
        if self.Var_count < self.N_DIM:
            raise ValueError(
                f"DAT file {path!r} declares only {self.Var_count} variables; expected at least {self.N_DIM}"
            )
        if not zone_matches:
            raise ValueError(f"DAT file {path!r} contains no ZONE blocks")

        self.Zones = []
        for i, match in enumerate(zone_matches):
            start = match.end()
            end = zone_matches[i + 1].start() if i + 1 < len(zone_matches) else len(raw_content)
            block = raw_content[start:end]
            zone_progress = None
            if progress is not None:
                prefix = f"zone {i + 1}/{len(zone_matches)}"
                zone_progress = lambda phase, current, total, _p=prefix: progress(
                    f"{_p} {phase}", current, total
                )
            self.Zones.append(
                Zone.Zone_3D(
                    block, self.Var_count, self.N_DIM, self.Variables,
                    parse_topology=topology, progress=zone_progress,
                )
            )
