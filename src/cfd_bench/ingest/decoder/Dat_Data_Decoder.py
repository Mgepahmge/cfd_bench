from __future__ import annotations

from . import Zone


class CAE_Decoder:
    """Decode Tecplot-style .dat files into zone structures."""

    def __init__(self, dim: int):
        self.Title = ""
        self.Variables: list[str] = []
        self.Var_count = -1
        self.Zones: list[Zone.Zone_3D] = []
        self.N_DIM = dim

    def Decode_dat_file(self, path: str) -> None:
        with open(path, "r", encoding="UTF-8") as fh:
            raw_content = fh.read()

        paragraphs = raw_content.split("ZONE  T=")
        for line in paragraphs[0].split("\n"):
            line = line.strip()
            if line.startswith("TITLE"):
                self.Title = line.split("=", 1)[1].replace('"', "").strip()
            elif line.startswith("VARIABLES"):
                tokens = line.split("=", 1)[1].replace('"', "").split()
                self.Variables = [v for v in tokens if v]
                self.Var_count = len(self.Variables)

        # Fluid + hull zones (first two ZONE blocks).
        for i in range(1, min(len(paragraphs), 3)):
            zone = Zone.Zone_3D(paragraphs[i], self.Var_count, self.N_DIM, self.Variables)
            self.Zones.append(zone)
