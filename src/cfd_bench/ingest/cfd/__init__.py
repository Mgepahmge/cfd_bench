"""Canonical legacy-CFD DAT ingest helpers.

This package is intentionally separate from ``ingest.h5``.  H5/structural
ingest is a frozen interface; all legacy Tecplot refactoring lives here.
"""

from .canonical import CFDFrame, CFDZoneFrame, load_cfd_topology, iter_cfd_frames

__all__ = ["CFDFrame", "CFDZoneFrame", "load_cfd_topology", "iter_cfd_frames"]
