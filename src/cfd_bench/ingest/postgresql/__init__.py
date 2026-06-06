"""PostgreSQL ingest pipeline."""

from cfd_bench.ingest.postgresql.load_topology import main as load_topology_main
from cfd_bench.ingest.postgresql.load_cell_vars import main as load_cell_vars_main

__all__ = ["load_topology_main", "load_cell_vars_main"]
