"""PostgreSQL ingest pipeline.

Heavy PostgreSQL dependencies are imported lazily so HDF5 inspection can run
without psycopg2 installed.
"""


def load_topology_main():
    from cfd_bench.ingest.postgresql.load_topology import main
    return main()


def load_cell_vars_main():
    from cfd_bench.ingest.postgresql.load_cell_vars import main
    return main()


__all__ = ["load_topology_main", "load_cell_vars_main"]
