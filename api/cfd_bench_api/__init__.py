"""HTTP adapter for CFD-Bench.

This package deliberately sits outside the ``cfd_bench`` core package.  The
long-running ingest/benchmark paths invoke the frozen CLI contract, while the
interactive interpolation endpoint reuses the existing interpolation engine
in-process.
"""

__version__ = "0.1.0"
