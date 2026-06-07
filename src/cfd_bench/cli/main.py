"""CFD-Bench root CLI."""

from __future__ import annotations

import argparse
import sys

from cfd_bench.cli.ingest_cmd import add_ingest_parser
from cfd_bench.cli.run_cmd import add_run_parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cfd-bench",
        description="CFD-Bench: ingest CFD data and run database benchmark workloads",
    )
    subparsers = ap.add_subparsers(dest="command", required=True)
    add_ingest_parser(subparsers)
    add_run_parser(subparsers)
    return ap


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
