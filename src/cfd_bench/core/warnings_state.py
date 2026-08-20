"""Helpers for isolating third-party changes to Python's warning filters."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def preserve_warning_filters() -> Iterator[None]:
    """Restore the caller's warning-filter state on exit.

    Apache IoTDB's Python client currently calls
    ``warnings.simplefilter('always', DeprecationWarning)`` while importing its
    Session module.  ``warnings.catch_warnings`` gives that import/call a copy of
    the process-wide filter list and restores the original list afterwards, so
    later backends (notably TileDB) see the same warning policy they would see
    in a standalone run.
    """

    with warnings.catch_warnings():
        yield
