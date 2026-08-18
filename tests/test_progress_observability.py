from __future__ import annotations

from cfd_bench.core.observability import BenchmarkProgress, benchmark_progress


def test_progress_disabled_does_not_start_reporter_thread():
    with benchmark_progress("unit", 1.0, enabled=False) as progress:
        progress.set_phase("query")
        progress.transaction()
        assert progress._thread is None
    assert progress.transactions == 1
    assert progress.phase == "query"


def test_progress_heartbeat_is_observational_only(capsys):
    # A very short interval is clamped internally; invoke the reporter's state
    # methods directly and verify it does not expose cancellation/deadline APIs.
    progress = BenchmarkProgress("unit", 60.0, enabled=False)
    with progress:
        progress.set_phase("scalar query")
        progress.transaction(3)
    out = capsys.readouterr().out
    assert "benchmark loop start" in out
    assert "benchmark loop end" in out
    assert progress.transactions == 3
    assert not hasattr(progress, "cancel")
    assert not hasattr(progress, "deadline")
