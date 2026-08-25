"""Single-worker job scheduler for ingest and benchmark subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from .config import ApiConfig
from .state import StateStore


class ExecutionGate:
    """Coordinate benchmark-sensitive work without serialising HTTP itself.

    Heavy ingest/benchmark subprocesses are single-worker jobs.  Interpolation
    is an interactive database read and upload PATCH calls are streaming disk
    writes.  A benchmark waits for any already-active interpolation/upload
    chunk to finish, then prevents new ones from entering until the benchmark
    exits.  This closes the check-then-start race that a plain status flag
    would leave open.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._active_job_type: Optional[str] = None
        self._active_upload_chunks = 0
        self._active_interactive_reads = 0

    def enter_job(self, job_type: str) -> None:
        kind = str(job_type)
        with self._cond:
            if kind == "benchmark":
                while self._active_upload_chunks or self._active_interactive_reads:
                    self._cond.wait()
            else:
                # Ingest may overlap unrelated uploads, but not interpolation,
                # because both can stress the same data backend.
                while self._active_interactive_reads:
                    self._cond.wait()
            self._active_job_type = kind

    def leave_job(self) -> None:
        with self._cond:
            self._active_job_type = None
            self._cond.notify_all()

    @property
    def active_job_type(self) -> Optional[str]:
        with self._cond:
            return self._active_job_type

    def benchmark_running(self) -> bool:
        return self.active_job_type == "benchmark"

    def heavy_job_running(self) -> bool:
        return self.active_job_type in {"benchmark", "ingest"}

    def try_begin_upload_chunk(self) -> bool:
        with self._cond:
            if self._active_job_type == "benchmark":
                return False
            self._active_upload_chunks += 1
            return True

    def end_upload_chunk(self) -> None:
        with self._cond:
            self._active_upload_chunks = max(0, self._active_upload_chunks - 1)
            self._cond.notify_all()

    def try_begin_interactive_read(self) -> bool:
        with self._cond:
            if self._active_job_type in {"benchmark", "ingest"}:
                return False
            self._active_interactive_reads += 1
            return True

    def end_interactive_read(self) -> None:
        with self._cond:
            self._active_interactive_reads = max(0, self._active_interactive_reads - 1)
            self._cond.notify_all()


class JobManager:
    def __init__(self, config: ApiConfig, store: StateStore, gate: ExecutionGate):
        self.config = config
        self.store = store
        self.gate = gate
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._processes: Dict[str, subprocess.Popen] = {}
        self._process_lock = threading.Lock()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, name="cfd-bench-api-jobs", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._process_lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                self._terminate_process(process)
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=max(2.0, self.config.cancel_grace_sec + 1.0))

    def notify(self) -> None:
        self._wake.set()

    def new_job_id(self) -> str:
        return "job_" + uuid.uuid4().hex

    def create_job(
        self,
        *,
        job_type: str,
        dataset: Optional[str],
        upload_id: Optional[str],
        request: dict,
        command: list[str],
        result_csv: Optional[Path] = None,
    ) -> dict:
        job_id = self.new_job_id()
        job_dir = self.config.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        # Create log files immediately so clients can request them while queued.
        stdout_path.touch()
        stderr_path.touch()
        self.store.create_job(
            job_id=job_id,
            job_type=job_type,
            dataset=dataset,
            upload_id=upload_id,
            request=request,
            command=command,
            result_csv=str(result_csv) if result_csv else None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        self.notify()
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> Optional[dict]:
        job = self.store.request_cancel(job_id)
        if job is None:
            return None
        if job["status"] == "running":
            with self._process_lock:
                process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                self._terminate_process(process)
                threading.Thread(
                    target=self._kill_if_still_running,
                    args=(process,),
                    name=f"cfd-bench-api-cancel-{job_id}",
                    daemon=True,
                ).start()
        self.notify()
        return self.store.get_job(job_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued_job()
            if job is None:
                self._wake.wait(self.config.scheduler_poll_sec)
                self._wake.clear()
                continue
            if not self.store.claim_job(job["job_id"]):
                continue
            running = self.store.get_job(job["job_id"])
            if running is None:
                continue
            self._execute(running)

    def _execute(self, job: dict) -> None:
        job_id = str(job["job_id"])
        job_type = str(job["type"])
        self.gate.enter_job(job_type)
        try:
            with open(job["stdout_path"], "ab", buffering=0) as stdout, open(
                job["stderr_path"], "ab", buffering=0
            ) as stderr:
                try:
                    process = subprocess.Popen(
                        job["command"],
                        stdout=stdout,
                        stderr=stderr,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception as exc:
                    self.store.finish_job(
                        job_id,
                        status="failed",
                        exit_code=None,
                        error=f"failed to start cfd-bench subprocess: {exc}",
                    )
                    return

                with self._process_lock:
                    self._processes[job_id] = process
                self.store.set_job_pid(job_id, process.pid)
                exit_code = process.wait()
                final = self.store.get_job(job_id)
                cancelled = bool(final and final.get("cancel_requested"))
                if cancelled:
                    status = "cancelled"
                    error = "job cancelled by user"
                elif exit_code == 0:
                    status = "succeeded"
                    error = None
                else:
                    status = "failed"
                    error = f"cfd-bench exited with code {exit_code}"
                self.store.finish_job(
                    job_id,
                    status=status,
                    exit_code=int(exit_code),
                    error=error,
                )
        finally:
            with self._process_lock:
                self._processes.pop(job_id, None)
            self.gate.leave_job()

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Docker runtime is Linux
                process.terminate()
        except ProcessLookupError:
            pass

    def _kill_if_still_running(self, process: subprocess.Popen) -> None:
        time.sleep(max(0.0, self.config.cancel_grace_sec))
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Docker runtime is Linux
                process.kill()
        except ProcessLookupError:
            pass
