"""Single-worker job scheduler for ingest and benchmark subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from .config import ApiConfig
from .state import StateStore


class ExecutionGate:
    """Coordinate benchmark-sensitive work without serialising HTTP itself.

    Heavy ingest/benchmark/coupling subprocesses are single-worker jobs. Interpolation
    is an interactive database read and upload PATCH calls are streaming disk
    writes. A benchmark waits for any already-active interpolation/upload
    chunk to finish, then prevents new ones from entering until the benchmark
    exits.

    Waiting is deliberately cancellable. A queued benchmark must not become a
    zombie ``running`` job merely because it is waiting for an upload or
    interpolation request to leave the gate.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._active_job_type: Optional[str] = None
        self._active_upload_chunks = 0
        self._active_interactive_reads = 0

    def enter_job(
        self,
        job_type: str,
        *,
        should_cancel: Optional[Callable[[], bool]] = None,
        wait_interval_sec: float = 0.1,
    ) -> bool:
        """Enter the heavy-job gate, returning ``False`` if cancelled while waiting."""

        kind = str(job_type)
        wait_interval = max(0.01, float(wait_interval_sec))
        with self._cond:
            while True:
                if should_cancel is not None and should_cancel():
                    return False

                if kind == "benchmark":
                    blocked = bool(self._active_upload_chunks or self._active_interactive_reads)
                else:
                    # Ingest may overlap unrelated uploads, but not interpolation,
                    # because both can stress the same data backend.
                    blocked = bool(self._active_interactive_reads)

                if not blocked:
                    self._active_job_type = kind
                    return True
                self._cond.wait(timeout=wait_interval)

    def wake_waiters(self) -> None:
        """Wake gate waiters so cancellation/shutdown is observed immediately."""

        with self._cond:
            self._cond.notify_all()

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
        return self.active_job_type in {"benchmark", "ingest", "coupling"}

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
            if self._active_job_type in {"benchmark", "ingest", "coupling"}:
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
        # Serialises the short transition between a claimed job and registering
        # its Popen handle. cancel() uses the same lock, closing the race where a
        # cancellation could otherwise arrive between the pre-start check and
        # process registration.
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
        self.gate.wake_waiters()
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
        result_h5: Optional[Path] = None,
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
            result_h5=str(result_h5) if result_h5 else None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        self.notify()
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> Optional[dict]:
        job = self.store.request_cancel(job_id)
        if job is None:
            return None

        # A job waiting at ExecutionGate is still queued after this patch, so a
        # normal queued cancellation is already terminal. Wake the gate anyway
        # so its waiter sees the state change without waiting for a timeout.
        self.gate.wake_waiters()

        if job["status"] != "running":
            self.notify()
            return self.store.get_job(job_id)

        with self._process_lock:
            process = self._processes.get(job_id)
            latest = self.store.get_job(job_id)
            if latest is None:
                return None

            if process is not None and process.poll() is None:
                self._terminate_process(process)
                threading.Thread(
                    target=self._kill_if_still_running,
                    args=(process,),
                    name=f"cfd-bench-api-cancel-{job_id}",
                    daemon=True,
                ).start()
            else:
                pid = latest.get("pid")
                if pid is None:
                    # Claimed but Popen has not been registered. Because the
                    # worker uses _process_lock for its pre-start check + Popen
                    # registration, seeing no process and no PID under this lock
                    # means it is safe to converge the job immediately.
                    self.store.cancel_unstarted_job(job_id)
                else:
                    self._cancel_by_persisted_pid(latest)

        self.notify()
        return self.store.get_job(job_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued_job()
            if job is None:
                self._wake.wait(self.config.scheduler_poll_sec)
                self._wake.clear()
                continue
            self._execute(job)

    def _execute(self, job: dict) -> None:
        job_id = str(job["job_id"])
        job_type = str(job["type"])

        # Crucially, do not claim the job before the gate. While it waits for an
        # upload chunk or interpolation request it remains `queued`, which means
        # POST /cancel can terminate it synchronously without creating the
        # `running + pid=NULL` zombie state observed in production.
        entered = self.gate.enter_job(
            job_type,
            should_cancel=lambda: self._stop.is_set() or self._job_no_longer_queued(job_id),
        )
        if not entered:
            return

        try:
            if not self.store.claim_job(job_id):
                return

            with open(job["stdout_path"], "ab", buffering=0) as stdout, open(
                job["stderr_path"], "ab", buffering=0
            ) as stderr:
                process: Optional[subprocess.Popen] = None
                with self._process_lock:
                    latest = self.store.get_job(job_id)
                    if latest is None:
                        return
                    if latest["status"] != "running" or latest.get("cancel_requested"):
                        self.store.cancel_unstarted_job(job_id)
                        return

                    try:
                        process = subprocess.Popen(
                            job["command"],
                            stdout=stdout,
                            stderr=stderr,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    except Exception as exc:
                        self.store.finish_job_if_running(
                            job_id,
                            status="failed",
                            exit_code=None,
                            error=f"failed to start cfd-bench subprocess: {exc}",
                        )
                        return

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
                self.store.finish_job_if_running(
                    job_id,
                    status=status,
                    exit_code=int(exit_code),
                    error=error,
                )
        finally:
            with self._process_lock:
                self._processes.pop(job_id, None)
            self.gate.leave_job()

    def _job_no_longer_queued(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        return job is None or job.get("status") != "queued" or bool(job.get("cancel_requested"))

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

    def _cancel_by_persisted_pid(self, job: dict) -> None:
        """Best-effort cancellation when the in-memory Popen handle is missing.

        The worker persists the session-leader PID immediately after Popen.
        Normally `_processes` is authoritative, but this fallback prevents a
        stale in-memory map from making a running job impossible to cancel.
        Before signalling, verify that /proc still looks like the command saved
        for this job to reduce the risk of signalling a reused PID.
        """

        job_id = str(job["job_id"])
        pid = int(job["pid"])
        command = [str(part) for part in job.get("command", [])]

        if not self._pid_matches_command(pid, command):
            if not self._pid_exists(pid):
                self.store.cancel_running_for_pid(
                    job_id,
                    pid,
                    error="job cancelled by user; persisted subprocess was already gone",
                )
            return

        if self._signal_pid_group(pid, signal.SIGTERM):
            threading.Thread(
                target=self._kill_persisted_pid_if_needed,
                args=(job_id, pid, command),
                name=f"cfd-bench-api-cancel-pid-{job_id}",
                daemon=True,
            ).start()

    def _kill_persisted_pid_if_needed(
        self, job_id: str, pid: int, command: Sequence[str]
    ) -> None:
        time.sleep(max(0.0, self.config.cancel_grace_sec))
        latest = self.store.get_job(job_id)
        if latest is None or latest.get("status") != "running":
            return
        if int(latest.get("pid") or -1) != int(pid) or not latest.get("cancel_requested"):
            return

        if self._pid_matches_command(pid, command):
            if not self._signal_pid_group(pid, signal.SIGKILL):
                return
        # Once TERM observed the process disappear, or SIGKILL was delivered to
        # the verified process group, converge the orchestration record. If a
        # worker still owns process.wait(), its conditional finish is idempotent.
        self.store.cancel_running_for_pid(
            job_id,
            pid,
            error="job cancelled by user",
        )

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if os.name != "posix":  # pragma: no cover - Docker runtime is Linux
            return True
        return Path(f"/proc/{int(pid)}").exists()

    @staticmethod
    def _pid_matches_command(pid: int, command: Sequence[str]) -> bool:
        if os.name != "posix":  # pragma: no cover - Docker runtime is Linux
            return True
        path = Path(f"/proc/{int(pid)}/cmdline")
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return False
        actual = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        expected = [str(part) for part in command[1:]]
        if not actual or not expected:
            return False

        # Console scripts commonly appear as `python /path/cfd-bench ...` in
        # /proc, so compare the CLI arguments as a contiguous subsequence rather
        # than requiring argv[0] to match exactly.
        width = len(expected)
        return any(actual[index : index + width] == expected for index in range(len(actual) - width + 1))

    @staticmethod
    def _signal_pid_group(pid: int, sig: int) -> bool:
        try:
            pgid = os.getpgid(int(pid))
            os.killpg(pgid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False
