"""Small SQLite state store used by the API adapter.

SQLite contains only orchestration metadata.  Scientific data remains in the
existing CFD-Bench backends and benchmark results remain canonical CSV files.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    format TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS upload_files (
                    file_id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL REFERENCES uploads(upload_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    offset_bytes INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(upload_id, name)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset TEXT,
                    upload_id TEXT,
                    request_json TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    result_csv TEXT,
                    result_h5 TEXT,
                    stdout_path TEXT NOT NULL,
                    stderr_path TEXT NOT NULL,
                    pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_upload
                    ON jobs(upload_id);
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "result_h5" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN result_h5 TEXT")
            # A process that died cannot still own a child job from this API
            # instance.  Make that explicit instead of leaving stale "running"
            # records forever.
            now = utc_now()
            conn.execute(
                """
                UPDATE jobs
                SET status='failed', finished_at=?,
                    error=COALESCE(error, 'API process restarted while job was running')
                WHERE status='running'
                """,
                (now,),
            )

    def create_upload(self, upload_id: str, fmt: str, files: Iterable[Dict[str, Any]]) -> None:
        created = utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO uploads(upload_id, format, status, created_at) VALUES (?, ?, 'uploading', ?)",
                (upload_id, fmt, created),
            )
            conn.executemany(
                """
                INSERT INTO upload_files(file_id, upload_id, name, size_bytes, offset_bytes)
                VALUES (?, ?, ?, ?, 0)
                """,
                [
                    (item["file_id"], upload_id, item["name"], int(item["size_bytes"]))
                    for item in files
                ],
            )

    def get_upload(self, upload_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM uploads WHERE upload_id=?", (upload_id,)
            ).fetchone()
            if row is None:
                return None
            files = conn.execute(
                "SELECT * FROM upload_files WHERE upload_id=? ORDER BY rowid", (upload_id,)
            ).fetchall()
        out = dict(row)
        out["files"] = [dict(item) for item in files]
        return out

    def get_upload_file(self, upload_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM upload_files WHERE upload_id=? AND file_id=?",
                (upload_id, file_id),
            ).fetchone()
        return None if row is None else dict(row)

    def set_upload_offset(self, upload_id: str, file_id: str, expected: int, new_offset: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE upload_files SET offset_bytes=?
                WHERE upload_id=? AND file_id=? AND offset_bytes=?
                """,
                (int(new_offset), upload_id, file_id, int(expected)),
            )
            return cur.rowcount == 1

    def complete_upload(self, upload_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE uploads SET status='completed', completed_at=? WHERE upload_id=?",
                (utc_now(), upload_id),
            )

    def delete_upload(self, upload_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM uploads WHERE upload_id=?", (upload_id,))
            return cur.rowcount == 1

    def has_active_job_for_upload(self, upload_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM jobs
                WHERE upload_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (upload_id,),
            ).fetchone()
        return row is not None

    def create_job(
        self,
        *,
        job_id: str,
        job_type: str,
        dataset: Optional[str],
        upload_id: Optional[str],
        request: Dict[str, Any],
        command: List[str],
        result_csv: Optional[str],
        result_h5: Optional[str] = None,
        stdout_path: str,
        stderr_path: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, type, status, dataset, upload_id, request_json,
                    command_json, result_csv, result_h5, stdout_path, stderr_path, created_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    dataset,
                    upload_id,
                    json.dumps(request, separators=(",", ":"), ensure_ascii=False),
                    json.dumps(command, separators=(",", ":"), ensure_ascii=False),
                    result_csv,
                    result_h5,
                    stdout_path,
                    stderr_path,
                    utc_now(),
                ),
            )

    def next_queued_job(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
        return None if row is None else self._decode_job(dict(row))

    def claim_job(self, job_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET status='running', started_at=?
                WHERE job_id=? AND status='queued' AND cancel_requested=0
                """,
                (utc_now(), job_id),
            )
            return cur.rowcount == 1

    def set_job_pid(self, job_id: str, pid: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET pid=? WHERE job_id=?", (int(pid), job_id))

    def finish_job_if_running(
        self,
        job_id: str,
        *,
        status: str,
        exit_code: Optional[int],
        error: Optional[str] = None,
    ) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs SET status=?, exit_code=?, error=?, finished_at=?, pid=NULL
                WHERE job_id=? AND status='running'
                """,
                (status, exit_code, error, utc_now(), job_id),
            )
            return cur.rowcount == 1

    def cancel_unstarted_job(self, job_id: str) -> bool:
        """Converge a claimed job that has not registered a subprocess PID."""

        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status='cancelled', finished_at=?, pid=NULL,
                    error=COALESCE(error, 'job cancelled by user before subprocess start')
                WHERE job_id=? AND status='running' AND pid IS NULL
                  AND cancel_requested=1
                """,
                (utc_now(), job_id),
            )
            return cur.rowcount == 1

    def cancel_running_for_pid(self, job_id: str, pid: int, *, error: str) -> bool:
        """Converge a cancelled running job only if it still owns *pid*."""

        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status='cancelled', finished_at=?, pid=NULL, error=?
                WHERE job_id=? AND status='running' AND pid=?
                  AND cancel_requested=1
                """,
                (utc_now(), error, job_id, int(pid)),
            )
            return cur.rowcount == 1

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        exit_code: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET status=?, exit_code=?, error=?, finished_at=?, pid=NULL
                WHERE job_id=?
                """,
                (status, exit_code, error, utc_now(), job_id),
            )

    def request_cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            current = dict(row)
            if current["status"] == "queued":
                conn.execute(
                    """
                    UPDATE jobs SET status='cancelled', cancel_requested=1, finished_at=?
                    WHERE job_id=?
                    """,
                    (utc_now(), job_id),
                )
            elif current["status"] == "running":
                conn.execute(
                    "UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,)
                )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else self._decode_job(dict(row))

    def list_jobs(self, *, status: Optional[str] = None, job_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        clauses = []
        values: List[Any] = []
        if status:
            clauses.append("status=?")
            values.append(status)
        if job_type:
            clauses.append("type=?")
            values.append(job_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._decode_job(dict(row)) for row in rows]

    def registered_datasets(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT dataset FROM jobs
                WHERE type='ingest' AND status='succeeded' AND dataset IS NOT NULL
                ORDER BY dataset
                """
            ).fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _decode_job(row: Dict[str, Any]) -> Dict[str, Any]:
        row["request"] = json.loads(row.pop("request_json"))
        row["command"] = json.loads(row.pop("command_json"))
        row["cancel_requested"] = bool(row.get("cancel_requested"))
        return row
