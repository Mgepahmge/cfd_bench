"""Resumable, streaming upload implementation."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Dict

from fastapi import HTTPException, Request, status

from .config import ApiConfig
from .schemas import UploadCreateRequest
from .state import StateStore


class UploadService:
    def __init__(self, config: ApiConfig, store: StateStore):
        self.config = config
        self.store = store
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _file_lock(self, file_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(file_id, asyncio.Lock())

    def upload_dir(self, upload_id: str) -> Path:
        return self.config.uploads_root / upload_id

    def files_dir(self, upload_id: str) -> Path:
        return self.upload_dir(upload_id) / "files"

    def file_path(self, upload_id: str, name: str) -> Path:
        return self.files_dir(upload_id) / name

    def create(self, request: UploadCreateRequest) -> Dict[str, object]:
        upload_id = "upl_" + uuid.uuid4().hex
        files = [
            {
                "file_id": "file_" + uuid.uuid4().hex,
                "name": item.name,
                "size_bytes": int(item.size_bytes),
            }
            for item in request.files
        ]
        path = self.files_dir(upload_id)
        path.mkdir(parents=True, exist_ok=False)
        try:
            self.store.create_upload(upload_id, request.format, files)
        except Exception:
            shutil.rmtree(self.upload_dir(upload_id), ignore_errors=True)
            raise
        return self.get(upload_id)

    def get(self, upload_id: str) -> Dict[str, object]:
        upload = self.store.get_upload(upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="upload not found")
        return upload

    async def append_chunk(
        self,
        upload_id: str,
        file_id: str,
        request: Request,
        requested_offset: int,
    ) -> int:
        upload = self.get(upload_id)
        if upload["status"] != "uploading":
            raise HTTPException(status_code=409, detail="upload is already completed")
        item = self.store.get_upload_file(upload_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="upload file not found")

        lock = await self._file_lock(file_id)
        async with lock:
            item = self.store.get_upload_file(upload_id, file_id)
            assert item is not None
            current = int(item["offset_bytes"])
            expected_size = int(item["size_bytes"])
            if requested_offset != current:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={"message": "upload offset mismatch", "expected_offset": current},
                    headers={"Upload-Offset": str(current)},
                )
            if current >= expected_size:
                return current

            path = self.file_path(upload_id, str(item["name"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            actual_size = path.stat().st_size if path.exists() else 0
            if actual_size != current:
                raise HTTPException(
                    status_code=409,
                    detail="staging file size does not match persisted upload offset",
                )

            written = 0
            try:
                with path.open("ab") as fh:
                    async for chunk in request.stream():
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > self.config.max_chunk_size:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"chunk exceeds max size {self.config.max_chunk_size} bytes; "
                                    f"recommended chunk size is {self.config.recommended_chunk_size} bytes"
                                ),
                            )
                        if current + written > expected_size:
                            raise HTTPException(
                                status_code=413,
                                detail="chunk would exceed declared file size",
                            )
                        fh.write(chunk)
                    fh.flush()
            except Exception:
                # Keep the persisted offset and the staged file in sync when a
                # request is rejected midway through a streamed chunk.
                if path.exists():
                    with path.open("r+b") as fh:
                        fh.truncate(current)
                raise

            new_offset = current + written
            if not self.store.set_upload_offset(upload_id, file_id, current, new_offset):
                with path.open("r+b") as fh:
                    fh.truncate(current)
                latest = self.store.get_upload_file(upload_id, file_id)
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "concurrent upload offset update",
                        "expected_offset": int(latest["offset_bytes"]) if latest else current,
                    },
                )
            return new_offset

    def complete(self, upload_id: str) -> Dict[str, object]:
        upload = self.get(upload_id)
        if upload["status"] == "completed":
            return upload
        incomplete = [
            item
            for item in upload["files"]
            if int(item["offset_bytes"]) != int(item["size_bytes"])
        ]
        if incomplete:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "upload is incomplete",
                    "files": [
                        {
                            "file_id": item["file_id"],
                            "offset_bytes": item["offset_bytes"],
                            "size_bytes": item["size_bytes"],
                        }
                        for item in incomplete
                    ],
                },
            )
        self.store.complete_upload(upload_id)
        return self.get(upload_id)

    def delete(self, upload_id: str) -> None:
        self.get(upload_id)
        if self.store.has_active_job_for_upload(upload_id):
            raise HTTPException(
                status_code=409,
                detail="upload is referenced by a queued or running ingest job",
            )
        if not self.store.delete_upload(upload_id):
            raise HTTPException(status_code=404, detail="upload not found")
        shutil.rmtree(self.upload_dir(upload_id), ignore_errors=True)
