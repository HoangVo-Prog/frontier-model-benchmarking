"""Durable append-only JSONL recording with an inter-process file lock."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .schemas import safe_json


@contextmanager
def _lock(lockfile: Path) -> Iterator[None]:
    # Separate lock file avoids locking an append-only log inode during rotation.
    with lockfile.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JSONLLogger:
    def __init__(self, directory: str | Path = "logs") -> None:
        self.directory = Path(directory).expanduser()
        self.file = self.directory / "requests.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(safe_json(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        with _lock(self.directory / ".requests.lock"):
            with self.file.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
