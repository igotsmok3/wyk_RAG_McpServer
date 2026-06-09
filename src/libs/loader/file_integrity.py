import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity checking (5.4.1: first step of Ingestion Flow)."""

    @abstractmethod
    def compute_sha256(self, path: str) -> str: ...

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool: ...

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None: ...

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None: ...


class SqliteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed integrity checker.

    Uses WAL journal mode for concurrent-safe writes.
    Default db: data/db/ingestion_history.db (auto-created).
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS ingestion_history (
            file_hash   TEXT PRIMARY KEY,
            file_path   TEXT NOT NULL,
            status      TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            error_msg   TEXT
        )
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(self._DDL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def compute_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        return row is not None and row[0] == "success"

    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, status, processed_at, error_msg)
                   VALUES (?, ?, 'success', ?, NULL)""",
                (file_hash, file_path, now),
            )
            conn.commit()

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, status, processed_at, error_msg)
                   VALUES (?, '', 'failed', ?, ?)""",
                (file_hash, now, error_msg),
            )
            conn.commit()


# Spec alias — architecture doc refers to this as "MysqlIntegrityChecker"
MysqlIntegrityChecker = SqliteIntegrityChecker
