"""ImageStorage: persist image files and maintain an image_id→path index in MySQL.

Saves images to data/images/{collection}/ and records the mapping in a
MySQL table (image_index) using pymysql.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import pymysql
import pymysql.cursors


# ---------------------------------------------------------------------------
# Default connection config — override via constructor
# ---------------------------------------------------------------------------
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 3306
_DEFAULT_USER = "root"
_DEFAULT_PASSWORD = "011304"
_DEFAULT_DB = "WYK_RAG"

_DDL = """
CREATE TABLE IF NOT EXISTS image_index (
    image_id   VARCHAR(255) PRIMARY KEY,
    file_path  TEXT         NOT NULL,
    collection VARCHAR(255),
    doc_hash   VARCHAR(255),
    page_num   INT,
    created_at DATETIME     NOT NULL,
    INDEX idx_collection (collection),
    INDEX idx_doc_hash   (doc_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


class ImageStorage:
    """Store image files and maintain an image_id→path index in MySQL.

    Typical usage::

        storage = ImageStorage()
        path = storage.save(image_id="doc1_0_001", src_path="/tmp/img.png",
                            collection="tech_docs", doc_hash="abc123", page_num=1)
        entry = storage.get(image_id="doc1_0_001")
        entries = storage.list_by_collection("tech_docs")
    """

    def __init__(
        self,
        images_root: str = "data/images",
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        user: str = _DEFAULT_USER,
        password: str = _DEFAULT_PASSWORD,
        database: str = _DEFAULT_DB,
    ) -> None:
        self._images_root = images_root
        self._conn_kwargs: Dict[str, Any] = dict(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )
        os.makedirs(images_root, exist_ok=True)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        image_id: str,
        src_path: str,
        collection: str = "default",
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        trace: Any = None,
    ) -> str:
        """Copy an image file into managed storage and record the mapping.

        Idempotent: if image_id already exists the existing file_path is returned.

        Returns:
            Destination file path where the image was stored.
        """
        existing = self.get(image_id)
        if existing is not None:
            return existing["file_path"]

        dest_dir = os.path.join(self._images_root, collection)
        os.makedirs(dest_dir, exist_ok=True)

        ext = os.path.splitext(src_path)[1] or ".png"
        dest_path = os.path.join(dest_dir, f"{image_id}{ext}")
        shutil.copy2(src_path, dest_path)

        self._insert(image_id, dest_path, collection, doc_hash, page_num)
        return dest_path

    def save_bytes(
        self,
        image_id: str,
        data: bytes,
        ext: str = ".png",
        collection: str = "default",
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        trace: Any = None,
    ) -> str:
        """Write raw image bytes into managed storage and record the mapping.

        Idempotent: if image_id already exists, returns the existing path.

        Returns:
            Destination file path.
        """
        existing = self.get(image_id)
        if existing is not None:
            return existing["file_path"]

        dest_dir = os.path.join(self._images_root, collection)
        os.makedirs(dest_dir, exist_ok=True)

        if not ext.startswith("."):
            ext = f".{ext}"
        dest_path = os.path.join(dest_dir, f"{image_id}{ext}")
        with open(dest_path, "wb") as f:
            f.write(data)

        self._insert(image_id, dest_path, collection, doc_hash, page_num)
        return dest_path

    def get(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Return the index record for image_id, or None if not found."""
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT image_id, file_path, collection, doc_hash, page_num, created_at "
                "FROM image_index WHERE image_id = %s",
                (image_id,),
            )
            return cur.fetchone()

    def list_by_collection(self, collection: str) -> List[Dict[str, Any]]:
        """Return all index records for the given collection."""
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT image_id, file_path, collection, doc_hash, page_num, created_at "
                "FROM image_index WHERE collection = %s ORDER BY created_at",
                (collection,),
            )
            return list(cur.fetchall())

    def list_by_doc_hash(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Return all index records for the given document hash."""
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT image_id, file_path, collection, doc_hash, page_num, created_at "
                "FROM image_index WHERE doc_hash = %s ORDER BY page_num",
                (doc_hash,),
            )
            return list(cur.fetchall())

    def delete(self, image_id: str, remove_file: bool = True) -> bool:
        """Remove an image from the index and optionally delete the file.

        Returns True if the record existed and was removed.
        """
        entry = self.get(image_id)
        if entry is None:
            return False
        with self._cursor() as (conn, cur):
            cur.execute("DELETE FROM image_index WHERE image_id = %s", (image_id,))
            conn.commit()
        if remove_file and os.path.exists(entry["file_path"]):
            os.remove(entry["file_path"])
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> pymysql.connections.Connection:
        return pymysql.connect(**self._conn_kwargs)

    @contextmanager
    def _cursor(self) -> Generator:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                yield conn, cur
        finally:
            conn.close()

    def _ensure_table(self) -> None:
        with self._cursor() as (conn, cur):
            cur.execute(_DDL)
            conn.commit()

    def _insert(
        self,
        image_id: str,
        file_path: str,
        collection: Optional[str],
        doc_hash: Optional[str],
        page_num: Optional[int],
    ) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._cursor() as (conn, cur):
            cur.execute(
                """INSERT INTO image_index
                   (image_id, file_path, collection, doc_hash, page_num, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       file_path  = VALUES(file_path),
                       collection = VALUES(collection),
                       doc_hash   = VALUES(doc_hash),
                       page_num   = VALUES(page_num),
                       created_at = VALUES(created_at)
                """,
                (image_id, file_path, collection, doc_hash, page_num, now),
            )
            conn.commit()
