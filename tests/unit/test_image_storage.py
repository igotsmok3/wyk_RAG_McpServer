"""Tests for ImageStorage (C13) — MySQL backend.

Uses a real MySQL connection (127.0.0.1:3306, WYK_RAG database).
Each test cleans up its own image_ids to stay idempotent.
Image files are written to a tmp_path to avoid polluting the repo.
"""
from __future__ import annotations

import os
import shutil

import pymysql
import pytest

from ingestion.storage.image_storage import ImageStorage

# ---------------------------------------------------------------------------
# MySQL connection params (matches project default)
# ---------------------------------------------------------------------------
_DB_KWARGS = dict(host="127.0.0.1", port=3306, user="root", password="011304", database="WYK_RAG")

FIXTURE_IMAGE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "sample_pictures", "image.png"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_ids(*ids: str) -> None:
    """Delete test rows from MySQL to keep tests idempotent."""
    conn = pymysql.connect(**_DB_KWARGS)
    with conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(ids))
        cur.execute(f"DELETE FROM image_index WHERE image_id IN ({placeholders})", ids)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def storage(tmp_path):
    return ImageStorage(images_root=str(tmp_path / "images"))


@pytest.fixture()
def real_image(tmp_path):
    dest = str(tmp_path / "test_image.png")
    shutil.copy2(FIXTURE_IMAGE, dest)
    return dest


# ---------------------------------------------------------------------------
# save / get
# ---------------------------------------------------------------------------

def test_save_file_exists_on_disk(storage, real_image):
    try:
        path = storage.save("t_img_001", real_image, collection="col_a", doc_hash="abc", page_num=1)
        assert os.path.isfile(path)
    finally:
        _clean_ids("t_img_001")


def test_save_returns_correct_path(storage, real_image):
    try:
        path = storage.save("t_img_002", real_image, collection="col_a")
        assert "col_a" in path
        assert "t_img_002" in path
    finally:
        _clean_ids("t_img_002")


def test_get_returns_correct_record(storage, real_image):
    try:
        storage.save("t_img_003", real_image, collection="col_b", doc_hash="def", page_num=2)
        entry = storage.get("t_img_003")
        assert entry is not None
        assert entry["image_id"] == "t_img_003"
        assert entry["collection"] == "col_b"
        assert entry["doc_hash"] == "def"
        assert entry["page_num"] == 2
        assert os.path.isfile(entry["file_path"])
    finally:
        _clean_ids("t_img_003")


def test_get_nonexistent_returns_none(storage):
    assert storage.get("t_nonexistent_xyz") is None


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_save_idempotent(storage, real_image):
    try:
        path1 = storage.save("t_img_idem1", real_image, collection="col_x")
        path2 = storage.save("t_img_idem1", real_image, collection="col_x")
        assert path1 == path2
    finally:
        _clean_ids("t_img_idem1")


def test_save_idempotent_does_not_duplicate_record(storage, real_image):
    try:
        storage.save("t_img_idem2", real_image, collection="col_x2")
        storage.save("t_img_idem2", real_image, collection="col_x2")
        entries = storage.list_by_collection("col_x2")
        ids = [e["image_id"] for e in entries if e["image_id"] == "t_img_idem2"]
        assert len(ids) == 1
    finally:
        _clean_ids("t_img_idem2")


# ---------------------------------------------------------------------------
# save_bytes
# ---------------------------------------------------------------------------

def test_save_bytes_stores_correct_content(storage):
    data = b"\x89PNG\r\nfake_png_bytes"
    try:
        path = storage.save_bytes("t_img_bytes1", data, ext=".png", collection="col_bytes")
        assert os.path.isfile(path)
        with open(path, "rb") as f:
            assert f.read() == data
    finally:
        _clean_ids("t_img_bytes1")


def test_save_bytes_idempotent(storage):
    data = b"some_image_data"
    try:
        path1 = storage.save_bytes("t_img_bytes2", data, ext=".png", collection="col_b2")
        path2 = storage.save_bytes("t_img_bytes2", data, ext=".png", collection="col_b2")
        assert path1 == path2
        entries = [e for e in storage.list_by_collection("col_b2") if e["image_id"] == "t_img_bytes2"]
        assert len(entries) == 1
    finally:
        _clean_ids("t_img_bytes2")


# ---------------------------------------------------------------------------
# list_by_collection / list_by_doc_hash
# ---------------------------------------------------------------------------

def test_list_by_collection(storage, real_image, tmp_path):
    img2 = str(tmp_path / "img2.png")
    shutil.copy2(real_image, img2)
    try:
        storage.save("t_img_lc1", real_image, collection="t_docs", doc_hash="h1")
        storage.save("t_img_lc2", img2, collection="t_docs", doc_hash="h2")
        storage.save("t_img_lc3", real_image, collection="t_other")
        entries = storage.list_by_collection("t_docs")
        ids = {e["image_id"] for e in entries if e["image_id"].startswith("t_img_lc")}
        assert ids == {"t_img_lc1", "t_img_lc2"}
    finally:
        _clean_ids("t_img_lc1", "t_img_lc2", "t_img_lc3")


def test_list_by_collection_empty(storage):
    assert storage.list_by_collection("t_nonexistent_collection_xyz") == []


def test_list_by_doc_hash(storage, real_image, tmp_path):
    img2 = str(tmp_path / "img2.png")
    shutil.copy2(real_image, img2)
    try:
        storage.save("t_img_ldh1", real_image, collection="t_docs2", doc_hash="t_hash_X", page_num=1)
        storage.save("t_img_ldh2", img2, collection="t_docs2", doc_hash="t_hash_X", page_num=2)
        storage.save("t_img_ldh3", real_image, collection="t_docs2", doc_hash="t_hash_Y")
        entries = storage.list_by_doc_hash("t_hash_X")
        ids = [e["image_id"] for e in entries]
        assert "t_img_ldh1" in ids
        assert "t_img_ldh2" in ids
        assert "t_img_ldh3" not in ids
        page_nums = [e["page_num"] for e in entries if e["image_id"] in ("t_img_ldh1", "t_img_ldh2")]
        assert page_nums == [1, 2]
    finally:
        _clean_ids("t_img_ldh1", "t_img_ldh2", "t_img_ldh3")


# ---------------------------------------------------------------------------
# Persistence (different Storage instances share same MySQL)
# ---------------------------------------------------------------------------

def test_mapping_persists_across_instances(storage, real_image, tmp_path):
    try:
        path = storage.save("t_img_persist", real_image, collection="col_p")
        storage2 = ImageStorage(images_root=str(tmp_path / "images2"))
        entry = storage2.get("t_img_persist")
        assert entry is not None
        assert entry["file_path"] == path
    finally:
        _clean_ids("t_img_persist")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_removes_record_and_file(storage, real_image):
    try:
        path = storage.save("t_img_del1", real_image, collection="col_del")
        removed = storage.delete("t_img_del1", remove_file=True)
        assert removed is True
        assert storage.get("t_img_del1") is None
        assert not os.path.exists(path)
    finally:
        _clean_ids("t_img_del1")


def test_delete_nonexistent_returns_false(storage):
    assert storage.delete("t_no_such_id_xyz") is False


def test_delete_keep_file(storage, real_image):
    try:
        path = storage.save("t_img_keep1", real_image, collection="col_keep")
        storage.delete("t_img_keep1", remove_file=False)
        assert storage.get("t_img_keep1") is None
        assert os.path.exists(path)
    finally:
        _clean_ids("t_img_keep1")


# ---------------------------------------------------------------------------
# Real fixture image roundtrip
# ---------------------------------------------------------------------------

def test_real_image_roundtrip(storage):
    """End-to-end: save the real fixture PNG, verify file and MySQL record."""
    assert os.path.isfile(FIXTURE_IMAGE), "fixture image must exist"
    try:
        path = storage.save(
            image_id="t_fixture_img_001",
            src_path=FIXTURE_IMAGE,
            collection="t_test_collection",
            doc_hash="t_fixture_hash",
            page_num=1,
        )
        assert os.path.isfile(path)

        entry = storage.get("t_fixture_img_001")
        assert entry is not None
        assert entry["collection"] == "t_test_collection"
        assert entry["doc_hash"] == "t_fixture_hash"
        assert entry["page_num"] == 1

        entries = [
            e for e in storage.list_by_collection("t_test_collection")
            if e["image_id"] == "t_fixture_img_001"
        ]
        assert len(entries) == 1
    finally:
        _clean_ids("t_fixture_img_001")
