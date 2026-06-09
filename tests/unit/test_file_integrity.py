import os

import pytest

from src.libs.loader.file_integrity import FileIntegrityChecker, MysqlIntegrityChecker, SqliteIntegrityChecker


# ── helpers ──────────────────────────────────────────────────────────────────

def make_checker(tmp_path):
    return MysqlIntegrityChecker(str(tmp_path / "test.db"))


# ── compute_sha256 ────────────────────────────────────────────────────────────

class TestComputeSha256:
    def test_returns_64_char_hex(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello world")
        h = make_checker(tmp_path).compute_sha256(str(f))
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_consistent_across_calls(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"consistent content")
        checker = make_checker(tmp_path)
        assert checker.compute_sha256(str(f)) == checker.compute_sha256(str(f))

    def test_different_content_yields_different_hash(self, tmp_path):
        f1, f2 = tmp_path / "x.txt", tmp_path / "y.txt"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        checker = make_checker(tmp_path)
        assert checker.compute_sha256(str(f1)) != checker.compute_sha256(str(f2))

    def test_large_file(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(os.urandom(5 * 1024 * 1024))
        h = make_checker(tmp_path).compute_sha256(str(f))
        assert len(h) == 64


# ── should_skip ───────────────────────────────────────────────────────────────

class TestShouldSkip:
    def test_unknown_hash_returns_false(self, tmp_path):
        assert make_checker(tmp_path).should_skip("deadbeef") is False

    def test_after_mark_success_returns_true(self, tmp_path):
        checker = make_checker(tmp_path)
        checker.mark_success("abc", "/path/file.pdf")
        assert checker.should_skip("abc") is True

    def test_after_mark_failed_returns_false(self, tmp_path):
        checker = make_checker(tmp_path)
        checker.mark_failed("abc", "parse error")
        assert checker.should_skip("abc") is False

    def test_failed_overwritten_by_success(self, tmp_path):
        checker = make_checker(tmp_path)
        checker.mark_failed("abc", "first attempt failed")
        checker.mark_success("abc", "/path/file.pdf")
        assert checker.should_skip("abc") is True


# ── mark_success ─────────────────────────────────────────────────────────────

class TestMarkSuccess:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "sub" / "nested.db"
        checker = MysqlIntegrityChecker(str(db))
        checker.mark_success("h1", "/f.pdf")
        assert db.exists()

    def test_idempotent(self, tmp_path):
        checker = make_checker(tmp_path)
        checker.mark_success("h1", "/f.pdf")
        checker.mark_success("h1", "/f.pdf")
        assert checker.should_skip("h1") is True


# ── mark_failed ───────────────────────────────────────────────────────────────

class TestMarkFailed:
    def test_does_not_skip(self, tmp_path):
        checker = make_checker(tmp_path)
        checker.mark_failed("h2", "connection timeout")
        assert checker.should_skip("h2") is False

    def test_idempotent(self, tmp_path):
        checker = make_checker(tmp_path)
        checker.mark_failed("h2", "err1")
        checker.mark_failed("h2", "err2")
        assert checker.should_skip("h2") is False


# ── default db path ────────────────────────────────────────────────────────────

class TestDefaultDbPath:
    def test_default_path_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        checker = MysqlIntegrityChecker()
        checker.mark_success("h1", "/file.pdf")
        assert (tmp_path / "data" / "db" / "ingestion_history.db").exists()


# ── abstract interface ─────────────────────────────────────────────────────────

class TestAbstractInterface:
    def test_mysql_checker_is_subclass(self):
        assert issubclass(MysqlIntegrityChecker, FileIntegrityChecker)

    def test_sqlite_checker_is_subclass(self):
        assert issubclass(SqliteIntegrityChecker, FileIntegrityChecker)

    def test_alias_is_same_class(self):
        assert MysqlIntegrityChecker is SqliteIntegrityChecker
