"""Tests for BaseSplitter interface and SplitterFactory routing (B3)."""
import sys
import os
import pytest
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory, register_splitter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSplitter(BaseSplitter):
    def __init__(self, settings):
        self.settings = settings

    def split_text(self, text: str, trace=None) -> list[str]:
        if not text:
            return []
        size = getattr(self.settings, "chunk_size", 100)
        return [text[i:i + size] for i in range(0, len(text), size)]


class SemanticFakeSplitter(BaseSplitter):
    def __init__(self, settings):
        self.settings = settings

    def split_text(self, text: str, trace=None) -> list[str]:
        return text.split(". ") if text else []


@dataclass
class FakeIngestionSettings:
    splitter: str
    chunk_size: int = 50
    chunk_overlap: int = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_fakes():
    register_splitter("fake", FakeSplitter)
    register_splitter("semantic", SemanticFakeSplitter)
    yield


# ---------------------------------------------------------------------------
# BaseSplitter interface tests
# ---------------------------------------------------------------------------

def test_base_splitter_is_abstract():
    with pytest.raises(TypeError):
        BaseSplitter()  # type: ignore


# ---------------------------------------------------------------------------
# SplitterFactory routing tests
# ---------------------------------------------------------------------------

def test_factory_creates_fake_splitter():
    s = FakeIngestionSettings(splitter="fake")
    splitter = SplitterFactory.create(s)
    assert isinstance(splitter, FakeSplitter)


def test_factory_routes_by_type():
    s = FakeIngestionSettings(splitter="semantic")
    splitter = SplitterFactory.create(s)
    assert isinstance(splitter, SemanticFakeSplitter)


def test_factory_type_case_insensitive():
    s = FakeIngestionSettings(splitter="FAKE")
    splitter = SplitterFactory.create(s)
    assert isinstance(splitter, FakeSplitter)


def test_factory_unknown_type_raises():
    s = FakeIngestionSettings(splitter="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        SplitterFactory.create(s)


def test_factory_error_lists_known_types():
    s = FakeIngestionSettings(splitter="nonexistent")
    with pytest.raises(ValueError, match="fake"):
        SplitterFactory.create(s)


def test_factory_empty_splitter_raises():
    s = FakeIngestionSettings(splitter="")
    with pytest.raises(ValueError, match="empty"):
        SplitterFactory.create(s)


# ---------------------------------------------------------------------------
# FakeSplitter functional tests
# ---------------------------------------------------------------------------

def test_split_text_empty_returns_empty_list():
    s = FakeIngestionSettings(splitter="fake", chunk_size=50)
    splitter = SplitterFactory.create(s)
    assert splitter.split_text("") == []


def test_split_text_short_returns_single_chunk():
    s = FakeIngestionSettings(splitter="fake", chunk_size=100)
    splitter = SplitterFactory.create(s)
    result = splitter.split_text("hello world")
    assert result == ["hello world"]


def test_split_text_respects_chunk_size():
    s = FakeIngestionSettings(splitter="fake", chunk_size=5)
    splitter = SplitterFactory.create(s)
    result = splitter.split_text("abcdefghij")
    assert result == ["abcde", "fghij"]


def test_split_text_accepts_trace_param():
    s = FakeIngestionSettings(splitter="fake", chunk_size=100)
    splitter = SplitterFactory.create(s)
    result = splitter.split_text("hello", trace=None)
    assert len(result) == 1


def test_split_text_different_strategies():
    fake = SplitterFactory.create(FakeIngestionSettings(splitter="fake", chunk_size=3))
    semantic = SplitterFactory.create(FakeIngestionSettings(splitter="semantic"))
    text = "abc. def. ghi"
    fake_chunks = fake.split_text(text)
    semantic_chunks = semantic.split_text(text)
    assert fake_chunks != semantic_chunks
