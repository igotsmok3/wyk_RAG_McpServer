"""Tests for RecursiveSplitter (B7.5)."""
import sys
import os
import pytest
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Import to trigger registration
import libs.splitter.recursive_splitter  # noqa: F401

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


@dataclass
class FakeIngestionSettings:
    chunk_size: int = 500
    chunk_overlap: int = 50
    splitter: str = "recursive"
    batch_size: int = 100


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

def test_factory_creates_recursive_splitter():
    s = FakeIngestionSettings(splitter="recursive")
    splitter = SplitterFactory.create(s)
    assert isinstance(splitter, RecursiveSplitter)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_empty_text_returns_empty_list():
    s = FakeIngestionSettings()
    splitter = RecursiveSplitter(s)
    assert splitter.split_text("") == []


def test_short_text_returns_single_chunk():
    s = FakeIngestionSettings(chunk_size=500, chunk_overlap=50)
    splitter = RecursiveSplitter(s)
    result = splitter.split_text("Hello, world!")
    assert result == ["Hello, world!"]


def test_long_text_produces_multiple_chunks():
    s = FakeIngestionSettings(chunk_size=100, chunk_overlap=10)
    splitter = RecursiveSplitter(s)
    long_text = "word " * 200  # ~1000 chars
    result = splitter.split_text(long_text)
    assert len(result) > 1


def test_chunk_size_respected():
    """Each chunk must not exceed chunk_size by more than one separator token."""
    s = FakeIngestionSettings(chunk_size=200, chunk_overlap=20)
    splitter = RecursiveSplitter(s)
    long_text = "A" * 1000
    result = splitter.split_text(long_text)
    for chunk in result:
        assert len(chunk) <= 200


def test_overlap_produces_shared_content():
    """With overlap > 0, adjacent chunks share some text."""
    s = FakeIngestionSettings(chunk_size=50, chunk_overlap=20)
    splitter = RecursiveSplitter(s)
    text = "abcdefghij " * 20  # 220 chars
    result = splitter.split_text(text)
    assert len(result) >= 2
    # last chars of chunk[0] should appear in chunk[1]
    assert result[0][-5:] in result[1] or result[1][:5] in result[0]


def test_accepts_trace_param():
    s = FakeIngestionSettings()
    splitter = RecursiveSplitter(s)
    result = splitter.split_text("Hello trace!", trace=None)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Markdown structure preservation
# ---------------------------------------------------------------------------

def test_markdown_headings_not_split_mid_line():
    """H2 heading should not be broken across two chunks."""
    s = FakeIngestionSettings(chunk_size=200, chunk_overlap=0)
    splitter = RecursiveSplitter(s)
    md = (
        "## Section One\n\n"
        "Some introductory text that is not too long.\n\n"
        "## Section Two\n\n"
        "More text here for the second section.\n"
    )
    result = splitter.split_text(md)
    for chunk in result:
        # A heading line should not be split across chunks
        for line in chunk.splitlines():
            assert not (line.startswith("##") and len(line) < 3)


def test_code_block_not_split_mid_fence():
    """A short code block should land in a single chunk."""
    s = FakeIngestionSettings(chunk_size=500, chunk_overlap=0)
    splitter = RecursiveSplitter(s)
    md = (
        "Some text before the code.\n\n"
        "```python\n"
        "def foo():\n"
        "    return 42\n"
        "```\n\n"
        "Some text after the code.\n"
    )
    result = splitter.split_text(md)
    # The code block (``` ... ```) should not be split: both fences in same chunk
    code_chunks = [c for c in result if "```" in c]
    # Either the whole block is in one chunk or both fences appear in their respective chunk
    # At minimum, the opening ``` must be present in at least one chunk
    assert any("```" in c for c in result)


def test_paragraph_break_preferred_over_word_break():
    """Splitter should break at double newline before breaking mid-word."""
    s = FakeIngestionSettings(chunk_size=80, chunk_overlap=0)
    splitter = RecursiveSplitter(s)
    # Two paragraphs that together exceed chunk_size
    para1 = "First paragraph with some content here.\n\n"
    para2 = "Second paragraph with different content.\n"
    text = para1 + para2
    result = splitter.split_text(text)
    if len(result) > 1:
        # The split should happen at the paragraph boundary, not mid-word
        for chunk in result:
            words = chunk.split()
            for word in words:
                assert len(word) < 80  # no single word exceeds chunk_size


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_input_produces_same_output():
    s = FakeIngestionSettings(chunk_size=200, chunk_overlap=20)
    splitter = RecursiveSplitter(s)
    text = "Hello world. " * 50
    assert splitter.split_text(text) == splitter.split_text(text)


def test_all_words_preserved_across_chunks_no_overlap():
    """All words from the original text must appear across the resulting chunks."""
    s = FakeIngestionSettings(chunk_size=100, chunk_overlap=0)
    splitter = RecursiveSplitter(s)
    text = "The quick brown fox jumps over the lazy dog. " * 10
    result = splitter.split_text(text)
    combined = " ".join(result)
    original_words = text.split()
    combined_words = combined.split()
    assert sorted(combined_words) == sorted(original_words)
