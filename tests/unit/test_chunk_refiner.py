"""Unit tests for ChunkRefiner (27 tests, all using mocks)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.settings import (
    ChunkRefinerSettings,
    IngestionSettings,
    LLMSettings,
    Settings,
    VectorStoreSettings,
    EmbeddingSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner, _rule_based_refine
from libs.llm.base_llm import ChatResponse

FIXTURES = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"
_NOISY_CASES = json.loads(FIXTURES.read_text(encoding="utf-8"))
_CASE_MAP = {c["id"]: c for c in _NOISY_CASES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(use_llm: bool = False) -> Settings:
    llm = LLMSettings(provider="openai", model="gpt-4o-mini", api_key="fake-key")
    embedding = EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="fake-key")
    vs = VectorStoreSettings(provider="milvus")
    cr = ChunkRefinerSettings(use_llm=use_llm)
    ingestion = IngestionSettings(chunk_refiner=cr)
    return Settings(llm=llm, embedding=embedding, vector_store=vs, ingestion=ingestion)


def _make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={})


def _mock_llm(return_text: str = "LLM cleaned text"):
    llm = MagicMock()
    llm.chat.return_value = ChatResponse(content=return_text)
    return llm


# ===========================================================================
# Group 1: rule_based_refine standalone (8 tests — one per fixture scenario)
# ===========================================================================

class TestRuleBasedRefine:

    def _case(self, case_id: str):
        return _CASE_MAP[case_id]

    def test_typical_noise_scenario(self):
        c = self._case("typical_noise_scenario")
        result = _rule_based_refine(c["input"])
        assert "Chapter 3: Introduction" in result
        assert "valuable paragraph" in result
        assert "Page 15" not in result
        assert "Footer" not in result

    def test_ocr_errors_preserved(self):
        c = self._case("ocr_errors")
        result = _rule_based_refine(c["input"])
        assert result == c["expected_clean"]

    def test_page_header_footer(self):
        c = self._case("page_header_footer")
        result = _rule_based_refine(c["input"])
        assert "actual content starts here" in result
        assert "Page 42" not in result
        assert "Footer" not in result

    def test_excessive_whitespace(self):
        c = self._case("excessive_whitespace")
        result = _rule_based_refine(c["input"])
        # each line should not have double spaces (outside code blocks)
        for line in result.split("\n"):
            assert "  " not in line, f"double space in line: {line!r}"
        # max 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_format_markers(self):
        c = self._case("format_markers")
        result = _rule_based_refine(c["input"])
        assert "<div" not in result
        assert "<!--" not in result
        assert "**Bold text**" in result
        assert "*italic text*" in result

    def test_clean_text_not_over_cleaned(self):
        c = self._case("clean_text")
        result = _rule_based_refine(c["input"])
        assert result == c["expected_clean"]

    def test_code_blocks_preserved(self):
        c = self._case("code_blocks")
        result = _rule_based_refine(c["input"])
        assert "def    hello_world():" in result
        assert 'print("Hello,    World!")' in result
        assert "return    True" in result

    def test_mixed_noise(self):
        c = self._case("mixed_noise")
        result = _rule_based_refine(c["input"])
        assert "## Real Heading" in result
        assert "Page 123" not in result
        assert "Footer Text" not in result
        assert "<!--" not in result
        assert "\n\n\n" not in result


# ===========================================================================
# Group 2: ChunkRefiner without LLM (rule-only mode, 7 tests)
# ===========================================================================

class TestChunkRefinerRuleOnly:

    def _refiner(self):
        return ChunkRefiner(_settings(use_llm=False))

    def test_returns_same_number_of_chunks(self):
        chunks = [_make_chunk("Hello   World", f"c{i}") for i in range(3)]
        result = self._refiner().transform(chunks)
        assert len(result) == 3

    def test_metadata_refined_by_rule(self):
        chunk = _make_chunk("Page 1\n\nContent here")
        result = self._refiner().transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"

    def test_original_chunk_not_mutated(self):
        original_text = "Original   text\n\nPage 1"
        chunk = _make_chunk(original_text)
        self._refiner().transform([chunk])
        assert chunk.text == original_text

    def test_existing_metadata_preserved(self):
        chunk = _make_chunk("some text")
        chunk.metadata = {"source_path": "/foo/bar.pdf", "chunk_index": 2}
        result = self._refiner().transform([chunk])
        assert result[0].metadata["source_path"] == "/foo/bar.pdf"
        assert result[0].metadata["chunk_index"] == 2

    def test_single_failing_chunk_does_not_abort_batch(self):
        chunks = [_make_chunk("good text", "c1"), _make_chunk("good text", "c2")]
        refiner = self._refiner()
        with patch.object(refiner, "_rule_based_refine", side_effect=[Exception("boom"), "good text"]):
            result = refiner.transform(chunks)
        assert len(result) == 2
        assert result[0].metadata.get("refine_error") == "boom"
        assert result[1].text == "good text"

    def test_empty_chunk_list(self):
        assert self._refiner().transform([]) == []

    def test_trace_recorded(self):
        chunks = [_make_chunk("some   noise")]
        trace = TraceContext()
        self._refiner().transform(chunks, trace=trace)
        assert any(s.name == "chunk_refiner" for s in trace.stages)


# ===========================================================================
# Group 3: ChunkRefiner with LLM mock (7 tests)
# ===========================================================================

class TestChunkRefinerWithLLM:

    def _refiner_with_mock(self, llm_text="Cleaned by LLM"):
        mock_llm = _mock_llm(llm_text)
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        return refiner, mock_llm

    def test_llm_called_when_enabled(self):
        refiner, mock_llm = self._refiner_with_mock()
        refiner.transform([_make_chunk("noisy text")])
        mock_llm.chat.assert_called_once()

    def test_metadata_refined_by_llm(self):
        refiner, _ = self._refiner_with_mock()
        result = refiner.transform([_make_chunk("noisy text")])
        assert result[0].metadata["refined_by"] == "llm"

    def test_llm_output_used_as_final_text(self):
        refiner, _ = self._refiner_with_mock("Super clean output")
        result = refiner.transform([_make_chunk("noisy text")])
        assert result[0].text == "Super clean output"

    def test_llm_failure_falls_back_to_rules(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API error")
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        result = refiner.transform([_make_chunk("Page 1\n\nGood content here")])
        assert result[0].metadata["refined_by"] == "rule"
        assert "refine_fallback_reason" in result[0].metadata

    def test_llm_empty_response_falls_back_to_rules(self):
        mock_llm = _mock_llm("")
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        result = refiner.transform([_make_chunk("some text")])
        assert result[0].metadata["refined_by"] == "rule"

    def test_llm_prompt_contains_text(self):
        mock_llm = _mock_llm("result")
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        input_text = "My special text input"
        refiner.transform([_make_chunk(input_text)])
        call_args = mock_llm.chat.call_args[0][0]
        assert any(input_text in msg["content"] for msg in call_args)

    def test_use_llm_false_skips_llm(self):
        mock_llm = _mock_llm()
        refiner = ChunkRefiner(_settings(use_llm=False), llm=mock_llm)
        refiner.transform([_make_chunk("text")])
        mock_llm.chat.assert_not_called()


# ===========================================================================
# Group 4: config switch and edge cases (5 tests)
# ===========================================================================

class TestChunkRefinerConfig:

    def test_custom_prompt_path_loaded(self, tmp_path):
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text("Custom: {text}", encoding="utf-8")
        mock_llm = _mock_llm("result")
        refiner = ChunkRefiner(
            _settings(use_llm=True), llm=mock_llm, prompt_path=str(prompt_file)
        )
        refiner.transform([_make_chunk("hello")])
        call_content = mock_llm.chat.call_args[0][0][0]["content"]
        assert call_content == "Custom: hello"

    def test_missing_prompt_uses_default_fallback(self, tmp_path):
        refiner = ChunkRefiner(
            _settings(use_llm=False), prompt_path=str(tmp_path / "nonexistent.txt")
        )
        result = refiner.transform([_make_chunk("hello")])
        assert result[0].text == "hello"

    def test_whitespace_only_chunk_not_sent_to_llm(self):
        mock_llm = _mock_llm("something")
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        refiner.transform([_make_chunk("   \n\n   ")])
        mock_llm.chat.assert_not_called()

    def test_multiple_chunks_all_processed(self):
        mock_llm = _mock_llm("cleaned")
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        chunks = [_make_chunk(f"text {i}", f"c{i}") for i in range(5)]
        result = refiner.transform(chunks)
        assert len(result) == 5
        assert mock_llm.chat.call_count == 5

    def test_trace_counts_llm_and_rule_refined(self):
        mock_llm = MagicMock()
        # First call succeeds, second fails → fallback to rule
        mock_llm.chat.side_effect = [
            ChatResponse(content="cleaned"),
            RuntimeError("fail"),
        ]
        refiner = ChunkRefiner(_settings(use_llm=True), llm=mock_llm)
        chunks = [_make_chunk("text1", "c1"), _make_chunk("text2", "c2")]
        trace = TraceContext()
        refiner.transform(chunks, trace=trace)
        stage = next(s for s in trace.stages if s.name == "chunk_refiner")
        assert stage.data["llm_refined"] == 1
        assert stage.data["rule_refined"] == 1
