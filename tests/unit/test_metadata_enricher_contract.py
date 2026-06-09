"""Contract tests for MetadataEnricher (C6)."""
from __future__ import annotations

import copy
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.settings import (
    ChunkRefinerSettings,
    IngestionSettings,
    LLMSettings,
    MetadataEnricherSettings,
    Settings,
    EmbeddingSettings,
    VectorStoreSettings,
)
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher, _rule_based_enrich, _parse_llm_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings(use_llm: bool = False) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai"),
        vector_store=VectorStoreSettings(provider="milvus"),
        ingestion=IngestionSettings(
            metadata_enricher=MetadataEnricherSettings(use_llm=use_llm)
        ),
    )


def _make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=f"doc_000{idx}_abcd1234",
        text=text,
        metadata={"source_path": "/data/test.pdf", "chunk_index": idx},
    )


def _make_llm_response(content: str):
    resp = MagicMock()
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Rule-based enrichment tests
# ---------------------------------------------------------------------------

class TestRuleBasedEnrich:
    def test_returns_required_keys(self):
        result = _rule_based_enrich("Some content here.")
        assert "title" in result
        assert "summary" in result
        assert "tags" in result

    def test_extracts_markdown_heading_as_title(self):
        text = "## Introduction to Machine Learning\n\nThis section covers basics."
        result = _rule_based_enrich(text)
        assert "Introduction to Machine Learning" in result["title"]

    def test_first_line_as_title_when_no_heading(self):
        text = "Quick Start Guide\n\nFollow these steps to get started."
        result = _rule_based_enrich(text)
        assert "Quick Start Guide" in result["title"]

    def test_title_truncated_to_80_chars(self):
        long_line = "A" * 100
        result = _rule_based_enrich(long_line)
        assert len(result["title"]) <= 80

    def test_summary_non_empty_for_normal_text(self):
        text = "This is a detailed explanation of neural networks and their applications."
        result = _rule_based_enrich(text)
        assert len(result["summary"]) > 0

    def test_summary_truncated_to_300_chars(self):
        long_text = "word " * 200
        result = _rule_based_enrich(long_text)
        assert len(result["summary"]) <= 304  # 300 + "..."

    def test_tags_is_list(self):
        result = _rule_based_enrich("Machine learning models require training data.")
        assert isinstance(result["tags"], list)

    def test_image_placeholders_ignored(self):
        text = "[IMAGE: doc_001_page1_001] Some content about neural networks."
        result = _rule_based_enrich(text)
        assert "[IMAGE:" not in result["title"]
        assert "[IMAGE:" not in result["summary"]

    def test_empty_text_fallback(self):
        result = _rule_based_enrich("")
        assert result["title"] == "Untitled"

    def test_title_fallback_for_blank_lines(self):
        result = _rule_based_enrich("\n\n\nHello world\n\nContent.")
        assert result["title"] == "Hello world"


# ---------------------------------------------------------------------------
# _parse_llm_json tests
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    def test_valid_json(self):
        raw = '{"title": "Test Title", "summary": "A summary.", "tags": ["a", "b"]}'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["title"] == "Test Title"
        assert result["tags"] == ["a", "b"]

    def test_strips_markdown_code_fence(self):
        raw = '```json\n{"title": "T", "summary": "S", "tags": []}\n```'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["title"] == "T"

    def test_returns_none_for_invalid_json(self):
        assert _parse_llm_json("not json at all") is None

    def test_returns_none_missing_required_field(self):
        raw = '{"title": "T", "summary": "S"}'  # missing tags
        assert _parse_llm_json(raw) is None

    def test_coerces_tags_to_list_when_wrong_type(self):
        raw = '{"title": "T", "summary": "S", "tags": "single"}'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["tags"] == []

    def test_filters_empty_tags(self):
        raw = '{"title": "T", "summary": "S", "tags": ["valid", "", "  "]}'
        result = _parse_llm_json(raw)
        assert result["tags"] == ["valid"]

    def test_returns_none_for_non_dict(self):
        assert _parse_llm_json('["a", "b"]') is None


# ---------------------------------------------------------------------------
# MetadataEnricher - rule mode (use_llm=False)
# ---------------------------------------------------------------------------

class TestMetadataEnricherRuleMode:
    def test_adds_title_summary_tags_to_metadata(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        chunks = [_make_chunk("## Section 1\n\nContent about databases.")]
        result = enricher.transform(chunks)
        assert result[0].metadata["title"]
        assert result[0].metadata["summary"]
        assert isinstance(result[0].metadata["tags"], list)

    def test_enriched_by_rule(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        chunks = [_make_chunk("Sample text for enrichment.")]
        result = enricher.transform(chunks)
        assert result[0].metadata["enriched_by"] == "rule"

    def test_preserves_existing_metadata(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        chunks = [_make_chunk("Content.", 0)]
        result = enricher.transform(chunks)
        assert result[0].metadata["source_path"] == "/data/test.pdf"
        assert result[0].metadata["chunk_index"] == 0

    def test_processes_multiple_chunks(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        chunks = [_make_chunk(f"Chunk {i} content.", i) for i in range(5)]
        result = enricher.transform(chunks)
        assert len(result) == 5
        for r in result:
            assert r.metadata["enriched_by"] == "rule"

    def test_does_not_mutate_original_chunk(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        chunk = _make_chunk("Original text.")
        original_meta = copy.deepcopy(chunk.metadata)
        enricher.transform([chunk])
        assert chunk.metadata == original_meta

    def test_failed_chunk_marked_none(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)

        # Simulate _enrich_chunk raising by patching it directly
        with patch.object(enricher, "_enrich_chunk", side_effect=RuntimeError("fail")):
            chunk = _make_chunk("some text")
            result = enricher.transform([chunk])
        assert result[0].metadata.get("enriched_by") == "none"
        assert "enrich_error" in result[0].metadata


# ---------------------------------------------------------------------------
# MetadataEnricher - LLM mode
# ---------------------------------------------------------------------------

class TestMetadataEnricherLlmMode:
    def _make_enricher_with_mock_llm(self, llm_response: str):
        settings = _make_settings(use_llm=True)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _make_llm_response(llm_response)
        return MetadataEnricher(settings, llm=mock_llm)

    def test_llm_called_and_result_used(self):
        llm_output = '{"title": "Deep Learning", "summary": "Overview of deep learning.", "tags": ["neural", "deep learning"]}'
        enricher = self._make_enricher_with_mock_llm(llm_output)
        chunks = [_make_chunk("Convolutional neural networks are used in image classification.")]
        result = enricher.transform(chunks)
        assert result[0].metadata["enriched_by"] == "llm"
        assert result[0].metadata["title"] == "Deep Learning"
        assert result[0].metadata["summary"] == "Overview of deep learning."
        assert "neural" in result[0].metadata["tags"]

    def test_llm_failure_falls_back_to_rule(self):
        settings = _make_settings(use_llm=True)
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("connection error")
        enricher = MetadataEnricher(settings, llm=mock_llm)
        chunks = [_make_chunk("Fallback content about databases.")]
        result = enricher.transform(chunks)
        assert result[0].metadata["enriched_by"] == "rule"
        assert result[0].metadata["enrich_fallback_reason"] == "llm_returned_none"
        # Rule-based values should be present
        assert result[0].metadata["title"]

    def test_llm_invalid_json_falls_back_to_rule(self):
        enricher = self._make_enricher_with_mock_llm("not valid json")
        chunks = [_make_chunk("Content for testing invalid JSON fallback.")]
        result = enricher.transform(chunks)
        assert result[0].metadata["enriched_by"] == "rule"
        assert result[0].metadata.get("enrich_fallback_reason") == "llm_returned_none"

    def test_llm_json_with_code_fence(self):
        llm_output = '```json\n{"title": "API Design", "summary": "RESTful API patterns.", "tags": ["api", "rest"]}\n```'
        enricher = self._make_enricher_with_mock_llm(llm_output)
        chunks = [_make_chunk("RESTful APIs follow stateless principles.")]
        result = enricher.transform(chunks)
        assert result[0].metadata["enriched_by"] == "llm"
        assert result[0].metadata["title"] == "API Design"

    def test_llm_not_called_when_use_llm_false(self):
        settings = _make_settings(use_llm=False)
        mock_llm = MagicMock()
        enricher = MetadataEnricher(settings, llm=mock_llm)
        chunks = [_make_chunk("Some content.")]
        enricher.transform(chunks)
        mock_llm.chat.assert_not_called()

    def test_llm_not_called_for_empty_text(self):
        settings = _make_settings(use_llm=True)
        mock_llm = MagicMock()
        enricher = MetadataEnricher(settings, llm=mock_llm)
        chunks = [_make_chunk("")]
        enricher.transform(chunks)
        mock_llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# TraceContext integration
# ---------------------------------------------------------------------------

class TestMetadataEnricherTrace:
    def test_trace_record_stage_called(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        mock_trace = MagicMock()
        chunks = [_make_chunk("Content A."), _make_chunk("Content B.")]
        enricher.transform(chunks, trace=mock_trace)
        mock_trace.record_stage.assert_called_once()
        call_kwargs = mock_trace.record_stage.call_args
        assert call_kwargs[0][0] == "metadata_enricher"

    def test_trace_counts_rule_enriched(self):
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=None)
        mock_trace = MagicMock()
        chunks = [_make_chunk(f"Content {i}.") for i in range(3)]
        enricher.transform(chunks, trace=mock_trace)
        _, kwargs = mock_trace.record_stage.call_args
        assert kwargs["total"] == 3
        assert kwargs["rule_enriched"] == 3
        assert kwargs["llm_enriched"] == 0

    def test_trace_counts_llm_enriched(self):
        settings = _make_settings(use_llm=True)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _make_llm_response(
            '{"title": "T", "summary": "S", "tags": ["t"]}'
        )
        enricher = MetadataEnricher(settings, llm=mock_llm)
        mock_trace = MagicMock()
        chunks = [_make_chunk("Content about LLM.")]
        enricher.transform(chunks, trace=mock_trace)
        _, kwargs = mock_trace.record_stage.call_args
        assert kwargs["llm_enriched"] == 1
        assert kwargs["rule_enriched"] == 0
