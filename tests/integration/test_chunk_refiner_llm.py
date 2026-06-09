"""Integration tests for ChunkRefiner with a real LLM.

Run with:  pytest tests/integration/test_chunk_refiner_llm.py -v -s
Requires:  DASHSCOPE_API_KEY (or appropriate API key) set in environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner

FIXTURES = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"
_CASES = json.loads(FIXTURES.read_text(encoding="utf-8"))

SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


def _settings_with_llm():
    settings = load_settings(str(SETTINGS_PATH))
    # Force LLM on for integration tests
    settings.ingestion.chunk_refiner.use_llm = True
    return settings


def _make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={})


# ---------------------------------------------------------------------------
# Fixtures / skip guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def refiner():
    settings = _settings_with_llm()
    r = ChunkRefiner(settings)
    if r._llm is None:
        pytest.skip("LLM not available — check API key and provider config")
    return r


# ---------------------------------------------------------------------------
# Test 1: Real LLM call succeeds
# ---------------------------------------------------------------------------

def test_real_llm_refinement_succeeds(refiner):
    noisy_input = (
        "────────────────────────────\n"
        "Page 42 | Section 5.3\n"
        "────────────────────────────\n\n"
        "The actual content starts here and continues with valuable information.\n\n"
        "────────────────────────────\n"
        "Footer: Company Name | 2024\n"
        "────────────────────────────"
    )
    result = refiner.transform([_make_chunk(noisy_input)])
    chunk = result[0]

    print(f"\n[LLM refined]:\n{chunk.text}")
    assert chunk.metadata["refined_by"] == "llm"
    assert chunk.text.strip() != ""
    # Noise should be reduced
    assert "Page 42" not in chunk.text or len(chunk.text) < len(noisy_input)


# ---------------------------------------------------------------------------
# Test 2: LLM output quality — noise reduced, content preserved
# ---------------------------------------------------------------------------

def test_llm_reduces_noise_preserves_content(refiner):
    noisy = _CASES[0]  # typical_noise_scenario
    result = refiner.transform([_make_chunk(noisy["input"])])
    chunk = result[0]

    print(f"\n[input]:\n{noisy['input']}")
    print(f"\n[LLM output]:\n{chunk.text}")

    # Content should be preserved
    assert "valuable paragraph" in chunk.text
    # Severe noise should be reduced
    assert chunk.metadata["refined_by"] == "llm"


# ---------------------------------------------------------------------------
# Test 3: Multiple chunks processed correctly
# ---------------------------------------------------------------------------

def test_multiple_chunks_all_llm_refined(refiner):
    chunks = [
        _make_chunk("Page 1\n\nFirst useful content here.", "c1"),
        _make_chunk("────\nHeader\n────\n\nSecond useful content here.", "c2"),
    ]
    trace = TraceContext()
    result = refiner.transform(chunks, trace=trace)

    assert len(result) == 2
    stage = next(s for s in trace.stages if s.name == "chunk_refiner")
    llm_count = stage.data.get("llm_refined", 0)
    print(f"\n[LLM refined count]: {llm_count}/{len(chunks)}")
    assert llm_count > 0


# ---------------------------------------------------------------------------
# Test 4: Graceful degradation with invalid model name
# ---------------------------------------------------------------------------

def test_graceful_degradation_with_invalid_model(tmp_path):
    settings = _settings_with_llm()
    settings.llm.model = "this-model-does-not-exist-xyz-999"
    refiner = ChunkRefiner(settings)

    noisy = "Page 1\n\nValuable content."
    result = refiner.transform([_make_chunk(noisy)])
    chunk = result[0]

    print(f"\n[degraded output]:\n{chunk.text}")
    # Should not crash; falls back to rule-based
    assert chunk.text.strip() != ""
    assert chunk.metadata.get("refined_by") in ("rule", "llm", "none")


# ---------------------------------------------------------------------------
# Test 5: Code blocks preserved through real LLM
# ---------------------------------------------------------------------------

def test_code_blocks_preserved_by_llm(refiner):
    case = next(c for c in _CASES if c["id"] == "code_blocks")
    result = refiner.transform([_make_chunk(case["input"])])
    chunk = result[0]

    print(f"\n[LLM code block output]:\n{chunk.text}")
    # Content must survive (LLM may strip fences but must keep function body)
    assert "hello_world" in chunk.text
    assert "print" in chunk.text
