"""Tests for config loading and validation (A3)."""
import sys
import os
import textwrap
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.settings import load_settings, validate_settings, Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "settings.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


MINIMAL_YAML = """\
llm:
  provider: "openai"
  model: "gpt-4o"
embedding:
  provider: "openai"
vector_store:
  provider: "milvus"
"""


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_load_settings_returns_settings(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    s = load_settings(path)
    assert isinstance(s, Settings)


def test_load_settings_llm_fields(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    s = load_settings(path)
    assert s.llm.provider == "openai"
    assert s.llm.model == "gpt-4o"


def test_load_settings_embedding_fields(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    s = load_settings(path)
    assert s.embedding.provider == "openai"


def test_load_settings_vector_store_fields(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    s = load_settings(path)
    assert s.vector_store.provider == "milvus"


def test_load_settings_defaults_populated(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    s = load_settings(path)
    assert s.retrieval.dense_top_k == 20
    assert s.rerank.enabled is False
    assert s.observability.log_level == "INFO"


def test_load_real_settings_yaml():
    """Load the actual project settings.yaml from config/."""
    s = load_settings("config/settings.yaml")
    assert s.llm.provider == "qwen"
    assert s.embedding.provider == "qwen"
    assert s.vector_store.provider == "milvus"


def test_ingestion_nested_config(tmp_path):
    yaml = MINIMAL_YAML + textwrap.dedent("""\
    ingestion:
      chunk_size: 500
      chunk_refiner:
        use_llm: true
    """)
    path = _write_yaml(tmp_path, yaml)
    s = load_settings(path)
    assert s.ingestion.chunk_size == 500
    assert s.ingestion.chunk_refiner.use_llm is True


def test_vision_llm_settings(tmp_path):
    yaml = MINIMAL_YAML + textwrap.dedent("""\
    vision_llm:
      enabled: true
      provider: "azure"
    """)
    path = _write_yaml(tmp_path, yaml)
    s = load_settings(path)
    assert s.vision_llm.enabled is True
    assert s.vision_llm.provider == "azure"


# ---------------------------------------------------------------------------
# Validation / error-path tests
# ---------------------------------------------------------------------------

def test_missing_llm_provider_raises(tmp_path):
    yaml = """\
llm:
  model: "gpt-4o"
embedding:
  provider: "openai"
vector_store:
  provider: "milvus"
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="llm.provider"):
        load_settings(path)


def test_missing_embedding_provider_raises(tmp_path):
    yaml = """\
llm:
  provider: "openai"
embedding:
  model: "text-embedding-3-small"
vector_store:
  provider: "milvus"
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="embedding.provider"):
        load_settings(path)


def test_missing_vector_store_provider_raises(tmp_path):
    yaml = """\
llm:
  provider: "openai"
embedding:
  provider: "openai"
vector_store:
  host: "localhost"
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="vector_store.provider"):
        load_settings(path)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_settings("nonexistent_path/settings.yaml")


def test_error_message_lists_all_missing_fields(tmp_path):
    yaml = """\
llm:
  model: "gpt-4o"
embedding:
  model: "text-embedding-3-small"
vector_store:
  host: "localhost"
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError) as exc_info:
        load_settings(path)
    msg = str(exc_info.value)
    assert "llm.provider" in msg
    assert "embedding.provider" in msg
    assert "vector_store.provider" in msg
