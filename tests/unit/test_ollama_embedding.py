"""Tests for OllamaEmbedding (B7.4). All HTTP calls are mocked."""
from __future__ import annotations

import sys
import os
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.embedding.embedding_factory import EmbeddingFactory


@dataclass
class EmbedCfg:
    provider: str
    model: str = "nomic-embed-text"
    dimensions: int = 768
    api_key: str = ""
    base_url: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""


def _fake_response(embeddings: list[list[float]], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.ok = status < 400
    resp.status_code = status
    resp.text = "error" if status >= 400 else ""
    resp.json.return_value = {"model": "nomic-embed-text", "embeddings": embeddings}
    return resp


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingFactory:
    def test_factory_routes_to_ollama(self):
        from libs.embedding.ollama_embedding import OllamaEmbedding
        cfg = EmbedCfg(provider="ollama")
        emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, OllamaEmbedding)

    def test_factory_case_insensitive(self):
        from libs.embedding.ollama_embedding import OllamaEmbedding
        cfg = EmbedCfg(provider="Ollama")
        emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, OllamaEmbedding)

    def test_empty_model_raises(self):
        cfg = EmbedCfg(provider="ollama", model="")
        with pytest.raises(ValueError, match="model"):
            EmbeddingFactory.create(cfg)


# ---------------------------------------------------------------------------
# Normal responses
# ---------------------------------------------------------------------------

class TestOllamaEmbed:
    def test_embed_returns_vectors(self):
        cfg = EmbedCfg(provider="ollama")
        vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response(vecs)):
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["hello", "world"])
        assert result == vecs

    def test_embed_count_matches_input(self):
        cfg = EmbedCfg(provider="ollama")
        vecs = [[float(i)] * 4 for i in range(3)]
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response(vecs)):
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["a", "b", "c"])
        assert len(result) == 3

    def test_embed_posts_to_api_embed_endpoint(self):
        cfg = EmbedCfg(provider="ollama", base_url="http://localhost:11434")
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response([[0.1]])) as mock_post:
            emb = EmbeddingFactory.create(cfg)
            emb.embed(["test"])
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:11434/api/embed"

    def test_embed_uses_default_base_url(self):
        cfg = EmbedCfg(provider="ollama", base_url="")
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response([[0.1]])) as mock_post:
            emb = EmbeddingFactory.create(cfg)
            emb.embed(["test"])
        call_url = mock_post.call_args[0][0]
        assert "localhost:11434" in call_url

    def test_embed_uses_custom_base_url(self):
        cfg = EmbedCfg(provider="ollama", base_url="http://remote-host:11434")
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response([[0.1]])) as mock_post:
            emb = EmbeddingFactory.create(cfg)
            emb.embed(["test"])
        call_url = mock_post.call_args[0][0]
        assert "remote-host:11434" in call_url

    def test_embed_sends_correct_payload(self):
        cfg = EmbedCfg(provider="ollama", model="mxbai-embed-large")
        texts = ["hello", "world"]
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response([[0.1], [0.2]])) as mock_post:
            emb = EmbeddingFactory.create(cfg)
            emb.embed(texts)
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "mxbai-embed-large"
        assert payload["input"] == texts

    def test_embed_accepts_trace_param(self):
        cfg = EmbedCfg(provider="ollama")
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response([[0.1, 0.2]])):
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["hello"], trace=object())
        assert len(result) == 1

    def test_trailing_slash_stripped_from_url(self):
        cfg = EmbedCfg(provider="ollama", base_url="http://localhost:11434/")
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=_fake_response([[0.1]])) as mock_post:
            emb = EmbeddingFactory.create(cfg)
            emb.embed(["test"])
        call_url = mock_post.call_args[0][0]
        assert not call_url.startswith("http://localhost:11434//")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestOllamaEmbedValidation:
    def test_empty_texts_raises(self):
        cfg = EmbedCfg(provider="ollama")
        emb = EmbeddingFactory.create(cfg)
        with pytest.raises(ValueError, match="ollama"):
            emb.embed([])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestOllamaEmbedErrors:
    def test_connection_error_wrapped(self):
        import requests as req_mod
        cfg = EmbedCfg(provider="ollama")
        emb = EmbeddingFactory.create(cfg)
        with patch(
            "libs.embedding.ollama_embedding.requests.post",
            side_effect=req_mod.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(ConnectionError, match="ollama"):
                emb.embed(["hi"])

    def test_timeout_error_wrapped(self):
        import requests as req_mod
        cfg = EmbedCfg(provider="ollama")
        emb = EmbeddingFactory.create(cfg)
        with patch(
            "libs.embedding.ollama_embedding.requests.post",
            side_effect=req_mod.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(TimeoutError, match="ollama"):
                emb.embed(["hi"])

    def test_generic_request_exception_wrapped(self):
        import requests as req_mod
        cfg = EmbedCfg(provider="ollama")
        emb = EmbeddingFactory.create(cfg)
        with patch(
            "libs.embedding.ollama_embedding.requests.post",
            side_effect=req_mod.exceptions.RequestException("unexpected"),
        ):
            with pytest.raises(RuntimeError, match="ollama"):
                emb.embed(["hi"])

    def test_http_error_status_wrapped(self):
        cfg = EmbedCfg(provider="ollama")
        emb = EmbeddingFactory.create(cfg)
        with patch(
            "libs.embedding.ollama_embedding.requests.post",
            return_value=_fake_response([], status=404),
        ):
            with pytest.raises(RuntimeError, match="ollama"):
                emb.embed(["hi"])

    def test_missing_embeddings_key_raises(self):
        cfg = EmbedCfg(provider="ollama")
        bad_resp = MagicMock()
        bad_resp.ok = True
        bad_resp.status_code = 200
        bad_resp.json.return_value = {"model": "nomic-embed-text"}  # no 'embeddings' key
        emb = EmbeddingFactory.create(cfg)
        with patch("libs.embedding.ollama_embedding.requests.post", return_value=bad_resp):
            with pytest.raises(RuntimeError, match="embeddings"):
                emb.embed(["hi"])
