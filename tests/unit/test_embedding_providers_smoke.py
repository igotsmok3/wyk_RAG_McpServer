"""Smoke tests for OpenAI & Azure Embedding providers (B7.3).

All HTTP calls are mocked — no real network requests.
"""
import sys
import os
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.embedding.embedding_factory import EmbeddingFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed_response(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock openai.embeddings.create() response."""
    resp = MagicMock()
    items = []
    for i, vec in enumerate(embeddings):
        item = MagicMock()
        item.embedding = vec
        item.index = i
        items.append(item)
    resp.data = items
    return resp


@dataclass
class EmbedCfg:
    provider: str
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_key: str = "sk-test"
    base_url: str = ""
    azure_endpoint: str = "https://my-resource.openai.azure.com/"
    deployment_name: str = "text-embedding-ada-002"
    api_version: str = "2024-02-01"


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class TestOpenAIEmbedding:
    def test_factory_routes_to_openai(self):
        from libs.embedding.openai_embedding import OpenAIEmbedding
        cfg = EmbedCfg(provider="openai")
        with patch("libs.embedding.openai_embedding.OpenAI"):
            emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, OpenAIEmbedding)

    def test_embed_returns_vectors(self):
        cfg = EmbedCfg(provider="openai")
        vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        fake_resp = _fake_embed_response(vecs)
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["hello", "world"])
        assert result == vecs

    def test_embed_count_matches_input(self):
        cfg = EmbedCfg(provider="openai")
        texts = ["a", "b", "c"]
        vecs = [[float(i)] * 3 for i in range(3)]
        fake_resp = _fake_embed_response(vecs)
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(texts)
        assert len(result) == len(texts)

    def test_embed_passes_model(self):
        cfg = EmbedCfg(provider="openai", model="text-embedding-3-large")
        fake_resp = _fake_embed_response([[0.1]])
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            emb.embed(["test"])
            call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-large"

    def test_embed_empty_texts_raises(self):
        cfg = EmbedCfg(provider="openai")
        with patch("libs.embedding.openai_embedding.OpenAI"):
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(ValueError, match="openai"):
            emb.embed([])

    def test_embed_accepts_trace_param(self):
        cfg = EmbedCfg(provider="openai")
        fake_resp = _fake_embed_response([[0.1, 0.2]])
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["hello"], trace=None)
        assert len(result) == 1

    def test_connection_error_wrapped(self):
        from openai import APIConnectionError
        cfg = EmbedCfg(provider="openai")
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(ConnectionError, match="openai"):
            emb.embed(["hi"])

    def test_timeout_error_wrapped(self):
        from openai import APITimeoutError
        cfg = EmbedCfg(provider="openai")
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.side_effect = APITimeoutError(
                request=MagicMock()
            )
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(TimeoutError, match="openai"):
            emb.embed(["hi"])

    def test_api_error_wrapped(self):
        from openai import APIStatusError
        cfg = EmbedCfg(provider="openai")
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.embeddings.create.side_effect = APIStatusError(
                "Unauthorized", response=mock_response, body={"message": "Unauthorized"}
            )
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(RuntimeError, match="openai"):
            emb.embed(["hi"])

    def test_uses_custom_base_url(self):
        cfg = EmbedCfg(provider="openai", base_url="https://custom.endpoint/v1")
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            EmbeddingFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("base_url") == "https://custom.endpoint/v1"

    def test_result_order_stable(self):
        """Embeddings returned in same order as input texts."""
        cfg = EmbedCfg(provider="openai")
        vecs = [[float(i)] * 2 for i in range(5)]
        # Return data in reverse order to verify index sorting
        resp = MagicMock()
        items = []
        for i, vec in enumerate(vecs):
            item = MagicMock()
            item.embedding = vec
            item.index = i
            items.append(item)
        resp.data = list(reversed(items))
        with patch("libs.embedding.openai_embedding.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = resp
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["a", "b", "c", "d", "e"])
        assert result == vecs


# ---------------------------------------------------------------------------
# Azure provider
# ---------------------------------------------------------------------------

class TestAzureEmbedding:
    def test_factory_routes_to_azure(self):
        from libs.embedding.azure_embedding import AzureEmbedding
        cfg = EmbedCfg(provider="azure")
        with patch("libs.embedding.azure_embedding.AzureOpenAI"):
            emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, AzureEmbedding)

    def test_embed_returns_vectors(self):
        cfg = EmbedCfg(provider="azure", deployment_name="my-ada-002")
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        fake_resp = _fake_embed_response(vecs)
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["hello", "world"])
        assert result == vecs

    def test_uses_deployment_name_as_model(self):
        cfg = EmbedCfg(provider="azure", deployment_name="dep-ada", model="")
        fake_resp = _fake_embed_response([[0.1]])
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            emb.embed(["test"])
            call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "dep-ada"

    def test_missing_endpoint_raises(self):
        cfg = EmbedCfg(provider="azure", azure_endpoint="")
        with patch("libs.embedding.azure_embedding.AzureOpenAI"):
            with pytest.raises(ValueError, match="azure_endpoint"):
                EmbeddingFactory.create(cfg)

    def test_missing_deployment_and_model_raises(self):
        cfg = EmbedCfg(provider="azure", deployment_name="", model="")
        with patch("libs.embedding.azure_embedding.AzureOpenAI"):
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(ValueError, match="deployment_name"):
            emb.embed(["test"])

    def test_embed_empty_texts_raises(self):
        cfg = EmbedCfg(provider="azure")
        with patch("libs.embedding.azure_embedding.AzureOpenAI"):
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(ValueError, match="azure"):
            emb.embed([])

    def test_default_api_version_used(self):
        cfg = EmbedCfg(provider="azure", api_version="")
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            EmbeddingFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("api_version") == "2024-02-01"

    def test_custom_api_version_used(self):
        cfg = EmbedCfg(provider="azure", api_version="2024-06-01")
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            EmbeddingFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("api_version") == "2024-06-01"

    def test_connection_error_wrapped(self):
        from openai import APIConnectionError
        cfg = EmbedCfg(provider="azure")
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(ConnectionError, match="azure"):
            emb.embed(["hi"])

    def test_api_error_wrapped(self):
        from openai import APIStatusError
        cfg = EmbedCfg(provider="azure")
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_client.embeddings.create.side_effect = APIStatusError(
                "Forbidden", response=mock_response, body={"message": "Forbidden"}
            )
            emb = EmbeddingFactory.create(cfg)
        with pytest.raises(RuntimeError, match="azure"):
            emb.embed(["hi"])

    def test_embed_accepts_trace_param(self):
        cfg = EmbedCfg(provider="azure")
        fake_resp = _fake_embed_response([[0.1, 0.2]])
        with patch("libs.embedding.azure_embedding.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.embeddings.create.return_value = fake_resp
            emb = EmbeddingFactory.create(cfg)
            result = emb.embed(["hello"], trace=object())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Factory routing coverage
# ---------------------------------------------------------------------------

class TestFactoryRouting:
    def test_both_providers_registered(self):
        from libs.embedding.openai_embedding import OpenAIEmbedding
        from libs.embedding.azure_embedding import AzureEmbedding

        cfg = EmbedCfg(provider="openai")
        with patch("libs.embedding.openai_embedding.OpenAI"):
            emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, OpenAIEmbedding)

        cfg = EmbedCfg(provider="azure")
        with patch("libs.embedding.azure_embedding.AzureOpenAI"):
            emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, AzureEmbedding)

    def test_provider_name_case_insensitive(self):
        from libs.embedding.openai_embedding import OpenAIEmbedding
        cfg = EmbedCfg(provider="OpenAI")
        with patch("libs.embedding.openai_embedding.OpenAI"):
            emb = EmbeddingFactory.create(cfg)
        assert isinstance(emb, OpenAIEmbedding)
