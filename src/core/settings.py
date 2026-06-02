"""Settings: load and validate config/settings.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

import os


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    provider: str = ""
    model: str = ""
    deployment_name: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingSettings:
    provider: str = ""
    model: str = ""
    dimensions: int = 1536
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class VisionLLMSettings:
    enabled: bool = False
    provider: str = ""
    model: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    api_key: str = ""
    base_url: str = ""
    max_image_size: int = 2048


@dataclass
class VectorStoreSettings:
    provider: str = ""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "knowledge_hub"


@dataclass
class RetrievalSettings:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass
class RerankSettings:
    enabled: bool = False
    provider: str = "none"
    model: str = ""
    top_k: int = 5


@dataclass
class EvaluationSettings:
    enabled: bool = False
    provider: str = "custom"
    metrics: list[str] = field(default_factory=list)


@dataclass
class ObservabilitySettings:
    log_level: str = "INFO"
    trace_enabled: bool = False
    trace_file: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class ChunkRefinerSettings:
    use_llm: bool = False


@dataclass
class MetadataEnricherSettings:
    use_llm: bool = False


@dataclass
class IngestionSettings:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"
    batch_size: int = 100
    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _from_dict(cls, data: dict[str, Any]):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def _parse_ingestion(raw: dict[str, Any]) -> IngestionSettings:
    cr = ChunkRefinerSettings(**raw.pop("chunk_refiner", {}))
    me = MetadataEnricherSettings(**raw.pop("metadata_enricher", {}))
    return IngestionSettings(**raw, chunk_refiner=cr, metadata_enricher=me)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_settings(path: str = "config/settings.yaml") -> Settings:
    """Load settings.yaml → Settings. Raises ValueError on missing required fields."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = os.path.expandvars(f.read())  # 替换 ${VAR} 和 $VAR
        raw: dict[str, Any] = yaml.safe_load(content) or {}

    llm_raw = raw.get("llm", {})
    embedding_raw = raw.get("embedding", {})
    vector_store_raw = raw.get("vector_store", {})

    settings = Settings(
        llm=_from_dict(LLMSettings, llm_raw),
        embedding=_from_dict(EmbeddingSettings, embedding_raw),
        vector_store=_from_dict(VectorStoreSettings, vector_store_raw),
        vision_llm=_from_dict(VisionLLMSettings, raw.get("vision_llm", {})),
        retrieval=_from_dict(RetrievalSettings, raw.get("retrieval", {})),
        rerank=_from_dict(RerankSettings, raw.get("rerank", {})),
        evaluation=_from_dict(EvaluationSettings, raw.get("evaluation", {})),
        observability=_from_dict(ObservabilitySettings, raw.get("observability", {})),
        ingestion=_parse_ingestion(dict(raw.get("ingestion", {}))),
    )

    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Raise ValueError listing every missing required field."""
    errors: list[str] = []

    if not settings.llm.provider:
        errors.append("llm.provider")
    if not settings.embedding.provider:
        errors.append("embedding.provider")
    if not settings.vector_store.provider:
        errors.append("vector_store.provider")

    if errors:
        raise ValueError(
            "Missing required config fields: " + ", ".join(errors)
        )
