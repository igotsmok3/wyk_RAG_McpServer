"""AzureVisionLLM: Azure OpenAI Vision LLM backend (vision provider='azure')."""
from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APIError, APITimeoutError, AzureOpenAI
from PIL import Image

from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import register_vision_provider

if TYPE_CHECKING:
    from core.settings import VisionLLMSettings

_DEFAULT_API_VERSION = "2024-02-01"


class AzureVisionLLM(BaseVisionLLM):
    """Vision LLM backed by the Azure OpenAI GPT-4o / GPT-4-Vision API."""

    def __init__(self, settings: VisionLLMSettings) -> None:
        self._settings = settings
        if not settings.azure_endpoint:
            raise ValueError("[azure-vision] azure_endpoint must be set in config")
        self._client = AzureOpenAI(
            api_key=settings.api_key,
            azure_endpoint=settings.azure_endpoint,
            api_version=settings.api_version or _DEFAULT_API_VERSION,
        )

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a text + image prompt to Azure OpenAI Vision and return ChatResponse.

        Args:
            text: Text prompt to accompany the image.
            image: File path (str) or raw image bytes.
            trace: Optional TraceContext (unused here, reserved for observability).
        """
        if not text:
            raise ValueError("[azure-vision] text must not be empty")

        image_bytes = self.preprocess_image(image, self._settings.max_image_size)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        deployment = self._settings.deployment_name or self._settings.model
        if not deployment:
            raise ValueError(
                "[azure-vision] deployment_name or model must be configured"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        try:
            resp = self._client.chat.completions.create(
                model=deployment,
                messages=messages,  # type: ignore[arg-type]
            )
        except APITimeoutError as e:
            raise TimeoutError(f"[azure-vision] Request timed out: {e}") from e
        except APIConnectionError as e:
            raise ConnectionError(f"[azure-vision] Connection failed: {e}") from e
        except APIError as e:
            raise RuntimeError(
                f"[azure-vision] API error (status={e.status_code}): {e.message}"
            ) from e

        choice = resp.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else None,
        )

    def preprocess_image(self, image: str | bytes, max_size: int = 2048) -> bytes:
        """Load image and resize if either dimension exceeds max_size."""
        raw = super().preprocess_image(image, max_size)

        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if w <= max_size and h <= max_size:
            return raw

        ratio = min(max_size / w, max_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        if fmt.upper() not in {"JPEG", "PNG", "WEBP", "GIF"}:
            fmt = "JPEG"
        img.save(buf, format=fmt)
        return buf.getvalue()


register_vision_provider("azure", AzureVisionLLM)
