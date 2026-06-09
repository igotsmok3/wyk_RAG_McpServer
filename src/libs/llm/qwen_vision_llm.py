"""QwenVisionLLM: Qwen-VL vision backend via DashScope OpenAI-compatible API."""
from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from PIL import Image

from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import register_vision_provider

if TYPE_CHECKING:
    from core.settings import VisionLLMSettings

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-vl-plus"


class QwenVisionLLM(BaseVisionLLM):
    """Vision LLM backed by DashScope Qwen-VL via OpenAI-compatible API."""

    def __init__(self, settings: VisionLLMSettings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url or _DEFAULT_BASE_URL,
        )

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        if not text:
            raise ValueError("[qwen-vision] text must not be empty")

        image_bytes = self.preprocess_image(image, self._settings.max_image_size)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        model = self._settings.model or _DEFAULT_MODEL

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
                model=model,
                messages=messages,  # type: ignore[arg-type]
            )
        except APITimeoutError as e:
            raise TimeoutError(f"[qwen-vision] Request timed out: {e}") from e
        except APIConnectionError as e:
            raise ConnectionError(f"[qwen-vision] Connection failed: {e}") from e
        except APIError as e:
            raise RuntimeError(
                f"[qwen-vision] API error (status={e.status_code}): {e.message}"
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


register_vision_provider("qwen", QwenVisionLLM)
