"""Hugging Face Vision-Language service for image understanding and Q&A."""

import base64
import logging
from pathlib import Path

from fastapi import HTTPException, status
from huggingface_hub import InferenceClient

from backend.config import get_settings

logger = logging.getLogger(__name__)

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class VisionService:
    """Service wrapping Hugging Face Vision-Language models via InferenceClient."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_client(self) -> InferenceClient:
        token = self.settings.hf_token.strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hugging Face API token (HF_TOKEN) is not configured on the server.",
            )
        return InferenceClient(api_key=token, timeout=self.settings.hf_timeout_seconds)

    def _analyze_gemini(self, image_path: Path, prompt: str) -> str:
        api_key = self.settings.gemini_api_key.strip()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google Gemini API key (GEMINI_API_KEY) is not configured in .env. Get a free key at https://aistudio.google.com",
            )
        try:
            from google import genai
            from google.genai import types
            from PIL import Image

            client = genai.Client(api_key=api_key)
            img = Image.open(image_path)
            response = client.models.generate_content(
                model=self.settings.gemini_vision_model,
                contents=[img, prompt],
            )
            if not response.text or not response.text.strip():
                raise ValueError("Empty response from Gemini vision model.")
            return response.text.strip()
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Gemini vision analysis failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Gemini Vision error: {exc}",
            ) from exc

    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze an image file and answer the user's question using Gemini or Hugging Face."""
        if self.settings.ai_provider.lower().strip() == "gemini":
            return self._analyze_gemini(image_path, prompt)

        ext = image_path.suffix.lower()
        mime_type = MIME_MAP.get(ext, "image/jpeg")
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{encoded}"

        client = self._get_client()
        try:
            response = client.chat_completion(
                model=self.settings.hf_vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise vision-language assistant. Describe only what is actually "
                            "visible in the image and answer the user's question accurately."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
                max_tokens=700,
                temperature=0.4,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from vision model.")
            return content.strip()
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("HF vision completion failed: %s", exc.__class__.__name__)
            err_str = str(exc)
            if "402" in err_str or "Payment Required" in err_str:
                detail = "Hugging Face Inference credits exhausted (402 Payment Required). Please add credits or update HF_TOKEN."
            elif "401" in err_str or "Unauthorized" in err_str:
                detail = "Hugging Face API token is invalid or unauthorized (401)."
            else:
                detail = "The vision-language model is currently unavailable. Please try again shortly."
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            ) from exc


vision_service = VisionService()
