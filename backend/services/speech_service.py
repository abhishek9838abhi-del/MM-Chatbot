"""Speech-to-Text (ASR) and Text-to-Speech (TTS) services."""

import logging
from pathlib import Path
import uuid

from fastapi import HTTPException, status
from huggingface_hub import InferenceClient

from backend.config import get_settings

logger = logging.getLogger(__name__)


class SpeechService:
    """Handles audio transcription via Hugging Face ASR and voice synthesis via HF TTS + fallbacks."""

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

    def _transcribe_gemini(self, audio_path: Path) -> str:
        api_key = self.settings.gemini_api_key.strip()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google Gemini API key (GEMINI_API_KEY) is not configured in .env. Get a free key at https://aistudio.google.com",
            )
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            audio_bytes = audio_path.read_bytes()
            ext = audio_path.suffix.lower().lstrip(".")
            mime_type = f"audio/{ext}" if ext != "mp3" else "audio/mpeg"
            response = client.models.generate_content(
                model=self.settings.gemini_chat_model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    "Transcribe this audio verbatim. Output ONLY the transcribed words with no commentary or formatting.",
                ],
            )
            text = response.text.strip() if response.text else ""
            if not text:
                raise ValueError("Empty transcription returned by Gemini.")
            return text
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Gemini audio transcription failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Audio transcription error: {exc}",
            ) from exc

    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe an audio file into text using Gemini or Hugging Face ASR."""
        if self.settings.ai_provider.lower().strip() == "gemini":
            return self._transcribe_gemini(audio_path)

        client = self._get_client()
        try:
            result = client.automatic_speech_recognition(
                audio=str(audio_path),
                model=self.settings.hf_asr_model,
            )
            text = getattr(result, "text", None)
            if text is None and isinstance(result, dict):
                text = result.get("text")
            if not text or not str(text).strip():
                raise ValueError("Empty transcription returned by ASR model.")
            return str(text).strip()
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("HF ASR failed: %s", exc.__class__.__name__)
            err_str = str(exc)
            if "402" in err_str or "Payment Required" in err_str:
                detail = "Hugging Face Inference credits exhausted (402 Payment Required). Please add credits or update HF_TOKEN."
            elif "401" in err_str or "Unauthorized" in err_str:
                detail = "Hugging Face API token is invalid or unauthorized (401)."
            else:
                detail = "Speech recognition service is currently unavailable. Please try again shortly."
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            ) from exc

    def synthesize_speech(self, text: str) -> Path | None:
        """
        Convert assistant text to an audio file stored in storage/audio/.
        Tries Hugging Face TTS first (when HF_TOKEN is set), then configured local/gTTS fallback.
        Returns the generated file Path, or None if all TTS engines fail gracefully.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return None

        # Keep spoken output concise enough for reliable TTS synthesis
        spoken_text = clean_text[:1500]
        self.settings.ensure_storage_dirs()

        # 1. Try Hugging Face Inference TTS
        if self.settings.hf_token.strip():
            try:
                client = InferenceClient(
                    api_key=self.settings.hf_token.strip(),
                    timeout=self.settings.hf_timeout_seconds,
                )
                audio_bytes = client.text_to_speech(
                    text=spoken_text,
                    model=self.settings.hf_tts_model,
                )
                if audio_bytes:
                    out_path = self.settings.audio_dir / f"tts_{uuid.uuid4().hex}.flac"
                    out_path.write_bytes(audio_bytes)
                    return out_path
            except Exception as exc:
                logger.warning("HF TTS unavailable (%s), trying fallback.", exc.__class__.__name__)

        # 2. Fallback engine
        fallback = self.settings.tts_fallback_engine.lower().strip()
        if fallback == "pyttsx3":
            path = self._synthesize_pyttsx3(spoken_text)
            if path is not None:
                return path
            return self._synthesize_gtts(spoken_text)
        if fallback == "gtts":
            path = self._synthesize_gtts(spoken_text)
            if path is not None:
                return path
            return self._synthesize_pyttsx3(spoken_text)

        return None

    def _synthesize_pyttsx3(self, text: str) -> Path | None:
        try:
            import pyttsx3

            out_path = self.settings.audio_dir / f"tts_{uuid.uuid4().hex}.wav"
            engine = pyttsx3.init()
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
            engine.stop()
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
        except Exception as exc:
            logger.warning("pyttsx3 fallback failed: %s", exc.__class__.__name__)
        return None

    def _synthesize_gtts(self, text: str) -> Path | None:
        try:
            from gtts import gTTS

            out_path = self.settings.audio_dir / f"tts_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(str(out_path))
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
        except Exception as exc:
            logger.warning("gTTS fallback failed: %s", exc.__class__.__name__)
        return None


speech_service = SpeechService()
