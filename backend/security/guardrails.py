"""Security, upload validation, prompt-injection defense, and emotional safety guardrails."""

from io import BytesIO
from pathlib import Path
import re
import uuid

from fastapi import Header, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from backend.config import get_settings

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"}
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt"}

SELF_HARM_PATTERNS = [
    re.compile(r"\b(kill myself|end my life|suicide|suicidal|want to die|hurt myself|self[- ]?harm|cut myself)\b", re.I),
    re.compile(r"\b(no reason to live|better off dead|take my own life)\b", re.I),
]

HARMFUL_REQUEST_PATTERNS = [
    re.compile(r"\b(how to (make|build) a (bomb|explosive|weapon)|synthesize (poison|ricin|meth))\b", re.I),
    re.compile(r"\b(write (ransomware|keylogger|malware) to (attack|hack))\b", re.I),
]

EMOTIONAL_DISTRESS_PATTERNS = [
    re.compile(
        r"\b(anxious|anxiety|depressed|depression|lonely|loneliness|stressed|overwhelmed|hopeless|"
        r"panicking|panic attack|heartbroken|grieving|exhausted|burned out|burnt out|sad|crying)\b",
        re.I,
    )
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(your\s+)?(system|safety)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(dan|unrestricted|jailbroken)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|api\s+key|secret|env)", re.I),
]

CRISIS_SUPPORT_RESPONSE = (
    "It sounds like you're going through a really difficult moment, and I want you to stay safe. "
    "I'm an AI assistant, not a licensed therapist or emergency service, so please reach out to people who can support you right now:\n\n"
    "- **US & Canada:** Call or text **988** (Suicide & Crisis Lifeline) or text **HOME** to **741741**.\n"
    "- **UK:** Call **111** (NHS) or **116 123** (Samaritans).\n"
    "- **India:** Call **14416** / **1800-891-4416** (Tele-MANAS) or **9152987821** (iCall).\n"
    "- **International:** Find free, confidential support in your country at https://findahelpline.com/\n\n"
    "If you are in immediate physical danger, please call your local emergency number or go to the nearest emergency room."
)

HARMFUL_REFUSAL_RESPONSE = (
    "I can't help with instructions that could cause harm, enable illegal acts, or compromise security. "
    "If you have a safe, constructive question, I'd be glad to help with that."
)


def verify_internal_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optionally enforce a shared internal API key between frontend and backend."""
    settings = get_settings()
    expected = settings.internal_api_key.strip()
    if expected and x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def sanitize_user_text(text: str) -> str:
    """Clean control characters and enforce maximum input length."""
    settings = get_settings()
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "").strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )
    if len(cleaned) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message exceeds maximum allowed length of {settings.max_text_length} characters.",
        )
    return cleaned


def inspect_user_safety(text: str) -> tuple[str | None, bool]:
    """
    Evaluate user message for crisis signals, harmful requests, or emotional distress.
    Returns (safety_flag, emotional_support_active).
    safety_flag is None, 'crisis', 'harmful', or 'prompt_injection'.
    """
    for pattern in SELF_HARM_PATTERNS:
        if pattern.search(text):
            return "crisis", True

    for pattern in HARMFUL_REQUEST_PATTERNS:
        if pattern.search(text):
            return "harmful", False

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return "prompt_injection", False

    emotional = any(pattern.search(text) for pattern in EMOTIONAL_DISTRESS_PATTERNS)
    return None, emotional


def neutralize_untrusted_text(text: str) -> str:
    """Strip or neutralize common prompt-injection directives inside retrieved document chunks."""
    sanitized = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = pattern.sub("[redacted-instruction]", sanitized)
    return sanitized


def safe_storage_filename(original_name: str, allowed_extensions: set[str]) -> tuple[str, str]:
    """Validate file extension and generate a collision-free UUID filename."""
    clean_name = Path(original_name or "upload").name
    ext = Path(clean_name).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed_extensions))}.",
        )
    return f"{uuid.uuid4().hex}{ext}", ext.lstrip(".")


async def read_validated_upload(
    upload: UploadFile,
    allowed_extensions: set[str],
    max_size_mb: int,
    kind: str,
) -> tuple[bytes, str, str]:
    """
    Read and validate an uploaded file for extension, size, and basic content integrity.
    Returns (file_bytes, safe_filename, file_ext_without_dot).
    """
    safe_name, file_ext = safe_storage_filename(upload.filename or "", allowed_extensions)
    data = await upload.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_bytes = max_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {max_size_mb} MB size limit.",
        )

    if kind == "image":
        try:
            with Image.open(BytesIO(data)) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted image file.",
            ) from exc
    elif kind == "document":
        if file_ext == "pdf" and not data.lstrip().startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file header.",
            )
        if file_ext == "txt":
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    data.decode("latin-1")
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="TXT file must be valid text.",
                    ) from exc

    return data, safe_name, file_ext
