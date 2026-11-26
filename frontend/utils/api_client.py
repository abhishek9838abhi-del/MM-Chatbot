"""HTTP client wrapper for communicating with the FastAPI backend."""

import os
from typing import Any

from dotenv import load_dotenv
import requests

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
DEFAULT_TIMEOUT = 95


class BackendAPIError(Exception):
    """Raised when the FastAPI backend returns an error or is unreachable."""


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if INTERNAL_API_KEY:
        headers["X-API-Key"] = INTERNAL_API_KEY
    return headers


def _handle_response(resp: requests.Response) -> Any:
    if resp.status_code == 204:
        return None
    try:
        payload = resp.json()
    except ValueError:
        payload = None

    if not resp.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else resp.text
        raise BackendAPIError(detail or f"Backend error ({resp.status_code}).")
    return payload


def get_health() -> dict[str, Any]:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Backend API is unreachable. Start FastAPI with uvicorn.") from exc


def list_conversations() -> list[dict[str, Any]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/conversations", headers=_headers(), timeout=15)
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to load conversations.") from exc


def create_conversation(title: str = "New Conversation") -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/conversations",
            json={"title": title},
            headers=_headers(),
            timeout=15,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to create conversation.") from exc


def get_conversation(conversation_id: int) -> dict[str, Any]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/api/conversations/{conversation_id}",
            headers=_headers(),
            timeout=15,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to load conversation messages.") from exc


def clear_conversation_messages(conversation_id: int) -> None:
    try:
        resp = requests.delete(
            f"{BACKEND_URL}/api/conversations/{conversation_id}/messages",
            headers=_headers(),
            timeout=15,
        )
        _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to clear conversation messages.") from exc


def delete_conversation(conversation_id: int) -> None:
    try:
        resp = requests.delete(
            f"{BACKEND_URL}/api/conversations/{conversation_id}",
            headers=_headers(),
            timeout=15,
        )
        _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to delete conversation.") from exc


def send_text_chat(
    message: str,
    conversation_id: int | None = None,
    use_rag: bool = True,
    generate_audio: bool = False,
) -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "conversation_id": conversation_id,
                "message": message,
                "use_rag": use_rag,
                "generate_audio": generate_audio,
            },
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Chat request failed or timed out.") from exc


def analyze_image(
    file_bytes: bytes,
    filename: str,
    prompt: str,
    conversation_id: int | None = None,
    generate_audio: bool = False,
) -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/image/analyze",
            files={"file": (filename, file_bytes)},
            data={
                "prompt": prompt,
                "conversation_id": "" if conversation_id is None else str(conversation_id),
                "generate_audio": str(generate_audio).lower(),
            },
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Image analysis request failed or timed out.") from exc


def send_audio_chat(
    file_bytes: bytes,
    filename: str,
    conversation_id: int | None = None,
    use_rag: bool = True,
    generate_audio: bool = True,
) -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/audio/chat",
            files={"file": (filename, file_bytes)},
            data={
                "conversation_id": "" if conversation_id is None else str(conversation_id),
                "use_rag": str(use_rag).lower(),
                "generate_audio": str(generate_audio).lower(),
            },
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Audio chat request failed or timed out.") from exc


def synthesize_tts(text: str) -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/audio/tts",
            json={"text": text},
            headers=_headers(),
            timeout=45,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Voice synthesis failed.") from exc


def upload_document(
    file_bytes: bytes,
    filename: str,
    conversation_id: int | None = None,
) -> dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/documents/upload",
            files={"file": (filename, file_bytes)},
            data={"conversation_id": "" if conversation_id is None else str(conversation_id)},
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Document upload failed.") from exc


def list_documents(conversation_id: int | None = None) -> list[dict[str, Any]]:
    try:
        params = {"conversation_id": conversation_id} if conversation_id is not None else {}
        resp = requests.get(
            f"{BACKEND_URL}/api/documents",
            params=params,
            headers=_headers(),
            timeout=15,
        )
        return _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to load documents.") from exc


def delete_document(document_id: int) -> None:
    try:
        resp = requests.delete(
            f"{BACKEND_URL}/api/documents/{document_id}",
            headers=_headers(),
            timeout=15,
        )
        _handle_response(resp)
    except requests.RequestException as exc:
        raise BackendAPIError("Unable to delete document.") from exc


def media_full_url(relative_path: str) -> str:
    if relative_path.startswith("http"):
        return relative_path
    return f"{BACKEND_URL}{relative_path}"


def fetch_media_bytes(relative_path: str) -> bytes:
    resp = requests.get(media_full_url(relative_path), headers=_headers(), timeout=20)
    if not resp.ok:
        raise BackendAPIError("Unable to load media file.")
    return resp.content
