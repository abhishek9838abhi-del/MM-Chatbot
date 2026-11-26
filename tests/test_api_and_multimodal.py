"""End-to-end API, multimodal, RAG, persistence, and security guardrail tests."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import Base
from backend.main import app


@pytest.fixture()
def client(tmp_path: Path):
    """Create a FastAPI TestClient backed by an isolated SQLite DB and temp storage."""
    db_path = tmp_path / "test_chatbot.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(56, 189, 248)).save(buf, format="PNG")
    return buf.getvalue()


def test_text_chat_and_conversation_persistence(client: TestClient) -> None:
    with patch("backend.routes.chat.llm_service.generate_reply", return_value="Hello! How can I help you today?"):
        res1 = client.post("/api/chat", json={"message": "Hi there!", "use_rag": False})
    assert res1.status_code == 200
    data1 = res1.json()
    conv_id = data1["conversation_id"]
    assert data1["reply"] == "Hello! How can I help you today?"

    # Follow-up question in same conversation
    with patch("backend.routes.chat.llm_service.generate_reply", return_value="Sure, let's continue.") as mock_llm:
        res2 = client.post(
            "/api/chat",
            json={"conversation_id": conv_id, "message": "Can you expand on that?", "use_rag": False},
        )
    assert res2.status_code == 200
    _, kwargs = mock_llm.call_args
    assert len(kwargs["history"]) == 2

    # Retrieve conversation history
    conv_res = client.get(f"/api/conversations/{conv_id}")
    assert conv_res.status_code == 200
    assert len(conv_res.json()["messages"]) == 4

    # Start a new conversation
    new_conv = client.post("/api/conversations", json={"title": "Second Chat"})
    assert new_conv.status_code == 201
    assert new_conv.json()["id"] != conv_id

    # Clear messages
    clr = client.delete(f"/api/conversations/{conv_id}/messages")
    assert clr.status_code == 204
    assert client.get(f"/api/conversations/{conv_id}").json()["messages"] == []


def test_image_upload_and_analysis(client: TestClient) -> None:
    png_bytes = _make_png_bytes()
    with patch("backend.routes.image.vision_service.analyze_image", return_value="A solid sky-blue square."):
        res = client.post(
            "/api/image/analyze",
            files={"file": ("sample.png", png_bytes, "image/png")},
            data={"prompt": "Describe this image."},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "A solid sky-blue square."
    assert body["image_url"].startswith("/api/media/images/")

    # Verify media endpoint serves the uploaded image
    media_res = client.get(body["image_url"])
    assert media_res.status_code == 200


def test_audio_upload_and_tts_response(client: TestClient, tmp_path: Path) -> None:
    fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
    settings = get_settings()
    settings.ensure_storage_dirs()
    out_audio = settings.audio_dir / "test_tts_output.wav"
    out_audio.write_bytes(fake_wav)

    with (
        patch("backend.routes.audio.speech_service.transcribe_audio", return_value="What is the weather?"),
        patch("backend.routes.audio.llm_service.generate_reply", return_value="I cannot check live weather."),
        patch("backend.routes.audio.speech_service.synthesize_speech", return_value=out_audio),
    ):
        res = client.post(
            "/api/audio/chat",
            files={"file": ("voice.wav", fake_wav, "audio/wav")},
            data={"use_rag": "false", "generate_audio": "true"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["transcription"] == "What is the weather?"
    assert body["audio_url"] == "/api/media/audio/test_tts_output.wav"
    assert client.get(body["audio_url"]).status_code == 200
    out_audio.unlink(missing_ok=True)


def test_document_rag_upload_and_followup(client: TestClient) -> None:
    txt_content = (
        b"Project Orion uses FastAPI and MySQL for persistence.\n"
        b"The primary embedding model is all-MiniLM-L6-v2."
    )
    fake_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    with patch("backend.services.rag_service.rag_service.embed_texts", return_value=fake_vec):
        up_res = client.post(
            "/api/documents/upload",
            files={"file": ("orion_spec.txt", txt_content, "text/plain")},
        )
    assert up_res.status_code == 201
    doc_id = up_res.json()["document_id"]
    assert up_res.json()["chunk_count"] >= 1

    with (
        patch("backend.services.rag_service.rag_service.embed_texts", return_value=fake_vec),
        patch("backend.routes.chat.llm_service.generate_reply", return_value="Project Orion uses FastAPI and MySQL [orion_spec.txt, chunk #0]."),
    ):
        chat_res = client.post(
            "/api/chat",
            json={"message": "What database does Project Orion use?", "use_rag": True, "document_ids": [doc_id]},
        )
    assert chat_res.status_code == 200
    body = chat_res.json()
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["filename"] == "orion_spec.txt"

    # Cleanup document
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 204


def test_invalid_files_and_guardrails(client: TestClient) -> None:
    # 1. Invalid extension
    bad_ext = client.post(
        "/api/documents/upload",
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
    )
    assert bad_ext.status_code == 400

    # 2. Fake PDF header
    bad_pdf = client.post(
        "/api/documents/upload",
        files={"file": ("fake.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert bad_pdf.status_code == 400

    # 3. Corrupted image
    bad_img = client.post(
        "/api/image/analyze",
        files={"file": ("broken.png", b"corrupt-bytes", "image/png")},
        data={"prompt": "Describe"},
    )
    assert bad_img.status_code == 400

    # 4. Crisis self-harm guardrail
    crisis = client.post(
        "/api/chat",
        json={"message": "I feel like I want to hurt myself", "use_rag": False},
    )
    assert crisis.status_code == 200
    assert crisis.json()["safety_flag"] == "crisis"
    assert "988" in crisis.json()["reply"]

    # 5. Model unavailable returns clean 503 without secrets
    with patch("backend.routes.chat.llm_service.generate_reply") as mock_llm:
        from fastapi import HTTPException

        mock_llm.side_effect = HTTPException(status_code=503, detail="AI model unavailable.")
        err_res = client.post("/api/chat", json={"message": "Hello", "use_rag": False})
    assert err_res.status_code == 503
    assert "HF_TOKEN" not in err_res.text
