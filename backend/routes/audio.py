"""Audio transcription, voice chat, and Text-to-Speech API routes."""

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import Message
from backend.database.schemas import AudioChatResponse, SourceCitation
from backend.routes.chat import get_or_create_conversation, load_history
from backend.security.guardrails import (
    ALLOWED_AUDIO_EXTENSIONS,
    CRISIS_SUPPORT_RESPONSE,
    HARMFUL_REFUSAL_RESPONSE,
    inspect_user_safety,
    read_validated_upload,
    verify_internal_api_key,
)
from backend.services.llm_service import llm_service
from backend.services.rag_service import rag_service
from backend.services.speech_service import speech_service

router = APIRouter(prefix="/api/audio", tags=["Audio & Speech"], dependencies=[Depends(verify_internal_api_key)])


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class TTSResponse(BaseModel):
    audio_url: str


@router.post("/chat", response_model=AudioChatResponse)
async def audio_chat(
    file: UploadFile = File(...),
    conversation_id: int | None = Form(default=None),
    use_rag: bool = Form(default=True),
    generate_audio: bool = Form(default=True),
    db: Session = Depends(get_db),
):
    """Transcribe uploaded/recorded audio, process with LLM/RAG, and optionally return voice reply."""
    settings = get_settings()
    settings.ensure_storage_dirs()

    data, safe_name, _ = await read_validated_upload(
        upload=file,
        allowed_extensions=ALLOWED_AUDIO_EXTENSIONS,
        max_size_mb=settings.max_audio_size_mb,
        kind="audio",
    )

    saved_path = settings.audio_dir / safe_name
    saved_path.write_bytes(data)
    input_audio_url = f"/api/media/audio/{safe_name}"

    transcription = speech_service.transcribe_audio(saved_path)
    safety_flag, emotional_support = inspect_user_safety(transcription)

    conv = get_or_create_conversation(db, conversation_id, transcription)
    history = load_history(db, conv.id, settings.max_history_messages)

    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=transcription,
        modality="audio",
        attachment_path=input_audio_url,
    )
    db.add(user_msg)
    db.flush()

    sources: list[SourceCitation] = []
    if safety_flag == "crisis":
        reply_text = CRISIS_SUPPORT_RESPONSE
    elif safety_flag == "harmful":
        reply_text = HARMFUL_REFUSAL_RESPONSE
    else:
        if use_rag:
            try:
                sources = rag_service.retrieve_relevant_chunks(
                    db=db,
                    query=transcription,
                    conversation_id=conv.id,
                )
            except HTTPException:
                sources = []

        reply_text = llm_service.generate_reply(
            user_message=transcription,
            history=history,
            rag_sources=sources,
            emotional_support=emotional_support,
        )

    reply_audio_url: str | None = None
    if generate_audio:
        out_path = speech_service.synthesize_speech(reply_text)
        if out_path is not None:
            reply_audio_url = f"/api/media/audio/{out_path.name}"

    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=reply_text,
        modality="voice" if reply_audio_url else "text",
        attachment_path=reply_audio_url,
        metadata_json=json.dumps(
            {
                "transcription": transcription,
                "sources": [s.model_dump() for s in sources],
                "audio_url": reply_audio_url,
                "safety_flag": safety_flag,
            }
        ),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return AudioChatResponse(
        conversation_id=conv.id,
        user_message_id=user_msg.id,
        assistant_message_id=assistant_msg.id,
        transcription=transcription,
        reply=reply_text,
        audio_url=reply_audio_url,
        sources=sources,
        safety_flag=safety_flag,
    )


@router.post("/tts", response_model=TTSResponse)
def text_to_speech(payload: TTSRequest):
    """Convert provided text into a playable audio file."""
    out_path = speech_service.synthesize_speech(payload.text)
    if out_path is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio synthesis is currently unavailable.",
        )
    return TTSResponse(audio_url=f"/api/media/audio/{out_path.name}")
