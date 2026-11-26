"""Conversation management and text chat API routes."""

import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import Conversation, Message
from backend.database.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    SourceCitation,
)
from backend.security.guardrails import (
    CRISIS_SUPPORT_RESPONSE,
    HARMFUL_REFUSAL_RESPONSE,
    inspect_user_safety,
    sanitize_user_text,
    verify_internal_api_key,
)
from backend.services.llm_service import llm_service
from backend.services.rag_service import rag_service
from backend.services.speech_service import speech_service

router = APIRouter(prefix="/api", tags=["Chat & Conversations"], dependencies=[Depends(verify_internal_api_key)])


def get_or_create_conversation(db: Session, conversation_id: int | None, first_message: str) -> Conversation:
    """Fetch an existing conversation or create a new one titled from the first message."""
    if conversation_id is not None:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )
        return conv

    title = first_message[:60].strip() or "New Conversation"
    if len(first_message) > 60:
        title += "..."
    conv = Conversation(title=title)
    db.add(conv)
    db.flush()
    return conv


def load_history(db: Session, conversation_id: int, limit: int) -> list[dict[str, str]]:
    """Load recent messages for conversational context."""
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    return [{"role": row.role, "content": row.content} for row in rows]


@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(db: Session = Depends(get_db)):
    """List all saved conversations ordered by most recently updated."""
    return (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .all()
    )


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)):
    """Create a new empty conversation."""
    title = (payload.title or "New Conversation").strip() or "New Conversation"
    conv = Conversation(title=title[:255])
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Retrieve a conversation and all of its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conv


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete a conversation and its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    db.delete(conv)
    db.commit()


@router.delete("/conversations/{conversation_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def clear_conversation_messages(conversation_id: int, db: Session = Depends(get_db)):
    """Clear all messages inside an existing conversation while keeping the conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()
    db.commit()


@router.post("/chat", response_model=ChatResponse)
def chat_text(request: Request, payload: ChatRequest, db: Session = Depends(get_db)):
    """Process a text chat message with context history, RAG retrieval, safety checks, and optional TTS."""
    settings = get_settings()
    clean_message = sanitize_user_text(payload.message)
    safety_flag, emotional_support = inspect_user_safety(clean_message)

    conv = get_or_create_conversation(db, payload.conversation_id, clean_message)
    history = load_history(db, conv.id, settings.max_history_messages)

    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=clean_message,
        modality="text",
    )
    db.add(user_msg)
    db.flush()

    sources: list[SourceCitation] = []
    if safety_flag == "crisis":
        reply_text = CRISIS_SUPPORT_RESPONSE
    elif safety_flag == "harmful":
        reply_text = HARMFUL_REFUSAL_RESPONSE
    else:
        if payload.use_rag:
            try:
                sources = rag_service.retrieve_relevant_chunks(
                    db=db,
                    query=clean_message,
                    conversation_id=conv.id,
                    document_ids=payload.document_ids,
                )
            except HTTPException:
                # If no embeddings or embedding service is temporarily unreachable without docs,
                # allow regular chat only when no documents were explicitly requested.
                if payload.document_ids:
                    raise
                sources = []

        reply_text = llm_service.generate_reply(
            user_message=clean_message,
            history=history,
            rag_sources=sources,
            emotional_support=emotional_support,
        )

    audio_url: str | None = None
    if payload.generate_audio:
        audio_path = speech_service.synthesize_speech(reply_text)
        if audio_path is not None:
            audio_url = f"/api/media/audio/{audio_path.name}"

    meta = {
        "sources": [s.model_dump() for s in sources],
        "audio_url": audio_url,
        "safety_flag": safety_flag,
    }
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=reply_text,
        modality="voice" if audio_url else "text",
        attachment_path=audio_url,
        metadata_json=json.dumps(meta),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return ChatResponse(
        conversation_id=conv.id,
        user_message_id=user_msg.id,
        assistant_message_id=assistant_msg.id,
        reply=reply_text,
        modality=assistant_msg.modality,
        sources=sources,
        audio_url=audio_url,
        safety_flag=safety_flag,
        emotional_support_active=emotional_support,
    )
