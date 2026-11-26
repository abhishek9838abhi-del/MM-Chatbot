"""Pydantic request and response schemas for the REST API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    title: str | None = Field(default="New Conversation", max_length=255)


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    modality: str
    attachment_path: str | None = None
    metadata_json: str | None = None
    created_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse] = []


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str = Field(..., min_length=1, max_length=8000)
    use_rag: bool = True
    document_ids: list[int] | None = None
    generate_audio: bool = False


class SourceCitation(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    conversation_id: int
    user_message_id: int
    assistant_message_id: int
    reply: str
    modality: str = "text"
    sources: list[SourceCitation] = []
    audio_url: str | None = None
    safety_flag: str | None = None
    emotional_support_active: bool = False


class ImageAnalysisResponse(BaseModel):
    conversation_id: int
    user_message_id: int
    assistant_message_id: int
    reply: str
    image_url: str
    audio_url: str | None = None


class AudioChatResponse(BaseModel):
    conversation_id: int
    user_message_id: int
    assistant_message_id: int
    transcription: str
    reply: str
    audio_url: str | None = None
    sources: list[SourceCitation] = []
    safety_flag: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    file_type: str
    chunk_count: int
    conversation_id: int | None = None
    message: str


class DocumentInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int | None = None
    filename: str
    file_type: str
    created_at: datetime
    chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str
    database: str
    ai_provider: str
    api_key_configured: bool
    models: dict[str, Any]
