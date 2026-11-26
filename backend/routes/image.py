"""Image upload and Vision-Language analysis routes."""

import json

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import Message
from backend.database.schemas import ImageAnalysisResponse
from backend.routes.chat import get_or_create_conversation
from backend.security.guardrails import (
    ALLOWED_IMAGE_EXTENSIONS,
    read_validated_upload,
    sanitize_user_text,
    verify_internal_api_key,
)
from backend.services.speech_service import speech_service
from backend.services.vision_service import vision_service

router = APIRouter(prefix="/api/image", tags=["Vision"], dependencies=[Depends(verify_internal_api_key)])


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="Please describe this image in detail."),
    conversation_id: int | None = Form(default=None),
    generate_audio: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    """Upload an image (JPG/JPEG/PNG/WEBP), analyze it with the VLM, and store messages."""
    settings = get_settings()
    settings.ensure_storage_dirs()

    clean_prompt = sanitize_user_text(prompt)
    data, safe_name, _ = await read_validated_upload(
        upload=file,
        allowed_extensions=ALLOWED_IMAGE_EXTENSIONS,
        max_size_mb=settings.max_image_size_mb,
        kind="image",
    )

    saved_path = settings.images_dir / safe_name
    saved_path.write_bytes(data)
    image_url = f"/api/media/images/{safe_name}"

    conv = get_or_create_conversation(db, conversation_id, f"Image: {clean_prompt}")
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=clean_prompt,
        modality="image",
        attachment_path=image_url,
        metadata_json=json.dumps({"original_filename": file.filename}),
    )
    db.add(user_msg)
    db.flush()

    reply_text = vision_service.analyze_image(saved_path, clean_prompt)

    audio_url: str | None = None
    if generate_audio:
        audio_path = speech_service.synthesize_speech(reply_text)
        if audio_path is not None:
            audio_url = f"/api/media/audio/{audio_path.name}"

    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=reply_text,
        modality="voice" if audio_url else "text",
        attachment_path=audio_url,
        metadata_json=json.dumps({"image_url": image_url, "audio_url": audio_url}),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return ImageAnalysisResponse(
        conversation_id=conv.id,
        user_message_id=user_msg.id,
        assistant_message_id=assistant_msg.id,
        reply=reply_text,
        image_url=image_url,
        audio_url=audio_url,
    )
