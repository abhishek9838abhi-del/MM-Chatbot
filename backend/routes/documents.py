"""Document upload, listing, and deletion routes for RAG."""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import Conversation, DocumentChunk, UploadedDocument
from backend.database.schemas import DocumentInfo, DocumentUploadResponse
from backend.security.guardrails import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    read_validated_upload,
    verify_internal_api_key,
)
from backend.services.rag_service import rag_service

router = APIRouter(prefix="/api/documents", tags=["Documents & RAG"], dependencies=[Depends(verify_internal_api_key)])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
):
    """Upload a PDF or TXT file, extract and chunk text, generate embeddings, and store vectors."""
    settings = get_settings()
    settings.ensure_storage_dirs()

    if conversation_id is not None:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    data, safe_name, file_ext = await read_validated_upload(
        upload=file,
        allowed_extensions=ALLOWED_DOCUMENT_EXTENSIONS,
        max_size_mb=settings.max_document_size_mb,
        kind="document",
    )

    saved_path = settings.documents_dir / safe_name
    saved_path.write_bytes(data)

    original_filename = Path(file.filename or safe_name).name[:255]
    try:
        doc, chunk_count = rag_service.ingest_document(
            db=db,
            filename=original_filename,
            file_type=file_ext,
            file_path=saved_path,
            conversation_id=conversation_id,
        )
    except Exception:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        chunk_count=chunk_count,
        conversation_id=doc.conversation_id,
        message=f"Successfully indexed '{doc.filename}' into {chunk_count} searchable chunks.",
    )


@router.get("", response_model=list[DocumentInfo])
def list_documents(conversation_id: int | None = None, db: Session = Depends(get_db)):
    """List uploaded documents, optionally filtered by conversation."""
    query = db.query(UploadedDocument)
    if conversation_id is not None:
        query = query.filter(
            (UploadedDocument.conversation_id == conversation_id)
            | (UploadedDocument.conversation_id.is_(None))
        )
    docs = query.order_by(UploadedDocument.created_at.desc(), UploadedDocument.id.desc()).all()
    results: list[DocumentInfo] = []
    for doc in docs:
        chunk_count = (
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
        )
        results.append(
            DocumentInfo(
                id=doc.id,
                conversation_id=doc.conversation_id,
                filename=doc.filename,
                file_type=doc.file_type,
                created_at=doc.created_at,
                chunk_count=chunk_count,
            )
        )
    return results


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document record, its chunks, stored file, and vector index."""
    settings = get_settings()
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    file_path = Path(doc.file_path)
    vector_path = settings.vectors_dir / f"doc_{doc.id}.npy"
    db.delete(doc)
    db.commit()

    file_path.unlink(missing_ok=True)
    vector_path.unlink(missing_ok=True)
