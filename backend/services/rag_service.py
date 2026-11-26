"""RAG service: document extraction, chunking, Hugging Face embeddings, and FAISS vector search."""

import logging
from pathlib import Path
import re

from fastapi import HTTPException, status
from huggingface_hub import InferenceClient
import numpy as np
from pypdf import PdfReader
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.models import DocumentChunk, UploadedDocument
from backend.database.schemas import SourceCitation

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class RAGService:
    """Handles PDF/TXT ingestion, chunking, embedding generation, and vector retrieval."""

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

    @staticmethod
    def extract_text(file_path: Path, file_type: str) -> str:
        """Extract and clean text from a PDF or TXT document."""
        if file_type == "pdf":
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            raw_text = "\n\n".join(pages)
        elif file_type == "txt":
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw_text = file_path.read_text(encoding="latin-1")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported document type: {file_type}",
            )

        cleaned = re.sub(r"[ \t]+", " ", raw_text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if not cleaned:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No readable text could be extracted from the uploaded document.",
            )
        return cleaned

    def chunk_text(self, text: str) -> list[str]:
        """Split cleaned text into overlapping chunks preserving word boundaries where possible."""
        chunk_size = self.settings.rag_chunk_size
        overlap = min(self.settings.rag_chunk_overlap, chunk_size // 2)
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                split_at = text.rfind(" ", start + chunk_size // 2, end)
                if split_at != -1:
                    end = split_at
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= text_len:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _embed_gemini(self, texts: list[str]) -> np.ndarray:
        api_key = self.settings.gemini_api_key.strip()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google Gemini API key (GEMINI_API_KEY) is not configured in .env. Get a free key at https://aistudio.google.com",
            )
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            vectors: list[np.ndarray] = []
            for text in texts:
                res = client.models.embed_content(
                    model=self.settings.gemini_embedding_model,
                    contents=text,
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    values = res.embeddings[0].values
                elif hasattr(res, "embedding") and res.embedding:
                    values = res.embedding.values
                else:
                    raise ValueError("Unexpected embedding response structure from Gemini.")
                arr = np.asarray(values, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vectors.append(arr)
            return np.vstack(vectors).astype(np.float32)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Gemini embedding generation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Gemini embedding error: {exc}",
            ) from exc

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate L2-normalized embeddings via Gemini or Hugging Face."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.settings.ai_provider.lower().strip() == "gemini":
            return self._embed_gemini(texts)

        client = self._get_client()
        vectors: list[np.ndarray] = []
        try:
            for text in texts:
                raw = client.feature_extraction(
                    text=text,
                    model=self.settings.hf_embedding_model,
                )
                arr = np.asarray(raw, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                elif arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vectors.append(arr)
            return np.vstack(vectors).astype(np.float32)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("HF embedding generation failed: %s", exc.__class__.__name__)
            err_str = str(exc)
            if "402" in err_str or "Payment Required" in err_str:
                detail = "Hugging Face Inference credits exhausted (402 Payment Required). Please add credits or update HF_TOKEN."
            elif "401" in err_str or "Unauthorized" in err_str:
                detail = "Hugging Face API token is invalid or unauthorized (401)."
            else:
                detail = "Embedding model is currently unavailable. Please try again shortly."
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            ) from exc

    def ingest_document(
        self,
        db: Session,
        filename: str,
        file_type: str,
        file_path: Path,
        conversation_id: int | None = None,
    ) -> tuple[UploadedDocument, int]:
        """Extract text, chunk, embed, save vector matrix, and persist document/chunk rows."""
        extracted = self.extract_text(file_path, file_type)
        chunks = self.chunk_text(extracted)
        embeddings = self.embed_texts(chunks)

        doc = UploadedDocument(
            conversation_id=conversation_id,
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
        )
        db.add(doc)
        db.flush()

        self.settings.ensure_storage_dirs()
        vector_file = self.settings.vectors_dir / f"doc_{doc.id}.npy"
        np.save(str(vector_file), embeddings)

        for idx, chunk_text in enumerate(chunks):
            db.add(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_text=chunk_text,
                    embedding_reference=f"{vector_file.name}#{idx}",
                    chunk_index=idx,
                )
            )
        db.commit()
        db.refresh(doc)
        return doc, len(chunks)

    def retrieve_relevant_chunks(
        self,
        db: Session,
        query: str,
        conversation_id: int | None = None,
        document_ids: list[int] | None = None,
        top_k: int | None = None,
    ) -> list[SourceCitation]:
        """Retrieve the most similar document chunks using FAISS (or NumPy cosine similarity)."""
        doc_query = db.query(UploadedDocument)
        if document_ids:
            doc_query = doc_query.filter(UploadedDocument.id.in_(document_ids))
        elif conversation_id is not None:
            doc_query = doc_query.filter(
                (UploadedDocument.conversation_id == conversation_id)
                | (UploadedDocument.conversation_id.is_(None))
            )

        documents = doc_query.all()
        if not documents:
            return []

        all_vectors: list[np.ndarray] = []
        chunk_records: list[tuple[UploadedDocument, DocumentChunk]] = []

        for doc in documents:
            vector_path = self.settings.vectors_dir / f"doc_{doc.id}.npy"
            if not vector_path.exists():
                continue
            mat = np.load(str(vector_path)).astype(np.float32)
            chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == doc.id)
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )
            limit = min(len(chunks), mat.shape[0])
            for i in range(limit):
                all_vectors.append(mat[i])
                chunk_records.append((doc, chunks[i]))

        if not all_vectors:
            return []

        query_vec = self.embed_texts([query])[0].astype(np.float32)
        matrix = np.vstack(all_vectors).astype(np.float32)
        k = min(top_k or self.settings.rag_top_k, len(chunk_records))

        if faiss is not None:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            scores, indices = index.search(query_vec.reshape(1, -1), k)
            ranked = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            sims = matrix @ query_vec
            top_indices = np.argsort(-sims)[:k].tolist()
            ranked = [(idx, float(sims[idx])) for idx in top_indices]

        citations: list[SourceCitation] = []
        for idx, score in ranked:
            if idx < 0 or score < self.settings.rag_min_similarity:
                continue
            doc, chunk = chunk_records[idx]
            citations.append(
                SourceCitation(
                    document_id=doc.id,
                    filename=doc.filename,
                    chunk_index=chunk.chunk_index,
                    score=round(float(score), 4),
                    excerpt=chunk.chunk_text,
                )
            )
        return citations


rag_service = RAGService()
