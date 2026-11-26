"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed runtime settings for backend, AI models, database, and guardrails."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Multimodal AI Assistant"
    app_env: str = "development"
    debug: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    backend_url: str = "http://127.0.0.1:8000"
    internal_api_key: str = ""

    # MySQL Database
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "multimodal_chatbot"
    database_url: str = ""

    # AI Provider selection: "gemini" (default, free tier available) or "huggingface"
    ai_provider: str = "gemini"

    # Google Gemini Configuration (free API key from aistudio.google.com)
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_vision_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Hugging Face Models
    hf_token: str = ""
    hf_inference_mode: str = "api"
    hf_timeout_seconds: int = 90
    hf_chat_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_vision_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    hf_asr_model: str = "openai/whisper-large-v3-turbo"
    hf_tts_model: str = "facebook/mms-tts-eng"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tts_fallback_engine: str = "pyttsx3"

    # RAG Configuration
    rag_chunk_size: int = Field(default=800, ge=200, le=4000)
    rag_chunk_overlap: int = Field(default=150, ge=0, le=1000)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_min_similarity: float = Field(default=0.15, ge=0.0, le=1.0)

    # Security & Limits
    max_text_length: int = Field(default=4000, ge=100, le=32000)
    max_image_size_mb: int = Field(default=10, ge=1, le=50)
    max_audio_size_mb: int = Field(default=25, ge=1, le=100)
    max_document_size_mb: int = Field(default=20, ge=1, le=100)
    max_history_messages: int = Field(default=12, ge=2, le=50)
    rate_limit_per_minute: int = Field(default=30, ge=5, le=300)

    @computed_field
    @property
    def storage_dir(self) -> Path:
        return BASE_DIR / "storage"

    @computed_field
    @property
    def documents_dir(self) -> Path:
        return self.storage_dir / "documents"

    @computed_field
    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

    @computed_field
    @property
    def audio_dir(self) -> Path:
        return self.storage_dir / "audio"

    @computed_field
    @property
    def vectors_dir(self) -> Path:
        return self.storage_dir / "vectors"

    @computed_field
    @property
    def resolved_database_url(self) -> str:
        """Return explicit DATABASE_URL or construct a safe PyMySQL connection URI, falling back to local SQLite if MySQL host is unreachable."""
        if self.database_url.strip():
            return self.database_url.strip()
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        auth = f"{user}:{password}" if self.mysql_password else user
        return (
            f"mysql+pymysql://{auth}@{self.mysql_host}:{self.mysql_port}"
            f"/{self.mysql_database}?charset=utf8mb4"
        )

    @computed_field
    @property
    def sqlite_fallback_url(self) -> str:
        """Fallback local SQLite database path if MySQL server is offline."""
        sqlite_file = self.storage_dir / "chatbot.db"
        return f"sqlite:///{sqlite_file.as_posix()}"

    def ensure_storage_dirs(self) -> None:
        """Ensure all local file storage directories exist."""
        for directory in (
            self.documents_dir,
            self.images_dir,
            self.audio_dir,
            self.vectors_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings
