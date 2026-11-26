"""Phase 1 setup verification tests."""

from pathlib import Path

from backend.config import BASE_DIR, Settings, get_settings


def test_project_structure_exists() -> None:
    expected_dirs = [
        "frontend",
        "frontend/components",
        "frontend/styles",
        "frontend/utils",
        "backend",
        "backend/routes",
        "backend/services",
        "backend/database",
        "backend/security",
        "storage/documents",
        "storage/images",
        "storage/audio",
        "storage/vectors",
        "tests",
    ]
    for rel_path in expected_dirs:
        assert (BASE_DIR / rel_path).is_dir(), f"Missing directory: {rel_path}"


def test_config_files_exist() -> None:
    for filename in (".env.example", ".env", ".gitignore", "requirements.txt"):
        assert (BASE_DIR / filename).is_file(), f"Missing file: {filename}"


def test_settings_load_and_storage_dirs() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.hf_chat_model
    assert settings.hf_vision_model
    assert settings.hf_asr_model
    assert settings.hf_embedding_model
    assert settings.resolved_database_url.startswith("mysql+pymysql://")
    assert Path(settings.documents_dir).is_dir()
    assert Path(settings.images_dir).is_dir()
    assert Path(settings.audio_dir).is_dir()
    assert Path(settings.vectors_dir).is_dir()
