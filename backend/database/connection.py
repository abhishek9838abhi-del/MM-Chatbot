"""MySQL database engine, session factory, and health helpers."""

from collections.abc import Generator
import logging

from fastapi import HTTPException, status
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings
from backend.database.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    """Lazily create and return the SQLAlchemy engine with graceful local SQLite fallback if MySQL is offline."""
    global _engine
    if _engine is None:
        settings = get_settings()
        # Attempt primary database connection (MySQL)
        try:
            temp_engine = create_engine(
                settings.resolved_database_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5,
                max_overflow=10,
                echo=settings.debug,
            )
            with temp_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            _engine = temp_engine
            logger.info("Connected to MySQL database at %s:%s", settings.mysql_host, settings.mysql_port)
        except Exception as exc:
            logger.warning(
                "Could not connect to MySQL (%s). Falling back to persistent local SQLite database.",
                exc.__class__.__name__,
            )
            settings.ensure_storage_dirs()
            _engine = create_engine(
                settings.sqlite_fallback_url,
                connect_args={"check_same_thread": False},
                echo=settings.debug,
            )
    return _engine


def get_session_factory():
    """Return configured sessionmaker bound to the active engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def init_db() -> bool:
    """Create database tables if they do not exist. Returns True on success."""
    try:
        Base.metadata.create_all(bind=get_engine())
        return True
    except SQLAlchemyError as exc:
        logger.warning("Database initialization skipped/failed: %s", exc.__class__.__name__)
        return False


def check_db_health() -> bool:
    """Verify database connectivity with a lightweight SELECT 1 query."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session with safe error handling."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("Database operation failed: %s", exc.__class__.__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is currently unavailable. Please verify MySQL is running.",
        ) from exc
    finally:
        db.close()
