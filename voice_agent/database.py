"""Async SQLAlchemy engine, session factory, and FastAPI dependency.

Usage:
  1. Call :func:`init_db` at startup to create tables.
  2. Call :func:`configure_session_factory` with the engine once.
  3. Use :func:`get_db` as a FastAPI dependency for request-scoped sessions.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from voice_agent.config import Settings

logger = logging.getLogger(__name__)

# Module-level session factory (set by configure_session_factory)
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine(settings: Settings) -> Any:
    """Create an async SQLAlchemy engine from the application settings."""
    return create_async_engine(
        settings.database_url,
        echo=settings.debug,
        connect_args={"check_same_thread": False},
    )


def create_session_factory(engine: Any) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


def configure_session_factory(factory: async_sessionmaker[AsyncSession]) -> None:
    """Set the module-level session factory for :func:`get_db`."""
    global _async_session_factory  # noqa: PLW0603
    _async_session_factory = factory


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the configured session factory.

    Raises:
        RuntimeError: If :func:`configure_session_factory` was not called.
    """
    if _async_session_factory is None:
        raise RuntimeError(
            "Database not initialized. Call configure_session_factory first."
        )
    return _async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    The session is committed on success and rolled back on exception.

    Raises:
        RuntimeError: If :func:`configure_session_factory` was not called.
    """
    if _async_session_factory is None:
        raise RuntimeError(
            "Database not initialized. Call configure_session_factory first."
        )

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(settings: Settings) -> None:
    """Create all database tables.

    Uses the ORM metadata from voice_agent.models. Safe to call
    repeatedly — SQLAlchemy skips existing tables.
    """
    from voice_agent.models import Base

    engine = create_engine(settings)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as exc:
        logger.error("Failed to create database tables: %s", exc)
        raise
    finally:
        await engine.dispose()
