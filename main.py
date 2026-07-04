"""VoiceAgent — AI Voice Receptionist for Small Businesses.

FastAPI application entry point. Loads settings, initializes the
database, and exposes HTTP + WebSocket endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.middleware.sessions import SessionMiddleware

from voice_agent.config import Settings
from voice_agent.database import (
    configure_session_factory,
    create_engine,
    create_session_factory,
    init_db,
)
from voice_agent.twilio_handler import router as twilio_router
from web.auth import router as auth_router
from web.routes import router as dashboard_router

# Import tools to trigger auto-registration in the tool registry
import voice_agent.tools  # noqa: F401

# ---------------------------------------------------------------------------
# Globals (set during lifespan)
# ---------------------------------------------------------------------------
settings: Settings | None = None
engine = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB on startup, cleanup on shutdown."""
    global settings, engine, async_session_factory  # noqa: PLW0603

    settings = Settings()  # type: ignore[call-arg]
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting VoiceAgent server...")

    # Initialize database
    await init_db(settings)

    # Create engine + session factory for runtime use
    engine = create_engine(settings)
    async_session_factory = create_session_factory(engine)
    configure_session_factory(async_session_factory)

    logger.info(
        "VoiceAgent ready on %s:%d",
        settings.host,
        settings.port,
    )

    yield

    # Shutdown
    if engine is not None:
        await engine.dispose()
    logger.info("VoiceAgent shut down")


app = FastAPI(
    title="VoiceAgent",
    description="AI Voice Receptionist for Small Businesses",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware — Session Auth
# ---------------------------------------------------------------------------

# secret_key is loaded in lifespan but we need it here at module level.
# We use a placeholder; it gets replaced during lifespan lifespan.
_session_secret: str | None = None


def _get_secret() -> str:
    """Retrieve the session secret, configuring from Settings if needed."""
    global _session_secret  # noqa: PLW0603
    if _session_secret is None:
        _session_secret = Settings().secret_key  # type: ignore[call-arg]
    return _session_secret


app.add_middleware(SessionMiddleware, secret_key=_get_secret())


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="web/static"), name="static")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(twilio_router)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/dashboard")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", include_in_schema=False)
async def health() -> JSONResponse:
    """Return server health status."""
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Load settings for the port/host
    s = Settings()  # type: ignore[call-arg]
    uvicorn.run(
        "main:app",
        host=s.host,
        port=s.port,
        reload=s.debug,
        log_level=s.log_level.lower(),
    )
