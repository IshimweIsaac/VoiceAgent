"""Shared fixtures for all VoiceAgent tests."""

from __future__ import annotations

import asyncio
import os
import sys

import bcrypt
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables BEFORE any imports
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["TWILIO_ACCOUNT_SID"] = "test-twilio-sid"
os.environ["TWILIO_AUTH_TOKEN"] = "test-twilio-auth"
os.environ["SECRET_KEY"] = "test-secret-key-1234567890123456789012345678901234567890"
os.environ["ENCRYPTION_KEY"] = "dGVzdC1lbmNyeXB0aW9uLWtleS0zMi1ieXRlcy1iYXNlNjQ="
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["GOOGLE_CLIENT_ID"] = "test-client-id"
os.environ["GOOGLE_CLIENT_SECRET"] = "test-client-secret"
os.environ["GOOGLE_REDIRECT_URI"] = "http://test/callback"
os.environ["LOG_LEVEL"] = "CRITICAL"
os.environ["DEBUG"] = "false"


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session():
    """Create SQLite in-memory database with all tables."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from voice_agent.models import Base

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def test_business(db_session):
    """Create a test Business record."""
    from voice_agent.models import Business

    biz = Business(
        name="Test Business",
        slug="test-business",
        phone_number="+12025551234",
        greeting_message="Hello, you've reached {business_name}. How can I help you today?",
        timezone="America/New_York",
        business_hours={
            "monday": {"open": "09:00", "close": "17:00"},
            "tuesday": {"open": "09:00", "close": "17:00"},
            "wednesday": {"open": "09:00", "close": "17:00"},
            "thursday": {"open": "09:00", "close": "17:00"},
            "friday": {"open": "09:00", "close": "17:00"},
        },
        transfer_phone_number="+12025559876",
    )
    db_session.add(biz)
    await db_session.flush()
    await db_session.refresh(biz)
    return biz


@pytest_asyncio.fixture
async def test_user(db_session, test_business):
    """Create a test User linked to test_business."""
    from voice_agent.models import User

    user = User(
        business_id=test_business.id,
        email="admin@test.com",
        password_hash=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode(),
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def sample_faqs(db_session, test_business):
    """Create FAQ entries for test_business."""
    from voice_agent.models import FAQ

    faqs = []
    entries = [
        ("What are your hours?", "We are open Mon-Fri 9-5.", "hours", "hours, open, close"),
        ("What services do you offer?", "We offer plumbing and HVAC.", "services", "plumbing, hvac, services"),
        ("How much does it cost?", "Our rates start at $100/hour.", "pricing", "cost, price, rate, pricing"),
    ]
    for q, a, cat, kw in entries:
        faq = FAQ(
            business_id=test_business.id,
            question=q,
            answer=a,
            category=cat,
            keywords=kw,
        )
        db_session.add(faq)
    await db_session.flush()
    for faq in faqs:
        await db_session.refresh(faq)
    return faqs


@pytest.fixture
def mock_gemini():
    """Mock GeminiCallHandler.connect() to skip real API calls."""
    with patch("voice_agent.gemini_client.GeminiCallHandler.connect") as mock_connect:
        mock_connect.return_value = None
        yield mock_connect


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Reset tool registry between tests to avoid cross-test contamination."""
    import voice_agent.tool_registry as tr

    saved = dict(tr._TOOL_MAP)
    tr._TOOL_MAP.clear()
    yield
    tr._TOOL_MAP.clear()
    tr._TOOL_MAP.update(saved)


@pytest.fixture
def sample_pcm_data():
    """Generate 0.1 seconds of 8kHz PCM s16le data (800 samples)."""
    t = np.linspace(0, 0.1, 800, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    return samples.tobytes()
