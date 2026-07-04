"""Tests for FastAPI HTTP endpoints — auth, dashboard, settings, Twilio.

Requires httpx AsyncClient with ASGITransport.  Uses the app's full
middleware stack (session auth) but patches the lifespan to avoid
real database initialisation — instead we configure the session
factory manually.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select


# ---------------------------------------------------------------------------
# Helpers: patch lifespan so we control the DB session factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(_app):
    """No-op lifespan that does nothing — we manage the DB ourselves."""
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
def _patch_lifespan():
    """Patch main.lifespan once per module so ASGITransport doesn't
    auto-create a real DB engine.

    This must run before any test imports main, so it is session-scoped
    and applied at module load time via the fixture below.
    """
    import main as _main

    _main.lifespan = _noop_lifespan
    yield


@pytest_asyncio.fixture(autouse=True)
def _apply_patch(_patch_lifespan):
    """Ensure lifespan patch is applied before any test runs.

    This autouse fixture depends on _patch_lifespan so the module-level
    patch is guaranteed.
    """
    yield


@pytest_asyncio.fixture
async def api_db():
    """Create an in-memory SQLite engine, create tables, configure the
    app's session factory, and return the factory for use in tests.

    IMPORTANT: We do NOT re-import main here; the patch is already in
    place from _patch_lifespan.
    """
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from voice_agent.database import configure_session_factory
    from voice_agent.models import Base

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    configure_session_factory(factory)

    yield factory

    await engine.dispose()


@pytest_asyncio.fixture
async def client(api_db):
    """Create an HTTP test client with no-op lifespan and pre-configured DB."""
    import main as _main

    transport = ASGITransport(app=_main.app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        yield ac


# We use a helper to register a business via the API so the session is
# properly set up by the middleware.  This fixture returns a client that
# already has a valid session cookie.
_REGISTERED_EMAIL = "apitest@example.com"
_REGISTERED_PASSWORD = "password123"
_REGISTERED_PHONE = "+15551230001"
_REGISTERED_BUSINESS = "API Test Co"


@pytest_asyncio.fixture
async def auth_client(client):
    """Return a client with an authenticated session (already registered)."""
    resp = await client.post(
        "/register",
        data={
            "business_name": _REGISTERED_BUSINESS,
            "email": _REGISTERED_EMAIL,
            "password": _REGISTERED_PASSWORD,
            "phone_number": _REGISTERED_PHONE,
        },
    )
    # Response is 303 redirect to /dashboard; session cookie is stored
    # automatically by httpx's cookie jar.
    assert resp.status_code == 303
    return client


# ===================================================================
# Health
# ===================================================================


class TestHealth:
    """Health-check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, client):
        """GET /health returns {'status': 'ok'}."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}


# ===================================================================
# Registration
# ===================================================================


class TestRegistration:
    """Registration flow."""

    @pytest.mark.asyncio
    async def test_register_page_loads(self, client):
        """GET /register returns 200 with HTML."""
        resp = await client.get("/register")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_register_new_business_creates_and_redirects(self, client):
        """POST /register creates a Business+User and redirects to /dashboard."""
        resp = await client.post(
            "/register",
            data={
                "business_name": "Fresh Biz",
                "email": "fresh@example.com",
                "password": "securepass",
                "phone_number": "+15551230002",
            },
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/dashboard"

    @pytest.mark.asyncio
    async def test_register_duplicate_email_returns_409(self, client):
        """POST /register with existing email returns 409 error."""
        # First registration
        await client.post(
            "/register",
            data={
                "business_name": "First",
                "email": "dupe@example.com",
                "password": "password123",
                "phone_number": "+15551230003",
            },
        )
        # Second with same email
        resp = await client.post(
            "/register",
            data={
                "business_name": "Second",
                "email": "dupe@example.com",
                "password": "password456",
                "phone_number": "+15551230004",
            },
        )
        assert resp.status_code == 409
        assert "already exists" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_register_short_password_returns_400(self, client):
        """POST /register with password < 8 chars returns 400."""
        resp = await client.post(
            "/register",
            data={
                "business_name": "WeakPass",
                "email": "weak@example.com",
                "password": "short",
                "phone_number": "+15551230005",
            },
        )
        assert resp.status_code == 400
        assert "at least 8 characters" in resp.text.lower()


# ===================================================================
# Login
# ===================================================================


class TestLogin:
    """Login flow."""

    @pytest.mark.asyncio
    async def test_login_page_loads(self, client):
        """GET /login returns 200 with HTML."""
        resp = await client.get("/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_login_valid_credentials_redirects(self, client):
        """POST /login with valid credentials redirects to /dashboard."""
        # First register
        await client.post(
            "/register",
            data={
                "business_name": "LoginTest",
                "email": "login@example.com",
                "password": "password123",
                "phone_number": "+15551230006",
            },
        )

        # Now login
        resp = await client.post(
            "/login",
            data={
                "email": "login@example.com",
                "password": "password123",
            },
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/dashboard"

    @pytest.mark.asyncio
    async def test_login_invalid_password_returns_401(self, client):
        """POST /login with wrong password returns 401."""
        await client.post(
            "/register",
            data={
                "business_name": "BadPass",
                "email": "badpass@example.com",
                "password": "password123",
                "phone_number": "+15551230007",
            },
        )

        resp = await client.post(
            "/login",
            data={
                "email": "badpass@example.com",
                "password": "wrongpassword",
            },
        )
        assert resp.status_code == 401
        assert "invalid" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_login_unknown_email_returns_401(self, client):
        """POST /login with unknown email returns 401."""
        resp = await client.post(
            "/login",
            data={
                "email": "nobody@example.com",
                "password": "password123",
            },
        )
        assert resp.status_code == 401
        assert "invalid" in resp.text.lower()


# ===================================================================
# Dashboard  (authenticated)
# ===================================================================


class TestDashboard:
    """Dashboard authenticated routes."""

    @pytest.mark.asyncio
    async def test_dashboard_redirects_to_login_when_unauthenticated(self, client):
        """GET /dashboard without session redirects to /login."""
        resp = await client.get("/dashboard", follow_redirects=False)
        assert resp.status_code in (303, 307)
        location = resp.headers.get("location", "")
        assert "/login" in location

    @pytest.mark.asyncio
    async def test_dashboard_after_login(self, auth_client):
        """GET /dashboard with authenticated session returns 200."""
        resp = await auth_client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ===================================================================
# FAQ CRUD  (authenticated)
# ===================================================================


class TestFAQCRUD:
    """FAQ management routes."""

    @pytest.mark.asyncio
    async def test_faq_add_creates_entry(self, auth_client):
        """POST /dashboard/faqs/add creates FAQ and redirects."""
        resp = await auth_client.post(
            "/dashboard/faqs/add",
            data={
                "question": "What is your return policy?",
                "answer": "30-day return policy.",
                "category": "policy",
                "keywords": "return, refund",
            },
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/dashboard/faqs"

    @pytest.mark.asyncio
    async def test_faq_edit_updates_entry(self, auth_client):
        """POST /dashboard/faqs/{id}/edit updates a FAQ."""
        # First create a FAQ
        await auth_client.post(
            "/dashboard/faqs/add",
            data={
                "question": "Old question?",
                "answer": "Old answer.",
            },
        )
        # Fetch the FAQ page to get the ID. Since we can't parse HTML easily
        # we rely on the redirect having created the FAQ.  We verify by
        # checking the DB directly.
        from voice_agent.database import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            from voice_agent.models import FAQ

            result = await session.execute(
                select(FAQ).where(FAQ.question == "Old question?")
            )
            faq = result.scalar_one_or_none()
            assert faq is not None, "FAQ should have been created"

            faq_id = faq.id

        # Edit the FAQ
        resp = await auth_client.post(
            f"/dashboard/faqs/{faq_id}/edit",
            data={
                "question": "Updated question?",
                "answer": "Updated answer.",
            },
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/dashboard/faqs"

        # Verify update in DB
        async with factory() as session:
            result = await session.execute(
                select(FAQ).where(FAQ.id == faq_id)
            )
            updated = result.scalar_one_or_none()
            assert updated is not None
            assert updated.question == "Updated question?"
            assert updated.answer == "Updated answer."

    @pytest.mark.asyncio
    async def test_faq_delete_removes_entry(self, auth_client):
        """POST /dashboard/faqs/{id}/delete removes the FAQ."""
        # Create a FAQ to delete
        await auth_client.post(
            "/dashboard/faqs/add",
            data={
                "question": "Delete me?",
                "answer": "Gone soon.",
            },
        )
        from voice_agent.database import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            from voice_agent.models import FAQ

            result = await session.execute(
                select(FAQ).where(FAQ.question == "Delete me?")
            )
            faq = result.scalar_one_or_none()
            assert faq is not None
            faq_id = faq.id

        # Delete
        resp = await auth_client.post(
            f"/dashboard/faqs/{faq_id}/delete"
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/dashboard/faqs"

        # Verify deletion
        async with factory() as session:
            result = await session.execute(
                select(FAQ).where(FAQ.id == faq_id)
            )
            assert result.scalar_one_or_none() is None


# ===================================================================
# Calls page  (authenticated)
# ===================================================================


class TestCalls:
    """Call history page."""

    @pytest.mark.asyncio
    async def test_calls_page_loads(self, auth_client):
        """GET /dashboard/calls returns 200."""
        resp = await auth_client.get("/dashboard/calls")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ===================================================================
# Logout
# ===================================================================


class TestLogout:
    """Logout flow."""

    @pytest.mark.asyncio
    async def test_logout_clears_session(self, client):
        """GET /logout clears session and redirects to /login."""
        resp = await client.get("/logout")
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/login"


# ===================================================================
# Twilio incoming
# ===================================================================


class TestTwilio:
    """Twilio incoming-call webhook."""

    @pytest.mark.asyncio
    async def test_twilio_incoming_returns_twiml(self, client):
        """POST /twilio/incoming with form data returns XML TwiML."""
        resp = await client.post(
            "/twilio/incoming",
            data={
                "To": "+12025551234",
                "From": "+15551230999",
                "CallSid": "CAdeadbeef",
            },
        )
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "xml" in content_type
        body = resp.text
        assert "<Response>" in body
        # The business +12025551234 matches test_business fixture if it exists.
        # But since lifespan is noop and we didn't pre-populate, it will be unknown.
        # So we check for the "not configured" or "goodbye" message.
        assert "not configured" in body.lower() or "goodbye" in body

    @pytest.mark.asyncio
    async def test_twilio_incoming_unknown_number_returns_goodbye(self, client):
        """POST /twilio/incoming with unknown number returns 'not configured' TwiML."""
        resp = await client.post(
            "/twilio/incoming",
            data={
                "To": "+99999999999",
                "From": "+15551230999",
                "CallSid": "CAdeadbeef",
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "not configured" in body.lower() or "goodbye" in body

    @pytest.mark.asyncio
    async def test_twilio_incoming_known_number_returns_connect(self, client):
        """POST /twilio/incoming with known business number returns <Connect><Stream>."""
        # Pre-populate a business matching the "To" number
        from voice_agent.database import get_session_factory
        from voice_agent.models import Business

        factory = get_session_factory()
        async with factory() as session:
            biz = Business(
                name="TwilioBiz",
                slug="twilio-biz",
                phone_number="+12025551234",
            )
            session.add(biz)
            await session.commit()

        resp = await client.post(
            "/twilio/incoming",
            data={
                "To": "+12025551234",
                "From": "+15551230999",
                "CallSid": "CAdeadbeef",
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "<Connect>" in body
        assert "<Stream" in body


# ===================================================================
# Settings  (authenticated)
# ===================================================================


class TestSettings:
    """Settings page and save."""

    @pytest.mark.asyncio
    async def test_settings_page_loads(self, auth_client):
        """GET /dashboard/settings returns 200."""
        resp = await auth_client.get("/dashboard/settings")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_settings_save_updates_business(self, auth_client):
        """POST /dashboard/settings saves changes and redirects."""
        resp = await auth_client.post(
            "/dashboard/settings",
            data={
                "name": "Updated Biz Name",
                "greeting_message": "Welcome to Updated Biz!",
                "business_hours": '{"monday": {"open": "08:00", "close": "18:00"}}',
                "timezone": "America/Chicago",
                "transfer_phone_number": "+15551230999",
            },
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/dashboard/settings"

        # Verify in DB
        from voice_agent.database import get_session_factory
        from voice_agent.models import Business

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Business).where(Business.slug == "api-test-co")
            )
            biz = result.scalar_one_or_none()
            assert biz is not None
            assert biz.name == "Updated Biz Name"
            assert biz.greeting_message == "Welcome to Updated Biz!"
            assert biz.business_hours == {
                "monday": {"open": "08:00", "close": "18:00"}
            }
            assert biz.timezone == "America/Chicago"
            assert biz.transfer_phone_number == "+15551230999"
