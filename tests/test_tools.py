"""Tests for tool handler functions — FAQ, Appointment, Transfer, SMS.

Each tool auto-registers via register() at import time. Because the
conftest reset_tool_registry fixture clears the map before each test,
we explicitly re-register all tools in the `registered_tools` fixture.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from sqlalchemy import select


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def registered_tools():
    """Manually register all tool handlers so they're available after
    reset_tool_registry clears _TOOL_MAP."""
    from voice_agent.tool_registry import register
    from voice_agent.tools.faq import TOOL_SCHEMA, handler as faq_handler
    from voice_agent.tools.appointment import (
        CHECK_AVAILABILITY_SCHEMA,
        BOOK_APPOINTMENT_SCHEMA,
        handle_check_availability,
        handle_book_appointment,
    )
    from voice_agent.tools.transfer import (
        TOOL_SCHEMA as transfer_schema,
        handler as transfer_handler,
    )
    from voice_agent.tools.sms import (
        TOOL_SCHEMA as sms_schema,
        handler as sms_handler,
    )

    register("lookup_faq", TOOL_SCHEMA, faq_handler)
    register(
        "check_availability",
        CHECK_AVAILABILITY_SCHEMA,
        handle_check_availability,
    )
    register(
        "book_appointment",
        BOOK_APPOINTMENT_SCHEMA,
        handle_book_appointment,
    )
    register("transfer_to_human", transfer_schema, transfer_handler)
    register("send_sms_confirmation", sms_schema, sms_handler)
    yield


@pytest_asyncio.fixture
async def tool_business(db_session):
    """A minimal Business record for tool tests that need one."""
    from voice_agent.models import Business

    biz = Business(
        name="Tool Test Biz",
        slug="tool-test-biz",
        phone_number="+15551111100",
        timezone="America/New_York",
        business_hours={
            "monday": {"open": "09:00", "close": "17:00"},
        },
        transfer_phone_number="+15551111101",
    )
    db_session.add(biz)
    await db_session.flush()
    await db_session.refresh(biz)
    return biz


@pytest_asyncio.fixture
async def tool_faqs(db_session, tool_business):
    """FAQ entries for tool_business."""
    from voice_agent.models import FAQ

    entries = [
        ("What are your hours?", "We are open Mon-Fri 9-5.", "hours", "hours, open, close"),
        ("What services do you offer?", "We offer plumbing and HVAC.", "services", "plumbing, hvac"),
    ]
    faqs = []
    for q, a, cat, kw in entries:
        faq = FAQ(
            business_id=tool_business.id,
            question=q,
            answer=a,
            category=cat,
            keywords=kw,
        )
        db_session.add(faq)
        faqs.append(faq)
    await db_session.flush()
    for faq in faqs:
        await db_session.refresh(faq)
    return faqs


@pytest_asyncio.fixture
def mock_calendar():
    """Mock GoogleCalendarClient with default success responses."""
    mock = AsyncMock()
    mock.check_availability.return_value = [{"start": "09:00", "end": "10:00"}]
    mock.create_event.return_value = "event-id-123"
    return mock


@pytest_asyncio.fixture
def mock_twilio():
    """Mock TwilioClient."""
    mock = AsyncMock()
    mock.send_sms_async.return_value = True
    return mock


@pytest_asyncio.fixture
def mock_call_manager():
    """Mock call manager with state flags."""
    mock = AsyncMock()
    mock.transferred = False
    mock.appointment_booked = False
    return mock


# ===================================================================
# FAQ Tool — lookup_faq
# ===================================================================


class TestFAQTool:
    """lookup_faq handler."""

    @pytest.mark.asyncio
    async def test_faq_lookup_matches_question(
        self, registered_tools, db_session, tool_business, tool_faqs
    ):
        """Search for 'hours' returns the hours FAQ."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "lookup_faq",
            {"query": "hours"},
            business_id=tool_business.id,
            db_session=db_session,
        )
        assert "What are your hours?" in result
        assert "We are open Mon-Fri 9-5" in result

    @pytest.mark.asyncio
    async def test_faq_lookup_no_match_returns_fallback(
        self, registered_tools, db_session, tool_business, tool_faqs
    ):
        """Search for 'nonexistent' returns fallback message."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "lookup_faq",
            {"query": "nonexistent"},
            business_id=tool_business.id,
            db_session=db_session,
        )
        assert "don't have that information" in result.lower()

    @pytest.mark.asyncio
    async def test_faq_lookup_empty_query_returns_prompt(
        self, registered_tools, db_session, tool_business
    ):
        """Empty query returns 'Please provide a question'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "lookup_faq",
            {"query": ""},
            business_id=tool_business.id,
            db_session=db_session,
        )
        assert "provide a question" in result.lower()

    @pytest.mark.asyncio
    async def test_faq_lookup_uses_keywords(
        self, registered_tools, db_session, tool_business, tool_faqs
    ):
        """Search term that only matches keywords still returns result."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "lookup_faq",
            {"query": "open"},
            business_id=tool_business.id,
            db_session=db_session,
        )
        assert "What are your hours?" in result


# ===================================================================
# Appointment Tool — check_availability
# ===================================================================


class TestCheckAvailability:
    """check_availability handler."""

    @pytest.mark.asyncio
    async def test_check_availability_no_date_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """Empty date returns 'Please provide a date'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "check_availability",
            {"date": ""},
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=AsyncMock(),
        )
        assert "provide a date" in result.lower()

    @pytest.mark.asyncio
    async def test_check_availability_no_calendar_returns_not_configured(
        self, registered_tools, db_session, tool_business
    ):
        """None calendar_service returns 'calendar is not yet configured'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "check_availability",
            {"date": "2026-07-10"},
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=None,
        )
        assert "not yet configured" in result.lower()

    @pytest.mark.asyncio
    async def test_check_availability_calendar_error_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """calendar_service raises exception returns 'having trouble'."""
        from voice_agent.tool_registry import dispatch

        mock = AsyncMock()
        mock.check_availability.side_effect = RuntimeError("API error")

        result = await dispatch(
            "check_availability",
            {"date": "2026-07-10"},
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=mock,
        )
        assert "having trouble" in result.lower()

    @pytest.mark.asyncio
    async def test_check_availability_with_slots_returns_formatted(
        self, registered_tools, db_session, tool_business, mock_calendar
    ):
        """calendar_service returns slots, they're formatted correctly."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "check_availability",
            {"date": "2026-07-10"},
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=mock_calendar,
        )
        assert "Available slots" in result
        assert "2026-07-10" in result
        assert "09:00" in result
        assert "10:00" in result


# ===================================================================
# Appointment Tool — book_appointment
# ===================================================================


class TestBookAppointment:
    """book_appointment handler."""

    @pytest.mark.asyncio
    async def test_book_appointment_missing_fields_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """Missing required fields returns 'provide all the details'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "book_appointment",
            {"date": "", "time": "", "customer_name": "", "customer_phone": ""},
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=AsyncMock(),
        )
        assert "provide all the details" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_no_calendar_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """None calendar_service returns 'not yet configured'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "book_appointment",
            {
                "date": "2026-07-10",
                "time": "09:00",
                "customer_name": "John",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=None,
        )
        assert "not yet configured" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_invalid_date_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """Invalid date format returns 'couldn't understand the date'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "book_appointment",
            {
                "date": "not-a-date",
                "time": "09:00",
                "customer_name": "John",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=AsyncMock(),
        )
        assert "couldn't understand the date" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_invalid_time_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """Invalid time format returns 'couldn't understand the time'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "book_appointment",
            {
                "date": "2026-07-10",
                "time": "25:00",
                "customer_name": "John",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=AsyncMock(),
        )
        assert "couldn't understand the time" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_no_available_slots_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """No available slots returns 'no available slots'."""
        from voice_agent.tool_registry import dispatch

        mock = AsyncMock()
        mock.check_availability.return_value = []

        result = await dispatch(
            "book_appointment",
            {
                "date": "2026-07-10",
                "time": "09:00",
                "customer_name": "John",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=mock,
        )
        assert "no available slots" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_slot_not_available_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """Specific time not available returns 'not available'."""
        from voice_agent.tool_registry import dispatch

        mock = AsyncMock()
        # Only slot is 14:00-15:00, requesting 09:00
        mock.check_availability.return_value = [{"start": "14:00", "end": "15:00"}]

        result = await dispatch(
            "book_appointment",
            {
                "date": "2026-07-10",
                "time": "09:00",
                "customer_name": "John",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=mock,
        )
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_calendar_error_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """calendar_service.create_event raises RuntimeError."""
        from voice_agent.tool_registry import dispatch

        mock = AsyncMock()
        mock.check_availability.return_value = [{"start": "09:00", "end": "10:00"}]
        mock.create_event.side_effect = RuntimeError("Calendar disconnected")

        result = await dispatch(
            "book_appointment",
            {
                "date": "2026-07-10",
                "time": "09:00",
                "customer_name": "John",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=mock,
        )
        assert "not fully connected" in result.lower() or "sorry" in result.lower()

    @pytest.mark.asyncio
    async def test_book_appointment_success_returns_confirmation(
        self, registered_tools, db_session, tool_business, mock_calendar
    ):
        """Full successful booking returns confirmation with date, time, name, phone."""
        from voice_agent.tool_registry import dispatch
        from voice_agent.models import Appointment

        result = await dispatch(
            "book_appointment",
            {
                "date": "2026-07-10",
                "time": "09:00",
                "customer_name": "John Doe",
                "customer_phone": "+15551234567",
            },
            business_id=tool_business.id,
            db_session=db_session,
            calendar_service=mock_calendar,
        )

        # Verify confirmation message
        assert "booked" in result.lower()
        assert "2026-07-10" in result
        assert "09:00" in result
        assert "John Doe" in result
        assert "+15551234567" in result

        # Verify appointment was stored in DB
        stmt = select(Appointment).where(Appointment.customer_name == "John Doe")
        appt = (await db_session.execute(stmt)).scalar_one_or_none()
        assert appt is not None
        assert appt.customer_phone == "+15551234567"
        assert appt.google_event_id == "event-id-123"


# ===================================================================
# Transfer Tool — transfer_to_human
# ===================================================================


class TestTransferTool:
    """transfer_to_human handler."""

    @pytest.mark.asyncio
    async def test_transfer_with_number_returns_hold_message(
        self, registered_tools, db_session, tool_business
    ):
        """Business has transfer_phone_number, returns 'Please hold'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "transfer_to_human",
            {"reason": "Customer requested"},
            business_id=tool_business.id,
            db_session=db_session,
        )
        assert "hold" in result.lower()
        assert "transfer" in result.lower()

    @pytest.mark.asyncio
    async def test_transfer_without_number_returns_unavailable(
        self, registered_tools, db_session
    ):
        """No transfer_phone_number returns 'no one available'."""
        from voice_agent.models import Business
        from voice_agent.tool_registry import dispatch

        biz = Business(
            name="NoTransfer",
            slug="no-transfer",
            phone_number="+15551111102",
            transfer_phone_number="",
        )
        db_session.add(biz)
        await db_session.flush()

        result = await dispatch(
            "transfer_to_human",
            {"reason": "Need help"},
            business_id=biz.id,
            db_session=db_session,
        )
        assert "no one available" in result.lower()

    @pytest.mark.asyncio
    async def test_transfer_sets_call_manager_flag(
        self, registered_tools, db_session, tool_business, mock_call_manager
    ):
        """call_manager.transferred is set to True."""
        from voice_agent.tool_registry import dispatch

        assert mock_call_manager.transferred is False

        await dispatch(
            "transfer_to_human",
            {"reason": "Help"},
            business_id=tool_business.id,
            db_session=db_session,
            call_manager=mock_call_manager,
        )
        assert mock_call_manager.transferred is True


# ===================================================================
# SMS Tool — send_sms_confirmation
# ===================================================================


class TestSMSTool:
    """send_sms_confirmation handler."""

    @pytest.mark.asyncio
    async def test_sms_missing_phone_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """No phone parameter returns 'need a phone number'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "send_sms_confirmation",
            {"phone": "", "message": "Hello"},
            business_id=tool_business.id,
            db_session=db_session,
            twilio_client=AsyncMock(),
        )
        assert "need a phone number" in result.lower()

    @pytest.mark.asyncio
    async def test_sms_missing_message_returns_error(
        self, registered_tools, db_session, tool_business
    ):
        """No message parameter returns 'need a message'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "send_sms_confirmation",
            {"phone": "+15551234567", "message": ""},
            business_id=tool_business.id,
            db_session=db_session,
            twilio_client=AsyncMock(),
        )
        assert "need a message" in result.lower()

    @pytest.mark.asyncio
    async def test_sms_no_twilio_client_returns_not_configured(
        self, registered_tools, db_session, tool_business
    ):
        """No twilio_client returns 'SMS service is not configured'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "send_sms_confirmation",
            {"phone": "+15551234567", "message": "Your appointment is confirmed."},
            business_id=tool_business.id,
            db_session=db_session,
            twilio_client=None,
        )
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_sms_success_returns_confirmation(
        self, registered_tools, db_session, tool_business, mock_twilio
    ):
        """With mock twilio_client, returns 'SMS confirmation sent'."""
        from voice_agent.tool_registry import dispatch

        result = await dispatch(
            "send_sms_confirmation",
            {"phone": "+15551234567", "message": "Your appointment is confirmed."},
            business_id=tool_business.id,
            db_session=db_session,
            twilio_client=mock_twilio,
        )
        assert "sms confirmation sent" in result.lower()
        assert "+15551234567" in result
