"""Tests for SQLAlchemy ORM models — Business, User, FAQ, Call, Appointment."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError


# ---------------------------------------------------------------------------
# Business
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_business_sets_fields(db_session):
    """Create a Business with all fields, verify they're stored correctly."""
    from voice_agent.models import Business

    biz = Business(
        name="Test Corp",
        slug="test-corp",
        phone_number="+15551111111",
        greeting_message="Hello, you've reached {business_name}.",
        timezone="America/Chicago",
        business_hours={"monday": {"open": "09:00", "close": "17:00"}},
        transfer_phone_number="+15552222222",
    )
    db_session.add(biz)
    await db_session.flush()
    await db_session.refresh(biz)

    assert biz.id is not None
    assert biz.name == "Test Corp"
    assert biz.slug == "test-corp"
    assert biz.phone_number == "+15551111111"
    assert biz.greeting_message == "Hello, you've reached {business_name}."
    assert biz.timezone == "America/Chicago"
    assert biz.business_hours == {"monday": {"open": "09:00", "close": "17:00"}}
    assert biz.transfer_phone_number == "+15552222222"
    assert biz.enabled is True
    assert biz.created_at is not None


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_user_links_to_business(db_session, test_business):
    """Create a User with business_id, verify relationship works."""
    from voice_agent.models import User

    user = User(
        business_id=test_business.id,
        email="user@test.com",
        password_hash="hashed_password",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    assert user.id is not None
    assert user.business_id == test_business.id
    assert user.email == "user@test.com"

    # Relationship: user.business
    assert user.business is not None
    assert user.business.id == test_business.id
    assert user.business.name == "Test Business"


@pytest.mark.asyncio
async def test_business_user_relationship(db_session, test_business, test_user):
    """Business.user returns the associated User (one-to-one)."""
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    from voice_agent.models import Business

    # Eagerly load the user relationship
    stmt = (
        select(Business)
        .where(Business.id == test_business.id)
        .options(selectinload(Business.user))
    )
    result = await db_session.execute(stmt)
    biz = result.scalar_one()
    assert biz.user is not None
    assert biz.user.id == test_user.id
    assert biz.user.email == "admin@test.com"


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_business_faqs_relationship(db_session, test_business):
    """Business.faqs returns associated FAQ records."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from voice_agent.models import Business, FAQ

    faq1 = FAQ(
        business_id=test_business.id, question="Q1?", answer="A1.", category="general"
    )
    faq2 = FAQ(
        business_id=test_business.id,
        question="Q2?",
        answer="A2.",
        category="services",
    )
    db_session.add_all([faq1, faq2])
    await db_session.flush()

    # Eagerly load FAQs
    stmt = (
        select(Business)
        .where(Business.id == test_business.id)
        .options(selectinload(Business.faqs))
    )
    result = await db_session.execute(stmt)
    biz = result.scalar_one()
    assert len(biz.faqs) == 2
    faq_questions = {f.question for f in biz.faqs}
    assert faq_questions == {"Q1?", "Q2?"}


@pytest.mark.asyncio
async def test_create_faq_and_verify(db_session):
    """Create an FAQ record and verify fields."""
    from voice_agent.models import Business, FAQ

    biz = Business(name="Biz", slug="biz-faq", phone_number="+15551111112")
    db_session.add(biz)
    await db_session.flush()

    faq = FAQ(
        business_id=biz.id,
        question="What are your hours?",
        answer="We are open 9-5.",
        category="hours",
        keywords="hours, open, close",
    )
    db_session.add(faq)
    await db_session.flush()
    await db_session.refresh(faq)

    assert faq.id is not None
    assert faq.question == "What are your hours?"
    assert faq.answer == "We are open 9-5."
    assert faq.category == "hours"
    assert faq.keywords == "hours, open, close"
    assert faq.enabled is True
    assert faq.business_id == biz.id


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_business_calls_relationship(db_session, test_business):
    """Business.calls returns associated Call records."""
    from voice_agent.models import Call

    call1 = Call(
        business_id=test_business.id, caller_number="+15551111111", outcome="hangup"
    )
    call2 = Call(
        business_id=test_business.id,
        caller_number="+15552222222",
        outcome="appointment_booked",
    )
    db_session.add_all([call1, call2])
    await db_session.flush()

    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    from voice_agent.models import Business

    stmt = (
        select(Business)
        .where(Business.id == test_business.id)
        .options(selectinload(Business.calls))
    )
    result = await db_session.execute(stmt)
    biz = result.scalar_one()
    assert len(biz.calls) == 2


@pytest.mark.asyncio
async def test_create_call_and_verify(db_session):
    """Create a Call record with all fields."""
    from voice_agent.models import Business, Call

    biz = Business(name="CallBiz", slug="call-biz", phone_number="+15551111113")
    db_session.add(biz)
    await db_session.flush()

    call = Call(
        business_id=biz.id,
        caller_number="+15553333333",
        duration_seconds=120,
        transcript_json=[{"role": "user", "text": "Hello"}],
        outcome="faq_answered",
        twilio_call_sid="CA123456789",
    )
    db_session.add(call)
    await db_session.flush()
    await db_session.refresh(call)

    assert call.id is not None
    assert call.business_id == biz.id
    assert call.caller_number == "+15553333333"
    assert call.duration_seconds == 120
    assert call.transcript_json == [{"role": "user", "text": "Hello"}]
    assert call.outcome == "faq_answered"
    assert call.twilio_call_sid == "CA123456789"
    assert call.created_at is not None


# ---------------------------------------------------------------------------
# Appointment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_appointment_and_verify(db_session):
    """Create an Appointment record with all fields."""
    from datetime import datetime

    from voice_agent.models import Appointment, Business

    biz = Business(name="ApptBiz", slug="appt-biz", phone_number="+15551111114")
    db_session.add(biz)
    await db_session.flush()

    start = datetime(2026, 7, 10, 9, 0)
    end = datetime(2026, 7, 10, 9, 30)

    appt = Appointment(
        business_id=biz.id,
        customer_name="John Doe",
        customer_phone="+15554444444",
        start_time=start,
        end_time=end,
        description="Fix leaky faucet",
        status="confirmed",
        google_event_id="event-abc-123",
    )
    db_session.add(appt)
    await db_session.flush()
    await db_session.refresh(appt)

    assert appt.id is not None
    assert appt.business_id == biz.id
    assert appt.customer_name == "John Doe"
    assert appt.customer_phone == "+15554444444"
    assert appt.start_time == start
    assert appt.end_time == end
    assert appt.description == "Fix leaky faucet"
    assert appt.status == "confirmed"
    assert appt.google_event_id == "event-abc-123"
    assert appt.created_at is not None


@pytest.mark.asyncio
async def test_appointment_links_to_call(db_session):
    """Appointment.call relationship works."""
    from datetime import datetime, timedelta

    from voice_agent.models import Appointment, Business, Call

    biz = Business(
        name="ApptCallBiz", slug="appt-call-biz", phone_number="+15551111115"
    )
    db_session.add(biz)
    await db_session.flush()

    call = Call(
        business_id=biz.id, caller_number="+15555555555", outcome="appointment_booked"
    )
    db_session.add(call)
    await db_session.flush()

    now = datetime.utcnow()
    appt = Appointment(
        business_id=biz.id,
        call_id=call.id,
        customer_name="Jane",
        customer_phone="+15556666666",
        start_time=now,
        end_time=now + timedelta(hours=1),
    )
    db_session.add(appt)
    await db_session.flush()
    await db_session.refresh(appt)

    assert appt.call is not None
    assert appt.call.id == call.id
    assert appt.call.caller_number == "+15555555555"


# ---------------------------------------------------------------------------
# Cascade delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_delete_business_removes_all_related(db_session):
    """When a Business is deleted, its related User, FAQs, Calls, and
    Appointments are also deleted (cascade='all, delete-orphan')."""
    from datetime import datetime, timedelta

    from sqlalchemy import select

    from voice_agent.models import Appointment, Business, Call, FAQ, User

    biz = Business(
        name="DelBiz", slug="del-biz-cascade", phone_number="+15551111116"
    )
    db_session.add(biz)
    await db_session.flush()
    biz_id = biz.id

    # Create related records
    user = User(business_id=biz_id, email="del@test.com", password_hash="hash")
    faq = FAQ(business_id=biz_id, question="Q?", answer="A!")
    call = Call(business_id=biz_id, caller_number="+15557777777")
    db_session.add_all([user, faq, call])
    await db_session.flush()

    now = datetime.utcnow()
    appt = Appointment(
        business_id=biz_id,
        call_id=call.id,
        customer_name="Del",
        customer_phone="+15558888888",
        start_time=now,
        end_time=now + timedelta(minutes=30),
    )
    db_session.add(appt)
    await db_session.flush()

    # Delete business
    await db_session.delete(biz)
    await db_session.flush()

    # Verify all related records are gone
    assert (
        await db_session.execute(
            select(User).where(User.email == "del@test.com")
        )
    ).scalar_one_or_none() is None

    assert (
        await db_session.execute(
            select(FAQ).where(FAQ.question == "Q?")
        )
    ).scalar_one_or_none() is None

    assert (
        await db_session.execute(
            select(Call).where(Call.caller_number == "+15557777777")
        )
    ).scalar_one_or_none() is None

    assert (
        await db_session.execute(
            select(Appointment).where(Appointment.customer_name == "Del")
        )
    ).scalar_one_or_none() is None


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unique_slug_constraint(db_session):
    """Creating two businesses with the same slug raises an integrity error."""
    from voice_agent.models import Business

    biz1 = Business(name="First", slug="same-slug", phone_number="+15551111117")
    db_session.add(biz1)
    await db_session.flush()

    biz2 = Business(
        name="Second", slug="same-slug", phone_number="+15551111118"
    )
    db_session.add(biz2)
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_unique_phone_number_constraint(db_session):
    """Creating two businesses with the same phone_number raises integrity error."""
    from voice_agent.models import Business

    biz1 = Business(
        name="PhoneBiz1", slug="phone-biz-1", phone_number="+15551111119"
    )
    db_session.add(biz1)
    await db_session.flush()

    biz2 = Business(
        name="PhoneBiz2", slug="phone-biz-2", phone_number="+15551111119"
    )
    db_session.add(biz2)
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_unique_email_constraint(db_session):
    """Creating two users with the same email raises an integrity error."""
    from voice_agent.models import Business, User

    biz = Business(
        name="EmailBiz", slug="email-biz", phone_number="+15551111120"
    )
    db_session.add(biz)
    await db_session.flush()

    user1 = User(
        business_id=biz.id, email="dupe@test.com", password_hash="hash1"
    )
    db_session.add(user1)
    await db_session.flush()

    user2 = User(
        business_id=biz.id, email="dupe@test.com", password_hash="hash2"
    )
    db_session.add(user2)
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_outcome_enum_values(db_session):
    """Verify all valid enum values for Call.outcome work."""
    from voice_agent.models import Business, Call

    biz = Business(
        name="EnumBiz", slug="enum-biz", phone_number="+15551111121"
    )
    db_session.add(biz)
    await db_session.flush()

    for outcome in (
        "appointment_booked",
        "faq_answered",
        "transferred",
        "hangup",
        "error",
    ):
        call = Call(
            business_id=biz.id,
            caller_number="+15559999999",
            outcome=outcome,
        )
        db_session.add(call)
        await db_session.flush()
        assert call.outcome == outcome


@pytest.mark.asyncio
async def test_appointment_status_enum_values(db_session):
    """Verify all valid enum values for Appointment.status work."""
    from datetime import datetime, timedelta

    from voice_agent.models import Appointment, Business

    biz = Business(
        name="EnumApptBiz", slug="enum-appt-biz", phone_number="+15551111122"
    )
    db_session.add(biz)
    await db_session.flush()

    now = datetime.utcnow()
    for status in ("confirmed", "cancelled", "no_show"):
        appt = Appointment(
            business_id=biz.id,
            customer_name="Test User",
            customer_phone="+15551111123",
            start_time=now,
            end_time=now + timedelta(minutes=30),
            status=status,
        )
        db_session.add(appt)
        await db_session.flush()
        assert appt.status == status
