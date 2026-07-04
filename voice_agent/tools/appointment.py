"""Appointment tools — check availability and book appointments.

Registered tools:
  - ``check_availability``
  - ``book_appointment``
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from voice_agent.models import Appointment, Business
from voice_agent.tool_registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------

CHECK_AVAILABILITY_SCHEMA: dict[str, Any] = {
    "name": "check_availability",
    "description": (
        "Check if a specific time slot is available for booking. "
        "Returns available slots for a given date."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "date": {
                "type": "STRING",
                "description": "Date to check availability in YYYY-MM-DD format",
            },
        },
        "required": ["date"],
    },
}


async def handle_check_availability(
    parameters: dict[str, Any],
    business_id: int,
    db_session: Any,
    calendar_service: Any = None,
    twilio_client: Any = None,
    call_manager: Any = None,
) -> str:
    """Check available appointment slots for a given date.

    Args:
        parameters: Must contain ``date`` (YYYY-MM-DD).
        business_id: Business to check availability for.
        db_session: Async SQLAlchemy session.
        calendar_service: Optional GoogleCalendarClient.
        twilio_client: Unused — dispatch compatibility.
        call_manager: Unused — dispatch compatibility.

    Returns:
        Formatted string of available slots, or a message explaining
        why none are available.
    """
    date = parameters.get("date", "").strip()
    if not date:
        return "Please provide a date to check availability for."

    if calendar_service is None:
        return (
            "I'm sorry, the calendar is not yet configured. "
            "Please contact the business directly to book an appointment."
        )

    try:
        available = await calendar_service.check_availability(date, duration_minutes=30)
    except Exception as exc:
        logger.error("Availability check failed for business %d: %s", business_id, exc)
        return (
            "I'm having trouble checking the calendar right now. "
            "Let me transfer you to someone who can help."
        )

    if not available:
        # Check if the business is closed that day
        try:
            stmt = select(Business).where(Business.id == business_id)
            result = await db_session.execute(stmt)
            business = result.scalar_one_or_none()
            if business:
                hours = business.business_hours or {}
                day_names = [
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ]
                parsed_date = datetime.strptime(date, "%Y-%m-%d")
                day_name = day_names[parsed_date.weekday()]
                day_hours = hours.get(day_name)
                if not day_hours or "open" not in day_hours:
                    return f"I'm sorry, we are closed on {date}."
        except Exception as exc:
            logger.warning("Failed to check business hours for date %s: %s", date, exc)
        return f"I'm sorry, there are no available slots on {date}. Would you like to try another date?"

    # Format available slots
    slot_strs: list[str] = []
    for slot in available:
        start = slot.get("start", "")
        end = slot.get("end", "")
        slot_strs.append(f"{start} - {end}")

    if len(slot_strs) <= 3:
        slots_formatted = ", ".join(slot_strs)
    else:
        slots_formatted = (
            ", ".join(slot_strs[:3]) + f", and {len(slot_strs) - 3} more slots"
        )

    return f"Available slots on {date}: {slots_formatted}."


# ---------------------------------------------------------------------------
# book_appointment
# ---------------------------------------------------------------------------

BOOK_APPOINTMENT_SCHEMA: dict[str, Any] = {
    "name": "book_appointment",
    "description": (
        "Book a new appointment for a customer. "
        "Asks for date, time, name, and phone. "
        "Confirms availability before booking."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "date": {
                "type": "STRING",
                "description": "Date for the appointment in YYYY-MM-DD format",
            },
            "time": {
                "type": "STRING",
                "description": "Time for the appointment in HH:MM 24-hour format",
            },
            "duration_minutes": {
                "type": "INTEGER",
                "description": "Duration in minutes (default 30)",
            },
            "customer_name": {
                "type": "STRING",
                "description": "Full name of the customer",
            },
            "customer_phone": {
                "type": "STRING",
                "description": "Customer's phone number",
            },
            "description": {
                "type": "STRING",
                "description": "Optional notes about the appointment",
            },
        },
        "required": ["date", "time", "customer_name", "customer_phone"],
    },
}


async def handle_book_appointment(
    parameters: dict[str, Any],
    business_id: int,
    db_session: Any,
    calendar_service: Any = None,
    twilio_client: Any = None,
    call_manager: Any = None,
) -> str:
    """Book an appointment: check availability, create event, store record.

    Args:
        parameters: Appointment details (date, time, customer_name, etc.).
        business_id: Business to book for.
        db_session: Async SQLAlchemy session.
        calendar_service: Optional GoogleCalendarClient.
        twilio_client: Unused — present for dispatch compatibility.
        call_manager: Optional call manager to flag that a booking was made.

    Returns:
        Confirmation message for Gemini to speak, or an error message.
    """
    # --- Extract and validate parameters ---
    date = (parameters.get("date") or "").strip()
    time = (parameters.get("time") or "").strip()
    duration = parameters.get("duration_minutes", 30)
    customer_name = (parameters.get("customer_name") or "").strip()
    customer_phone = (parameters.get("customer_phone") or "").strip()
    description = (parameters.get("description") or "").strip()

    if not date or not time or not customer_name or not customer_phone:
        return (
            "I need the date, time, your name, and phone number to book "
            "an appointment. Please provide all the details."
        )

    if calendar_service is None:
        return (
            "I'm sorry, the calendar is not yet configured. "
            "Please contact the business directly to book an appointment."
        )

    # --- Validate date/time format ---
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Sorry, I couldn't understand the date '{date}'. Please use YYYY-MM-DD format."

    try:
        datetime.strptime(time, "%H:%M")
    except ValueError:
        return f"Sorry, I couldn't understand the time '{time}'. Please use HH:MM 24-hour format."

    # --- Check availability first ---
    try:
        available_slots = await calendar_service.check_availability(date, duration)
    except Exception as exc:
        logger.error(
            "Availability check failed during booking for business %d: %s",
            business_id,
            exc,
        )
        return "I'm having trouble checking the calendar. Please try again later."

    if not available_slots:
        return (
            f"I'm sorry, there are no available slots on {date}. "
            "Would you like to try a different date?"
        )

    # Check if the specific time is within an available window
    slot_available = False
    for slot in available_slots:
        slot_start = slot.get("start", "")
        slot_end = slot.get("end", "")
        # Calculate requested end time
        hour, minute = int(time.split(":")[0]), int(time.split(":")[1])
        req_end_minutes = hour * 60 + minute + duration
        req_end_hour = req_end_minutes // 60
        req_end_min = req_end_minutes % 60
        req_end = f"{req_end_hour:02d}:{req_end_min:02d}"

        if slot_start <= time and slot_end >= req_end:
            slot_available = True
            break

    if not slot_available:
        return (
            f"I'm sorry, {time} on {date} is not available. "
            f"Please choose a different time."
        )

    # --- Create Google Calendar event ---
    try:
        google_event_id = await calendar_service.create_event(
            date=date,
            time=time,
            duration_minutes=duration,
            customer_name=customer_name,
            customer_phone=customer_phone,
            description=description,
        )
    except RuntimeError:
        return (
            "I'm sorry, the calendar is not fully connected. "
            "Please book through the business directly."
        )
    except Exception as exc:
        logger.error(
            "Failed to create calendar event for business %d: %s", business_id, exc
        )
        return (
            "I'm sorry, I couldn't create the appointment due to a "
            "technical issue. Please try again."
        )

    # --- Store appointment in database ---
    try:
        import zoneinfo

        stmt = select(Business).where(Business.id == business_id)
        result = await db_session.execute(stmt)
        business = result.scalar_one_or_none()
        tz_name = business.timezone if business and business.timezone else "UTC"
        tz = zoneinfo.ZoneInfo(tz_name)

        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        hour, minute = int(time.split(":")[0]), int(time.split(":")[1])
        start_dt = datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            hour,
            minute,
            tzinfo=tz,
        )
        end_dt = start_dt + timedelta(minutes=duration)

        appointment = Appointment(
            business_id=business_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            start_time=start_dt,
            end_time=end_dt,
            description=description,
            google_event_id=google_event_id,
            status="confirmed",
        )
        db_session.add(appointment)
        await db_session.flush()

        # Track the appointment ID on the call manager
        if call_manager is not None:
            call_manager.appointment_booked = True
            call_manager._appointment_id = appointment.id  # noqa: SLF001

    except Exception as exc:
        logger.error(
            "Failed to store appointment record for business %d: %s", business_id, exc
        )
        # Calendar event was created, but DB save failed — still return success
        # since the calendar event exists.

    logger.info(
        "Appointment booked for business %d: %s at %s (customer: %s, event: %s)",
        business_id,
        date,
        time,
        customer_name,
        google_event_id,
    )

    return (
        f"Your appointment has been booked for {date} at {time}. "
        f"A confirmation SMS will be sent to {customer_phone}. "
        f"Thank you, {customer_name}!"
    )


# ---------------------------------------------------------------------------
# Auto-register at import time
# ---------------------------------------------------------------------------
register("check_availability", CHECK_AVAILABILITY_SCHEMA, handle_check_availability)
register("book_appointment", BOOK_APPOINTMENT_SCHEMA, handle_book_appointment)
