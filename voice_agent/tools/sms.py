"""SMS confirmation tool — sends appointment confirmation messages.

Registered tool: ``send_sms_confirmation``
"""

from __future__ import annotations

import logging
from typing import Any

from voice_agent.tool_registry import register

logger = logging.getLogger(__name__)

TOOL_SCHEMA: dict[str, Any] = {
    "name": "send_sms_confirmation",
    "description": (
        "Send an SMS confirmation for an appointment that was just booked. "
        "Only call this AFTER book_appointment succeeds."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "phone": {
                "type": "STRING",
                "description": "Phone number to send the SMS to (E.164 format)",
            },
            "message": {
                "type": "STRING",
                "description": "SMS message content",
            },
        },
        "required": ["phone", "message"],
    },
}


async def handler(
    parameters: dict[str, Any],
    business_id: int,
    db_session: Any,
    calendar_service: Any = None,
    twilio_client: Any = None,
    call_manager: Any = None,
) -> str:
    """Send an SMS confirmation via the Twilio client.

    Args:
        parameters: Must contain ``phone`` and ``message``.
        business_id: Unused — present for dispatch compatibility.
        db_session: Unused — present for dispatch compatibility.
        calendar_service: Unused — dispatch compatibility.
        twilio_client: Optional TwilioClient for sending the message.
        call_manager: Unused — dispatch compatibility.

    Returns:
        A confirmation message for Gemini to speak.
    """
    phone = parameters.get("phone", "").strip()
    message = parameters.get("message", "").strip()

    if not phone:
        return "I need a phone number to send the SMS."
    if not message:
        return "I need a message to send."

    if twilio_client is None:
        logger.warning(
            "SMS requested for business %d but Twilio is not configured",
            business_id,
        )
        return (
            "I'm sorry, the SMS service is not configured. "
            "I will still confirm the appointment during this call."
        )

    # We need the business's Twilio phone number as the sender.
    # Load it from the database.
    try:
        from sqlalchemy import select
        from voice_agent.models import Business

        stmt = select(Business).where(Business.id == business_id)
        result = await db_session.execute(stmt)
        business = result.scalar_one_or_none()
    except Exception as exc:
        logger.error("Failed to load business %d for SMS: %s", business_id, exc)
        return "I'm sorry, I couldn't send the SMS due to a system error."

    if business is None or not business.phone_number:
        logger.warning(
            "Business %d has no phone number configured for SMS", business_id
        )
        return (
            "I'm sorry, I couldn't send the SMS. Your appointment is still confirmed."
        )

    success = await twilio_client.send_sms(
        to=phone,
        from_=business.phone_number,
        body=message,
    )

    if success:
        logger.info("SMS confirmation sent to %s for business %d", phone, business_id)
        return f"SMS confirmation sent to {phone}."
    else:
        logger.error("Failed to send SMS to %s for business %d", phone, business_id)
        return (
            "I couldn't send the SMS due to a technical issue. "
            "Your appointment is still confirmed."
        )


# Auto-register at import time
register("send_sms_confirmation", TOOL_SCHEMA, handler)
