"""Human escalation tool — transfers the call to a human team member.

Registered tool: ``transfer_to_human``
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from voice_agent.models import Business
from voice_agent.tool_registry import register

logger = logging.getLogger(__name__)

TOOL_SCHEMA: dict[str, Any] = {
    "name": "transfer_to_human",
    "description": (
        "Transfer the call to a human team member. Use when the caller "
        "asks for a human or when you cannot resolve their request."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "reason": {
                "type": "STRING",
                "description": "Brief reason for the transfer (optional)",
            },
        },
        "required": [],
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
    """Look up the business's transfer number and flag the call for handoff.

    Args:
        parameters: May contain ``reason``.
        business_id: Business whose transfer number to look up.
        db_session: Async SQLAlchemy session.
        calendar_service: Unused — dispatch compatibility.
        twilio_client: Unused — dispatch compatibility.
        call_manager: Optional call manager to set ``transferred = True``.

    Returns:
        A message for Gemini to announce the transfer.
    """
    reason = parameters.get("reason", "No reason given")

    try:
        stmt = select(Business).where(Business.id == business_id)
        result = await db_session.execute(stmt)
        business = result.scalar_one_or_none()
    except Exception as exc:
        logger.error("Transfer lookup failed for business %d: %s", business_id, exc)
        return "I'm sorry, I'm unable to process the transfer right now."

    if business is None or not business.transfer_phone_number:
        logger.warning(
            "Transfer requested but no phone number configured for business %d",
            business_id,
        )
        return (
            "I'm sorry, there is no one available to take your call right now. "
            "Please try again later."
        )

    # Flag the call manager so the Twilio handler can initiate <Dial>
    if call_manager is not None:
        call_manager.transferred = True

    logger.info(
        "Transfer requested for business %d to %s (reason: %s)",
        business_id,
        business.transfer_phone_number,
        reason,
    )

    return (
        "Please hold while I transfer you to a member of our team. One moment please."
    )


# Auto-register at import time
register("transfer_to_human", TOOL_SCHEMA, handler)
