"""Twilio REST API client — send SMS and manage calls.

Wraps the synchronous ``twilio.rest.Client`` with ``asyncio.to_thread``
so it can be called from async contexts without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from twilio.rest import Client

logger = logging.getLogger(__name__)


class TwilioClient:
    """Async wrapper around the Twilio REST API.

    Usage::

        client = TwilioClient(account_sid, auth_token)
        msg_sid = await client.send_sms("+12025551234", "+12025559876", "Hello!")
    """

    def __init__(self, account_sid: str, auth_token: str) -> None:
        """Create the underlying synchronous Twilio client.

        Args:
            account_sid: Twilio Account SID from the console.
            auth_token: Twilio Auth Token from the console.
        """
        self._client = Client(account_sid, auth_token)

    async def send_sms(self, to: str, from_: str, body: str) -> str:
        """Send an SMS message and return the Message SID.

        Args:
            to: Recipient phone number in E.164 format (e.g. ``+12025551234``).
            from_: Sender phone number (must be owned by the Twilio account).
            body: Message text content.

        Returns:
            The Twilio Message SID string (``SM...``).
        """
        try:
            message = await asyncio.to_thread(
                self._client.messages.create,
                to=to,
                from_=from_,
                body=body,
            )
            logger.info("SMS sent to %s: SID=%s", to, message.sid)
            return message.sid
        except Exception as exc:
            logger.error("Failed to send SMS to %s: %s", to, exc)
            raise

    async def get_call(self, call_sid: str) -> dict[str, Any]:
        """Get call details from Twilio.

        Args:
            call_sid: The Twilio Call SID (``CA...``).

        Returns:
            Dict representation of the Call resource.
        """
        try:
            call = await asyncio.to_thread(self._client.calls(call_sid).fetch)
            return {
                "sid": call.sid,
                "from": call.from_,
                "to": call.to,
                "status": call.status,
                "direction": call.direction,
                "duration": call.duration,
                "start_time": str(call.start_time) if call.start_time else None,
                "end_time": str(call.end_time) if call.end_time else None,
            }
        except Exception as exc:
            logger.error("Failed to fetch call %s: %s", call_sid, exc)
            raise

    @property
    def client(self) -> Client:
        """Access the underlying synchronous Twilio client directly if needed."""
        return self._client
