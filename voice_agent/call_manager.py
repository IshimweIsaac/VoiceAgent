"""Call Manager — orchestrates one phone call's lifecycle.

Wires together Twilio, Gemini Live API, and the tool system for a single
inbound call. Designed to be instantiated per-call with no shared mutable
state between instances.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voice_agent import audio_converter
from voice_agent.config import Settings
from voice_agent.database import get_session_factory
from voice_agent.gemini_client import GeminiCallHandler
from voice_agent.models import Business, Call, FAQ
from voice_agent.system_prompt import build_system_prompt
from voice_agent.tool_registry import get_schemas, trim_schemas, dispatch

logger = logging.getLogger(__name__)

# In-memory store of active calls keyed by Twilio Call SID
active_calls: dict[str, "CallManager"] = {}


class CallManager:
    """Manages one phone call's lifecycle.

    Attributes:
        business_id: The business ID for this call.
        gemini: The Gemini Live session handler (set after ``start_call``).
        twilio_stream_sid: Twilio Media Stream SID.
        twilio_call_sid: Twilio Call SID.
        caller_number: Caller's phone number in E.164.
        call_record_id: DB ``Call.id`` (set after DB record created).
        transcript: List of dicts with ``role``, ``text``, ``timestamp_ms``.
        appointment_booked: Whether an appointment was booked during this call.
        transferred: Whether the call was transferred to a human.
    """

    def __init__(self, business_id: int) -> None:
        """Initialize the call manager.

        Args:
            business_id: The business ID this call belongs to.
        """
        self.business_id = business_id
        self.gemini: GeminiCallHandler | None = None
        self.twilio_stream_sid: str | None = None
        self.twilio_call_sid: str | None = None
        self.caller_number: str | None = None
        self.call_record_id: int | None = None
        self.transcript: list[dict[str, Any]] = []
        self.appointment_booked: bool = False
        self.transferred: bool = False
        self._settings: Settings | None = None

        # Internal state
        self._audio_out_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = True
        self._start_time: datetime | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_call(
        self,
        twilio_stream_sid: str,
        twilio_call_sid: str,
        caller_number: str,
    ) -> None:
        """Initialize the Gemini session and create a DB call record.

        Args:
            twilio_stream_sid: Twilio Media Stream SID (``MZ...``).
            twilio_call_sid: Twilio Call SID (``CA...``).
            caller_number: Caller's phone number in E.164.

        Raises:
            RuntimeError: If the business cannot be found or DB is not ready.
        """
        self.twilio_stream_sid = twilio_stream_sid
        self.twilio_call_sid = twilio_call_sid
        self.caller_number = caller_number
        self._start_time = datetime.now(timezone.utc)

        # Load settings (set from global in main.py lifespan)
        from voice_agent.config import Settings as _Settings

        self._settings = _Settings()  # type: ignore[call-arg]

        # Acquire a DB session
        factory = get_session_factory()
        async with factory() as db:
            # 1. Load business + FAQs
            business = await self._load_business(db)
            if business is None:
                raise RuntimeError(f"Business {self.business_id} not found in database")

            faqs = await self._load_faqs(db)

            # 2. Build system prompt
            system_prompt = build_system_prompt(
                business_name=business.name,
                greeting_message=business.greeting_message,
                business_hours=business.business_hours,
                timezone=business.timezone,
                faqs=faqs,
            )

            # 3. Get tool schemas
            raw_schemas = get_schemas()
            tool_schemas = trim_schemas(raw_schemas)

            # 4. Create Gemini session
            self.gemini = GeminiCallHandler(
                api_key=self._settings.gemini_api_key,
                business_id=self.business_id,
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
            )
            await self.gemini.connect()

            # 5. Create Call record in DB
            call_record = Call(
                business_id=self.business_id,
                caller_number=caller_number,
                twilio_call_sid=twilio_call_sid,
                outcome="hangup",  # default; updated on end_call
                transcript_json=[],
            )
            db.add(call_record)
            await db.commit()
            await db.refresh(call_record)
            self.call_record_id = call_record.id

        logger.info(
            "Call %s started (business=%d, caller=%s, call_record=%d)",
            twilio_call_sid,
            self.business_id,
            caller_number,
            self.call_record_id,
        )

    async def handle_audio(self, twilio_payload_b64: str) -> None:
        """Convert a Twilio audio chunk and forward it to Gemini.

        Args:
            twilio_payload_b64: Base64-encoded µ-law audio from Twilio.
        """
        if self.gemini is None:
            logger.warning("handle_audio called before Gemini connected")
            return

        try:
            pcm_16khz = audio_converter.twilio_to_gemini(twilio_payload_b64)
            await self.gemini.send_audio(pcm_16khz)
        except Exception as exc:
            logger.error("handle_audio error: %s", exc)

    async def process_gemini_responses(self) -> None:
        """Receive from Gemini and dispatch audio / tool calls.

        This should be run as a background task (``asyncio.create_task``).
        Audio chunks are queued to ``_audio_out_queue`` for the Twilio
        send task to consume. Tool calls are dispatched immediately.
        """
        if self.gemini is None:
            logger.warning("process_gemini_responses called before Gemini connected")
            return

        async for response_type, data in self.gemini.receive():
            if response_type == "audio":
                # Convert 24 kHz PCM → 8 kHz µ-law base64 and queue
                twilio_chunk = audio_converter.gemini_to_twilio(data)
                await self._audio_out_queue.put(twilio_chunk)

            elif response_type == "tool_call":
                result = await self._dispatch_tool(data)
                await self.gemini.send_tool_response(result)

            elif response_type == "turn_complete":
                # Gemini finished speaking — no action needed on server
                pass

    async def send_audio_to_twilio(self, websocket: Any) -> None:
        """Send queued audio chunks to the Twilio WebSocket.

        Run this as a background task. It reads from ``_audio_out_queue``
        and writes JSON messages to the Twilio Media Stream WebSocket.

        Args:
            websocket: The FastAPI WebSocket connection to Twilio.
        """
        while self._running:
            try:
                chunk_b64 = await asyncio.wait_for(
                    self._audio_out_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            if not self._running:
                break

            try:
                await websocket.send_json(
                    {
                        "event": "media",
                        "streamSid": self.twilio_stream_sid,
                        "media": {"payload": chunk_b64},
                    }
                )
            except Exception as exc:
                logger.error("Failed to send audio to Twilio: %s", exc)
                break

    async def end_call(self, outcome: str) -> None:
        """Teardown — close Gemini session and update the DB call record.

        Args:
            outcome: One of ``"appointment_booked"``, ``"faq_answered"``,
                     ``"transferred"``, ``"hangup"``, or ``"error"``.
        """
        self._running = False

        # Close Gemini session
        if self.gemini is not None:
            await self.gemini.disconnect()

        # Update DB call record
        if self.call_record_id is not None:
            try:
                factory = get_session_factory()
                async with factory() as db:
                    call_record = await db.get(Call, self.call_record_id)
                    if call_record is not None:
                        call_record.outcome = outcome
                        call_record.transcript_json = self.transcript

                        if self._start_time is not None:
                            elapsed = datetime.now(timezone.utc) - self._start_time
                            call_record.duration_seconds = int(elapsed.total_seconds())

                        await db.commit()

                logger.info(
                    "Call %s ended (outcome=%s, duration=%ds)",
                    self.twilio_call_sid,
                    outcome,
                    call_record.duration_seconds if call_record else 0,
                )
            except Exception as exc:
                logger.error("Failed to update call record: %s", exc)

        # Remove from active calls
        if self.twilio_call_sid and self.twilio_call_sid in active_calls:
            del active_calls[self.twilio_call_sid]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_business(self, db: AsyncSession) -> Business | None:
        """Load a business by ID from the database."""
        result = await db.execute(
            select(Business).where(Business.id == self.business_id)
        )
        return result.scalar_one_or_none()

    async def _load_faqs(self, db: AsyncSession) -> list[FAQ]:
        """Load enabled FAQs for the business."""
        result = await db.execute(
            select(FAQ).where(
                FAQ.business_id == self.business_id,
                FAQ.enabled.is_(True),
            )
        )
        return list(result.scalars().all())

    async def _dispatch_tool(self, function_call: Any) -> list[dict[str, Any]]:
        """Execute a tool call and return a ``FunctionResponse``-compatible list.

        Args:
            function_call: A Gemini ``FunctionCall`` object with ``name``,
                          ``args``, and ``id`` attributes.

        Returns:
            List containing one dict: ``{"id": ..., "name": ..., "response": {"result": ...}}``.
        """
        name = function_call.name
        args = dict(function_call.args or {})

        logger.info(
            "Tool call: %s args=%s (call=%s)",
            name,
            args,
            self.twilio_call_sid,
        )

        # Build context for tool handlers
        context = {
            "business_id": self.business_id,
            "call_manager": self,
        }

        try:
            # Try dispatch through the registry
            result = await dispatch(name, parameters=args, **context)

            # Update call state based on tool
            if name == "book_appointment":
                self.appointment_booked = True
            elif name == "transfer_to_human":
                self.transferred = True

            return [
                {
                    "id": function_call.id,
                    "name": name,
                    "response": {"result": result},
                }
            ]
        except KeyError:
            # Unknown tool — Gemini will see the error and respond
            return [
                {
                    "id": function_call.id,
                    "name": name,
                    "response": {"result": f"Unknown tool: {name}"},
                }
            ]
        except Exception as exc:
            logger.error("Tool dispatch error %s: %s", name, exc)
            return [
                {
                    "id": function_call.id,
                    "name": name,
                    "response": {"result": f"Error executing {name}: {exc}"},
                }
            ]
