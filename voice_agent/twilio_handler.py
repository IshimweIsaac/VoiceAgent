"""Twilio HTTP + WebSocket handler — the bridge between Twilio and VoiceAgent.

Provides:

* ``POST /twilio/incoming`` — Called by Twilio when a call arrives. Returns
  TwiML with ``<Connect><Stream>`` pointing to the WebSocket endpoint.
* ``POST /twilio/status`` — Called by Twilio when call status changes.
* ``WebSocket /media/{business_slug}`` — Twilio Media Streams bidirectional
  audio channel for one call.

See SPEC sections 6.3, 6.4, and 13 for the full protocol specification.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select

from voice_agent.call_manager import CallManager, active_calls
from voice_agent.database import get_session_factory
from voice_agent.models import Business

logger = logging.getLogger(__name__)

router = APIRouter()


# ======================================================================
# HTTP endpoint — Twilio incoming call webhook
# ======================================================================


@router.post("/twilio/incoming", include_in_schema=False)
async def incoming_call(request: Request) -> HTMLResponse:
    """Handle an incoming Twilio call.

    Twilio POSTs to this endpoint when a call arrives. The handler:
    1. Parses form data (To, From, CallSid).
    2. Looks up the business by phone number.
    3. Returns TwiML with a ``<Connect><Stream>`` pointing to our WebSocket.

    If the business is not found, returns a simple "goodbye" message.
    """
    form = await request.form()
    to_number: str = form.get("To", "")  # Our Twilio number
    from_number: str = form.get("From", "")  # Caller number
    call_sid: str = form.get("CallSid", "")

    logger.info(
        "Incoming call: To=%s From=%s CallSid=%s",
        to_number,
        from_number,
        call_sid,
    )

    # Look up business by phone number
    business = await _get_business_by_phone(to_number)

    if business is None:
        logger.warning("No business found for number %s", to_number)
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">This number is not configured. Goodbye.</Say>
</Response>"""
        return HTMLResponse(content=twiml, media_type="application/xml")

    # Build the WebSocket URL (use wss:// for HTTPS, ws:// for HTTP)
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{request.url.hostname}/media/{business.slug}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}"/>
    </Connect>
    <Say voice="alice">One moment please...</Say>
</Response>"""

    return HTMLResponse(content=twiml, media_type="application/xml")


# ======================================================================
# HTTP endpoint — Twilio call status callback
# ======================================================================


@router.post("/twilio/status", include_in_schema=False)
async def call_status(request: Request) -> JSONResponse:
    """Handle Twilio call status callbacks.

    Twilio POSTs here when the call status changes (ringing, in-progress,
    completed, etc.). We log the event for observability.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    call_status = form.get("CallStatus", "unknown")
    logger.info("Call status update: %s → %s", call_sid, call_status)
    return JSONResponse({"status": "received"})


# ======================================================================
# WebSocket endpoint — Twilio Media Streams
# ======================================================================


@router.websocket("/media/{business_slug}")
async def media_stream(websocket: WebSocket, business_slug: str) -> None:
    """Handle a Twilio Media Stream WebSocket connection.

    Twilio connects here after the ``<Stream>`` TwiML is processed. The
    handler:

    1. Looks up the business by its URL slug.
    2. Creates a ``CallManager`` for this call.
    3. Processes Twilio JSON messages: ``connected``, ``start``, ``media``,
       ``stop``.
    4. On ``start``: initializes the Gemini session and starts background
       audio tasks.
    5. On ``media``: forwards audio to ``CallManager.handle_audio()``.
    6. On ``stop`` / disconnect: tears down the call.

    See SPEC section 6.3 for the message format.
    """
    await websocket.accept()
    logger.info("WebSocket connected: business_slug=%s", business_slug)

    # Look up business by slug
    business = await _get_business_by_slug(business_slug)

    if business is None:
        logger.warning("WebSocket rejected: unknown business slug %s", business_slug)
        await websocket.close(code=4004, reason="Unknown business")
        return

    call_manager = CallManager(business.id)
    stream_sid: str | None = None
    call_sid: str | None = None
    caller: str | None = None

    # Background task references (set on "start" event)
    send_task: Any = None
    receive_task: Any = None

    try:
        async for message in websocket.iter_json():
            event = message.get("event", "")

            # ----------------------------------------------------------
            # "connected" — Twilio confirms the WebSocket is open
            # ----------------------------------------------------------
            if event == "connected":
                logger.debug(
                    "Media stream connected (protocol=%s)", message.get("protocol")
                )
                continue

            # ----------------------------------------------------------
            # "start" — Twilio starts the media stream for this call
            # ----------------------------------------------------------
            elif event == "start":
                start_data = message.get("start", {})
                stream_sid = start_data.get("streamSid")
                call_sid = start_data.get("callSid")
                caller = start_data.get("from", "unknown")

                logger.info(
                    "Media stream start: stream=%s call=%s from=%s",
                    stream_sid,
                    call_sid,
                    caller,
                )

                # Initialize the call: Gemini session + DB record
                try:
                    await call_manager.start_call(
                        twilio_stream_sid=stream_sid,
                        twilio_call_sid=call_sid,
                        caller_number=caller,
                    )
                except Exception as exc:
                    logger.error("Failed to start call: %s", exc)
                    await websocket.close(code=1011, reason="Failed to initialize call")
                    return

                # Register in active calls
                if call_sid:
                    active_calls[call_sid] = call_manager

                # Start background tasks for bidirectional audio
                send_task = asyncio_create_task(
                    call_manager.send_audio_to_twilio(websocket),
                )
                receive_task = asyncio_create_task(
                    call_manager.process_gemini_responses(),
                )

            # ----------------------------------------------------------
            # "media" — Twilio sends an audio chunk from the caller
            # ----------------------------------------------------------
            elif event == "media":
                payload = message.get("media", {}).get("payload", "")
                if payload and call_manager.gemini is not None:
                    await call_manager.handle_audio(payload)

            # ----------------------------------------------------------
            # "stop" — Twilio ends the media stream (caller hung up)
            # ----------------------------------------------------------
            elif event == "stop":
                logger.info("Media stream stop: stream=%s", stream_sid)
                outcome = _determine_outcome(call_manager)
                await call_manager.end_call(outcome)
                _cancel_tasks(send_task, receive_task)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: slug=%s", business_slug)
        outcome = _determine_outcome(call_manager)
        await call_manager.end_call(outcome)
        _cancel_tasks(send_task, receive_task)

    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        await call_manager.end_call("error")
        _cancel_tasks(send_task, receive_task)


# ======================================================================
# Internal helpers
# ======================================================================


def asyncio_create_task(coro: Any) -> Any:
    """Safely create an asyncio task. Wrapped for testability."""
    import asyncio

    return asyncio.create_task(coro)


def _determine_outcome(call_manager: CallManager) -> str:
    """Determine the call outcome based on CallManager state."""
    if call_manager.appointment_booked:
        return "appointment_booked"
    if call_manager.transferred:
        return "transferred"
    return "hangup"


def _cancel_tasks(*tasks: Any) -> None:
    """Cancel one or more asyncio tasks safely."""
    for task in tasks:
        if task is not None and not task.done():
            task.cancel()


async def _get_business_by_phone(phone_number: str) -> Business | None:
    """Look up a business by their Twilio phone number."""
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(Business).where(Business.phone_number == phone_number)
            )
            return result.scalar_one_or_none()
    except Exception as exc:
        logger.error("Error looking up business by phone %s: %s", phone_number, exc)
        return None


async def _get_business_by_slug(slug: str) -> Business | None:
    """Look up a business by their URL slug."""
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(select(Business).where(Business.slug == slug))
            return result.scalar_one_or_none()
    except Exception as exc:
        logger.error("Error looking up business by slug %s: %s", slug, exc)
        return None
