"""Tests for voice_agent.call_manager — CallManager lifecycle.

Tests the full lifecycle of a phone call: start, audio handling, tool
dispatch, and teardown.  Database uses an in-memory SQLite engine; the
Gemini Live API is mocked entirely.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from voice_agent.call_manager import CallManager, active_calls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def call_db():
    """Configure the global SQLAlchemy session factory with an in-memory
    SQLite database, create all tables, and seed a test Business + FAQs.

    The global factory is set via ``configure_session_factory()`` so that
    ``CallManager.start_call()`` (which calls ``get_session_factory()``)
    operates on this database.
    """
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from voice_agent.database import configure_session_factory
    from voice_agent.models import Base, Business, FAQ

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    configure_session_factory(factory)

    async with factory() as session:
        biz = Business(
            name="Test Business",
            slug="test-business",
            phone_number="+12025551234",
            greeting_message="Hello! Welcome to {business_name}.",
            timezone="America/New_York",
            business_hours={
                "monday": {"open": "09:00", "close": "17:00"},
                "tuesday": {"open": "09:00", "close": "17:00"},
            },
            transfer_phone_number="+12025559876",
        )
        session.add(biz)
        await session.flush()
        await session.refresh(biz)

        # Seed a couple of FAQs
        faq1 = FAQ(
            business_id=biz.id,
            question="What are your hours?",
            answer="We are open Mon-Fri 9-5.",
            category="hours",
            keywords="hours, open, close",
        )
        session.add(faq1)
        await session.flush()

        yield biz

    await engine.dispose()


@pytest_asyncio.fixture
async def call_db_no_business():
    """Configure the global session factory with a clean database that has
    *no* Business records.  Used to test the ``start_call`` failure path.
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

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    configure_session_factory(factory)
    yield
    await engine.dispose()


@pytest.fixture
def mock_gemini_handler():
    """Mock ``GeminiCallHandler`` at the ``call_manager`` module level so
    that no real Gemini API calls are made.

    Returns ``(mock_class, mock_instance)`` — tests can further configure
    ``mock_instance`` (e.g. ``receive`` return values).
    """
    with patch("voice_agent.call_manager.GeminiCallHandler") as mock_class:
        instance = MagicMock()
        instance.connect = AsyncMock()
        instance.disconnect = AsyncMock()
        instance.send_audio = AsyncMock()
        instance.send_tool_response = AsyncMock()
        # receive() will be set per-test when needed
        instance.receive = MagicMock()
        mock_class.return_value = instance
        yield mock_class, instance


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """CallManager creation and default attribute values."""

    def test_create_call_manager_sets_defaults(self):
        """Creating a CallManager with a business ID sets all default attributes."""
        cm = CallManager(business_id=1)

        assert cm.business_id == 1
        assert cm.gemini is None
        assert cm.twilio_stream_sid is None
        assert cm.twilio_call_sid is None
        assert cm.caller_number is None
        assert cm.call_record_id is None
        assert cm.transcript == []
        assert cm.appointment_booked is False
        assert cm.transferred is False
        assert cm._running is True
        assert cm._start_time is None


# ---------------------------------------------------------------------------
# start_call
# ---------------------------------------------------------------------------


class TestStartCall:
    """Starting a new call — Gemini session + DB record."""

    @pytest.mark.asyncio
    async def test_start_call_initializes_gemini_and_creates_db_record(
        self, call_db, mock_gemini_handler,
    ):
        """After start_call, a GeminiCallHandler is created (and connected),
        and a Call record exists in the database."""
        mock_class, mock_instance = mock_gemini_handler
        biz = call_db

        cm = CallManager(business_id=biz.id)
        await cm.start_call(
            twilio_stream_sid="MZ_test_stream",
            twilio_call_sid="CA_test_call",
            caller_number="+12025551234",
        )

        # Gemini handler was created and connected
        mock_class.assert_called_once()
        mock_instance.connect.assert_awaited_once()
        assert cm.gemini is mock_instance

        # Twilio IDs stored
        assert cm.twilio_stream_sid == "MZ_test_stream"
        assert cm.twilio_call_sid == "CA_test_call"
        assert cm.caller_number == "+12025551234"

        # DB record was created
        assert cm.call_record_id is not None

        # Verify call record in database
        from sqlalchemy import select
        from voice_agent.database import get_session_factory
        from voice_agent.models import Call

        factory = get_session_factory()
        async with factory() as db:
            record = await db.get(Call, cm.call_record_id)
            assert record is not None
            assert record.business_id == biz.id
            assert record.caller_number == "+12025551234"
            assert record.twilio_call_sid == "CA_test_call"
            assert record.outcome == "hangup"  # default

    @pytest.mark.asyncio
    async def test_start_call_unknown_business_raises_error(
        self, call_db_no_business,
    ):
        """CallManager with an ID that does not exist in the database raises
        RuntimeError on start_call."""
        cm = CallManager(business_id=9999)

        with pytest.raises(RuntimeError, match="Business 9999 not found"):
            await cm.start_call(
                twilio_stream_sid="MZ_fail",
                twilio_call_sid="CA_fail",
                caller_number="+12025550000",
            )


# ---------------------------------------------------------------------------
# end_call
# ---------------------------------------------------------------------------


class TestEndCall:
    """Ending a call — teardown + DB update."""

    @pytest.mark.asyncio
    async def test_end_call_updates_db_record(
        self, call_db, mock_gemini_handler,
    ):
        """After start_call, end_call updates the DB Call record outcome
        and duration."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_1", "CA_1", "+12025551234")
        call_record_id = cm.call_record_id

        await cm.end_call("appointment_booked")

        # Gemini session was disconnected
        mock_instance.disconnect.assert_awaited_once()

        # DB record was updated
        from voice_agent.database import get_session_factory
        from voice_agent.models import Call

        factory = get_session_factory()
        async with factory() as db:
            record = await db.get(Call, call_record_id)
            assert record is not None
            assert record.outcome == "appointment_booked"
            assert record.duration_seconds is not None
            assert record.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_end_call_without_start_call_does_nothing(self, mock_gemini_handler):
        """Calling end_call without calling start_call first does not raise."""
        cm = CallManager(business_id=1)
        # No session, no DB record, no active_calls entry — should be safe
        await cm.end_call("hangup")  # Should not raise

    @pytest.mark.asyncio
    async def test_end_call_removes_from_active_calls(
        self, call_db, mock_gemini_handler,
    ):
        """After start_call, the call should be in the active_calls dict.
        After end_call, it should be removed."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_2", "CA_2", "+12025551234")

        # Manually register in active_calls (normally done by twilio_handler)
        active_calls["CA_2"] = cm
        assert "CA_2" in active_calls

        await cm.end_call("hangup")

        assert "CA_2" not in active_calls

    @pytest.mark.asyncio
    async def test_end_call_multiple_times_safe(
        self, call_db, mock_gemini_handler,
    ):
        """Calling end_call multiple times does not raise an error."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_3", "CA_3", "+12025551234")
        active_calls["CA_3"] = cm

        await cm.end_call("hangup")   # First call — normal teardown
        await cm.end_call("hangup")   # Second call — should not raise

        # disconnect is called each time because gemini handle is still set
        # (only the session handle inside GeminiCallHandler is cleared)
        assert mock_instance.disconnect.await_count >= 1


# ---------------------------------------------------------------------------
# handle_audio
# ---------------------------------------------------------------------------


class TestHandleAudio:
    """Processing incoming audio from Twilio."""

    @pytest.mark.asyncio
    async def test_handle_audio_before_connect_logs_warning(self, mock_gemini_handler):
        """handle_audio before Gemini is connected does not raise (just logs)."""
        cm = CallManager(business_id=1)
        await cm.handle_audio("base64payload")  # Should not raise

    @pytest.mark.asyncio
    async def test_handle_audio_after_start_converts_and_sends(
        self, call_db, mock_gemini_handler,
    ):
        """After start_call, handle_audio converts the payload and sends
        PCM to the Gemini handler."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_4", "CA_4", "+12025551234")

        # Mock the audio converter
        fake_pcm = b"\x00\x01\x02\x03"
        with patch(
            "voice_agent.call_manager.audio_converter.twilio_to_gemini",
            return_value=fake_pcm,
        ) as mock_convert:
            await cm.handle_audio("dGVzdC1hdWRpbw==")

            mock_convert.assert_called_once_with("dGVzdC1hdWRpbw==")
            mock_instance.send_audio.assert_awaited_once_with(fake_pcm)

    @pytest.mark.asyncio
    async def test_handle_audio_convert_error_handled_gracefully(
        self, call_db, mock_gemini_handler,
    ):
        """If audio conversion fails, handle_audio logs the error and
        does not propagate the exception."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_5", "CA_5", "+12025551234")

        with patch(
            "voice_agent.call_manager.audio_converter.twilio_to_gemini",
            side_effect=ValueError("bad audio"),
        ):
            await cm.handle_audio("aW52YWxpZA==")
            # Should not raise — error is logged
            mock_instance.send_audio.assert_not_called()


# ---------------------------------------------------------------------------
# send_audio_to_twilio
# ---------------------------------------------------------------------------


class TestSendAudioToTwilio:
    """Sending audio chunks to the Twilio WebSocket."""

    @pytest.mark.asyncio
    async def test_send_audio_to_twilio_sends_json(
        self, call_db, mock_gemini_handler,
    ):
        """send_audio_to_twilio reads from the audio queue and sends
        correct JSON media messages to the websocket."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call(
            twilio_stream_sid="MZ_stream",
            twilio_call_sid="CA_call",
            caller_number="+12025551234",
        )

        # Pre-populate the audio queue
        chunk_b64 = "dGVzdCBhdWRpbyBjaHVuaw=="
        await cm._audio_out_queue.put(chunk_b64)

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()

        # Run the send task
        task = asyncio.create_task(cm.send_audio_to_twilio(mock_ws))

        # Give the task a moment to process the queued item
        await asyncio.sleep(0.05)

        # Stop the loop
        cm._running = False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        mock_ws.send_json.assert_awaited_once_with(
            {
                "event": "media",
                "streamSid": "MZ_stream",
                "media": {"payload": chunk_b64},
            },
        )

    @pytest.mark.asyncio
    async def test_send_audio_to_twilio_break_on_send_error(
        self, call_db, mock_gemini_handler,
    ):
        """If websocket.send_json raises, the loop breaks."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_err", "CA_err", "+12025551234")

        await cm._audio_out_queue.put("c29tZSBhdWRpbw==")

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=RuntimeError("WS closed"))

        task = asyncio.create_task(cm.send_audio_to_twilio(mock_ws))

        await asyncio.sleep(0.05)

        # Task should have broken out of the loop
        assert task.done() or not cm._running

        cm._running = False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# process_gemini_responses
# ---------------------------------------------------------------------------


class TestProcessGeminiResponses:
    """Processing events from the Gemini session (background task)."""

    @pytest.mark.asyncio
    async def test_process_gemini_responses_before_connect_logs_warning(
        self, mock_gemini_handler,
    ):
        """Calling process_gemini_responses before connecting does not raise."""
        cm = CallManager(business_id=1)
        # This should just log a warning and return
        await cm.process_gemini_responses()

    @pytest.mark.asyncio
    async def test_process_gemini_responses_queues_audio(
        self, call_db, mock_gemini_handler,
    ):
        """Audio events from Gemini are converted and queued for Twilio."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_aud", "CA_aud", "+12025551234")

        # Set up receive() to yield an audio event
        async def _gen_audio():
            yield ("audio", b"\xAA\xBB\xCC")

        mock_instance.receive = MagicMock(return_value=_gen_audio())

        # Mock the audio converter for Gemini→Twilio
        fake_twilio_b64 = "ZmFrZSB0d2lsaW8="
        with patch(
            "voice_agent.call_manager.audio_converter.gemini_to_twilio",
            return_value=fake_twilio_b64,
        ) as mock_convert:
            await cm.process_gemini_responses()

            mock_convert.assert_called_once_with(b"\xAA\xBB\xCC")

        # Verify the converted audio is in the queue
        queued = await asyncio.wait_for(cm._audio_out_queue.get(), timeout=1.0)
        assert queued == fake_twilio_b64

    @pytest.mark.asyncio
    async def test_process_gemini_responses_dispatches_tool(
        self, call_db, mock_gemini_handler,
    ):
        """Tool_call events are dispatched and the response is sent back
        to Gemini."""
        mock_class, mock_instance = mock_gemini_handler
        cm = CallManager(business_id=call_db.id)
        await cm.start_call("MZ_tc", "CA_tc", "+12025551234")

        # Register a simple tool in the registry
        from voice_agent.tool_registry import register

        def dummy_handler(parameters, **context):
            return "Tool executed"

        register("test_tool", {"name": "test_tool"}, dummy_handler)

        # Create a mock FunctionCall
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {"param1": "value1"}
        fc.id = "fc_test"

        # yield a tool_call event
        async def _gen_tool():
            yield ("tool_call", fc)

        mock_instance.receive = MagicMock(return_value=_gen_tool())

        await cm.process_gemini_responses()

        # send_tool_response was called
        mock_instance.send_tool_response.assert_awaited_once()
        call_args = mock_instance.send_tool_response.call_args
        # FunctionResponses are passed as positional arg `function_responses`
        responses = call_args.kwargs.get("function_responses")
        if responses is None:
            args, _ = call_args
            responses = args[0] if args else None

        assert responses is not None and len(responses) > 0
        # The real GeminiCallHandler.send_tool_response converts dicts to
        # FunctionResponse objects, but our mock receives the raw dicts.
        assert responses[0]["name"] == "test_tool"
        assert "Tool executed" in responses[0]["response"]["result"]


# ---------------------------------------------------------------------------
# Dispatch tool (internal)
# ---------------------------------------------------------------------------


class TestDispatchTool:
    """Internal tool dispatch — state updates and error handling."""

    @pytest.mark.asyncio
    async def test_dispatch_tool_updates_state_on_book_appointment(
        self, mock_gemini_handler,
    ):
        """Dispatching 'book_appointment' sets appointment_booked = True."""
        from voice_agent.tool_registry import register

        def book_handler(parameters, **context):
            return "Appointment booked for 2025-07-10 at 10:00."

        register("book_appointment", {"name": "book_appointment"}, book_handler)

        cm = CallManager(business_id=1)
        fc = MagicMock()
        fc.name = "book_appointment"
        fc.args = {"date": "2025-07-10", "time": "10:00"}
        fc.id = "fc_book"

        result = await cm._dispatch_tool(fc)

        assert cm.appointment_booked is True
        assert len(result) == 1
        assert result[0]["name"] == "book_appointment"
        assert result[0]["id"] == "fc_book"
        assert "booked" in result[0]["response"]["result"].lower()

    @pytest.mark.asyncio
    async def test_dispatch_tool_updates_state_on_transfer_to_human(
        self, mock_gemini_handler,
    ):
        """Dispatching 'transfer_to_human' sets transferred = True."""
        from voice_agent.tool_registry import register

        def transfer_handler(parameters, **context):
            return "Transferring to a human agent."

        register("transfer_to_human", {"name": "transfer_to_human"}, transfer_handler)

        cm = CallManager(business_id=1)
        fc = MagicMock()
        fc.name = "transfer_to_human"
        fc.args = {}
        fc.id = "fc_transfer"

        result = await cm._dispatch_tool(fc)

        assert cm.transferred is True
        assert result[0]["name"] == "transfer_to_human"

    @pytest.mark.asyncio
    async def test_dispatch_tool_handles_unknown_tool(
        self, mock_gemini_handler,
    ):
        """Dispatching an unknown tool returns an error response (does not
        raise KeyError to the caller)."""
        cm = CallManager(business_id=1)
        fc = MagicMock()
        fc.name = "nonexistent_tool"
        fc.args = {}
        fc.id = "fc_unknown"

        # No registration for this tool — dispatch should catch KeyError
        result = await cm._dispatch_tool(fc)

        assert len(result) == 1
        assert result[0]["name"] == "nonexistent_tool"
        assert result[0]["id"] == "fc_unknown"
        assert "Unknown tool" in result[0]["response"]["result"]

    @pytest.mark.asyncio
    async def test_dispatch_tool_handler_error_returns_error_response(
        self, mock_gemini_handler,
    ):
        """If the tool handler raises, _dispatch_tool returns an error
        response (does not propagate)."""
        from voice_agent.tool_registry import register

        def failing_handler(parameters, **context):
            raise ValueError("Something went wrong")

        register("failing_tool", {"name": "failing_tool"}, failing_handler)

        cm = CallManager(business_id=1)
        fc = MagicMock()
        fc.name = "failing_tool"
        fc.args = {}
        fc.id = "fc_fail"

        result = await cm._dispatch_tool(fc)

        assert len(result) == 1
        assert "Error executing" in result[0]["response"]["result"]

    @pytest.mark.asyncio
    async def test_dispatch_tool_receives_context(
        self, mock_gemini_handler,
    ):
        """The tool handler receives business_id and call_manager in context."""
        from voice_agent.tool_registry import register

        captured = {}

        def inspect_handler(parameters, **context):
            captured["business_id"] = context.get("business_id")
            captured["call_manager"] = context.get("call_manager")
            return "ok"

        register("inspect_tool", {"name": "inspect_tool"}, inspect_handler)

        cm = CallManager(business_id=42)
        fc = MagicMock()
        fc.name = "inspect_tool"
        fc.args = {}
        fc.id = "fc_inspect"

        await cm._dispatch_tool(fc)

        assert captured["business_id"] == 42
        assert captured["call_manager"] is cm
