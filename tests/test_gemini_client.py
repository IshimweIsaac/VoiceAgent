"""Tests for voice_agent.gemini_client — GeminiCallHandler.

Tests the bidirectional Gemini Live API session manager. All external
Gemini API calls are mocked — no real network requests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch, ANY

import pytest

from voice_agent.gemini_client import GeminiCallHandler, LIVE_MODEL, VOICE_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _async_iterator(*items):
    """Build an async generator yielding *items in order."""
    for item in items:
        yield item


def _make_response_with_audio(data_bytes: bytes) -> MagicMock:
    """Build a mock LiveServerMessage carrying audio data."""
    resp = MagicMock()
    resp.data = MagicMock(spec_set=["data"])
    resp.data.data = data_bytes
    resp.tool_call = None
    resp.server_content = None
    return resp


def _make_response_with_tool_call(
    name: str,
    args: dict | None = None,
    call_id: str = "fc_001",
) -> MagicMock:
    """Build a mock LiveServerMessage carrying a tool (function) call."""
    fc = MagicMock()
    fc.name = name
    fc.args = args or {}
    fc.id = call_id

    resp = MagicMock()
    resp.data = MagicMock(spec_set=["data"])
    resp.data.data = None
    resp.tool_call = MagicMock()
    resp.tool_call.function_calls = [fc]
    resp.server_content = None
    return resp


def _make_response_turn_complete() -> MagicMock:
    """Build a mock LiveServerMessage signalling turn complete."""
    resp = MagicMock()
    resp.data = MagicMock(spec_set=["data"])
    resp.data.data = None
    resp.tool_call = None
    resp.server_content = MagicMock()
    resp.server_content.turn_complete = True
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handler() -> GeminiCallHandler:
    """Create a GeminiCallHandler with no real API connectivity."""
    return GeminiCallHandler(
        api_key="test-key",
        business_id=1,
        system_prompt="You are a test receptionist.",
        tool_schemas=[{"name": "test_tool", "description": "A test tool"}],
    )


@pytest.fixture(autouse=True)
def _mock_genai_client():
    """Mock the ``genai.Client`` so no real Gemini API calls are made.

    The mock session object is pre-configured with AsyncMock methods so
    that individual tests can override specific behaviours trivially.
    """
    with patch("voice_agent.gemini_client.genai.Client") as mock_cls:
        mock_instance = MagicMock()
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.send_realtime_input = AsyncMock()
        mock_session.send_tool_response = AsyncMock()
        # receive() is NOT set here — each test that needs it provides
        # its own return_value  (async generator).

        mock_instance.aio.live.connect = AsyncMock(return_value=mock_session)
        mock_cls.return_value = mock_instance

        yield mock_cls, mock_instance, mock_session


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    """GeminiCallHandler creation and configuration."""

    def test_create_handler_stores_config(self):
        """Constructor stores api_key, business_id, prompt, schemas,
        and creates a genai.Client."""
        handler = GeminiCallHandler(
            api_key="my-key",
            business_id=42,
            system_prompt="Hello",
            tool_schemas=[{"name": "tool_a"}],
        )
        assert handler._business_id == 42
        assert handler._system_prompt == "Hello"
        assert handler._tool_schemas == [{"name": "tool_a"}]
        assert handler._session is None
        assert handler._client is not None


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

class TestConnect:
    """Connecting to Gemini Live API."""

    @pytest.mark.asyncio
    async def test_connect_establishes_session(
        self, handler, _mock_genai_client,
    ):
        """After connect(), _session should not be None."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client

        await handler.connect()

        assert handler._session is not None
        mock_instance.aio.live.connect.assert_awaited_once()
        # Verify the model argument was passed
        call_kwargs = mock_instance.aio.live.connect.call_args.kwargs
        assert call_kwargs["model"] == LIVE_MODEL
        assert "config" in call_kwargs


# ---------------------------------------------------------------------------
# send_audio
# ---------------------------------------------------------------------------

class TestSendAudio:
    """Sending audio data to Gemini."""

    @pytest.mark.asyncio
    async def test_send_audio_before_connect_raises_error(self, handler):
        """Calling send_audio before connect raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot send audio"):
            await handler.send_audio(b"fake_audio")

    @pytest.mark.asyncio
    async def test_send_audio_after_connect_calls_send_realtime_input(
        self, handler, _mock_genai_client,
    ):
        """After connect, send_audio sends PCM to the session."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        await handler.connect()

        pcm_data = b"\x00\x01\x02\x03"
        await handler.send_audio(pcm_data)

        mock_session.send_realtime_input.assert_awaited_once_with(
            audio=pcm_data,
        )


# ---------------------------------------------------------------------------
# receive
# ---------------------------------------------------------------------------

class TestReceive:
    """Receiving messages from Gemini (async generator)."""

    @pytest.mark.asyncio
    async def test_receive_before_connect_raises_error(self, handler):
        """Calling receive() before connect raises RuntimeError."""
        gen = handler.receive()
        with pytest.raises(RuntimeError, match="Cannot receive"):
            async for _ in gen:
                pass

    @pytest.mark.asyncio
    async def test_receive_yields_audio_data(
        self, handler, _mock_genai_client,
    ):
        """receive() yields ("audio", pcm_bytes) when Gemini sends audio."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        sample_pcm = b"\x10\x20\x30\x40"
        response = _make_response_with_audio(sample_pcm)
        mock_session.receive = MagicMock(
            return_value=_async_iterator(response),
        )
        await handler.connect()

        results = []
        async for event_type, data in handler.receive():
            results.append((event_type, data))

        assert len(results) == 1
        assert results[0] == ("audio", sample_pcm)

    @pytest.mark.asyncio
    async def test_receive_yields_tool_call(
        self, handler, _mock_genai_client,
    ):
        """receive() yields ("tool_call", function_call) for tool calls."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        response = _make_response_with_tool_call(
            name="book_appointment",
            args={"date": "2025-07-10", "time": "10:00"},
            call_id="fc_001",
        )
        mock_session.receive = MagicMock(
            return_value=_async_iterator(response),
        )
        await handler.connect()

        results = []
        async for event_type, data in handler.receive():
            results.append((event_type, data))

        assert len(results) == 1
        event_type, fc = results[0]
        assert event_type == "tool_call"
        assert fc.name == "book_appointment"
        assert fc.id == "fc_001"
        assert fc.args["date"] == "2025-07-10"

    @pytest.mark.asyncio
    async def test_receive_yields_turn_complete(
        self, handler, _mock_genai_client,
    ):
        """receive() yields ("turn_complete", None) when turn finishes."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        response = _make_response_turn_complete()
        mock_session.receive = MagicMock(
            return_value=_async_iterator(response),
        )
        await handler.connect()

        results = []
        async for event_type, data in handler.receive():
            results.append((event_type, data))

        assert len(results) == 1
        assert results[0] == ("turn_complete", None)

    @pytest.mark.asyncio
    async def test_receive_skips_empty_data(
        self, handler, _mock_genai_client,
    ):
        """Messages without audio, tool_call, or turn_complete are skipped."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        empty = MagicMock()
        empty.data = MagicMock(spec_set=["data"])
        empty.data.data = None
        empty.tool_call = None
        empty.server_content = None

        mock_session.receive = MagicMock(
            return_value=_async_iterator(empty),
        )
        await handler.connect()

        results = []
        async for event_type, data in handler.receive():
            results.append((event_type, data))

        assert results == []

    @pytest.mark.asyncio
    async def test_receive_multiple_events(
        self, handler, _mock_genai_client,
    ):
        """Multiple events in sequence are all yielded."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        audio_resp = _make_response_with_audio(b"\xaa\xbb")
        tc_resp = _make_response_turn_complete()
        tool_resp = _make_response_with_tool_call("test_tool", {}, "fc_002")

        mock_session.receive = MagicMock(
            return_value=_async_iterator(audio_resp, tc_resp, tool_resp),
        )
        await handler.connect()

        results = []
        async for event_type, data in handler.receive():
            results.append((event_type, data))

        assert len(results) == 3
        assert results[0] == ("audio", b"\xaa\xbb")
        assert results[1] == ("turn_complete", None)
        assert results[2][0] == "tool_call"


# ---------------------------------------------------------------------------
# send_tool_response
# ---------------------------------------------------------------------------

class TestSendToolResponse:
    """Sending tool call results back to Gemini."""

    @pytest.mark.asyncio
    async def test_send_tool_response_before_connect_raises_error(
        self, handler,
    ):
        """Calling send_tool_response before connect raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot send tool response"):
            await handler.send_tool_response(
                [{"name": "test_tool", "response": {"result": "ok"}}],
            )

    @pytest.mark.asyncio
    async def test_send_tool_response_after_connect_calls_session(
        self, handler, _mock_genai_client,
    ):
        """After connect, send_tool_response calls session method."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        await handler.connect()

        responses = [
            {"id": "fc_001", "name": "test_tool", "response": {"result": "done"}},
        ]
        await handler.send_tool_response(responses)

        mock_session.send_tool_response.assert_awaited_once()
        call_args = mock_session.send_tool_response.call_args
        # Verify FunctionResponse objects were created
        fn_responses = call_args.kwargs.get("function_responses", [])
        assert len(fn_responses) == 1
        assert fn_responses[0].name == "test_tool"


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    """Gracefully closing the Gemini session."""

    @pytest.mark.asyncio
    async def test_disconnect_closes_session(
        self, handler, _mock_genai_client,
    ):
        """After connect, disconnect() calls session.close() and sets _session = None."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        await handler.connect()

        await handler.disconnect()

        mock_session.close.assert_awaited_once()
        assert handler._session is None

    @pytest.mark.asyncio
    async def test_disconnect_without_connect_does_nothing(self, handler):
        """Calling disconnect when session is None does nothing (no error)."""
        assert handler._session is None
        await handler.disconnect()  # Should not raise
        assert handler._session is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_close_error(
        self, handler, _mock_genai_client,
    ):
        """If session.close() raises, disconnect logs warning but still
        sets _session = None."""
        _mock_cls, mock_instance, mock_session = _mock_genai_client
        mock_session.close.side_effect = RuntimeError("Connection lost")
        await handler.connect()

        await handler.disconnect()  # Should not raise

        mock_session.close.assert_awaited_once()
        assert handler._session is None


# ---------------------------------------------------------------------------
# _build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    """Internal configuration builder."""

    def test_build_config_contains_expected_structure(
        self, handler,
    ):
        """_build_config() returns a LiveConnectConfig with AUDIO modality,
        Charon voice, system_instruction, and tools."""
        config = handler._build_config()

        assert config.response_modalities == ["AUDIO"]
        # Check voice
        assert config.speech_config is not None
        voice_config = config.speech_config.voice_config
        assert voice_config.prebuilt_voice_config.voice_name == VOICE_NAME
        # Check system instruction
        assert config.system_instruction == "You are a test receptionist."
        # Check tools — LiveConnectConfig wraps them as Tool objects
        assert config.tools is not None
        assert len(config.tools) == 1
        tool = config.tools[0]
        assert len(tool.function_declarations) == 1
        assert tool.function_declarations[0].name == "test_tool"
        assert tool.function_declarations[0].description == "A test tool"

    def test_build_config_without_tools_omits_tools(self):
        """When tool_schemas is empty, config.tools is None."""
        handler = GeminiCallHandler(
            api_key="key",
            business_id=1,
            system_prompt="Hi",
            tool_schemas=[],
        )
        config = handler._build_config()
        assert config.tools is None
        assert config.response_modalities == ["AUDIO"]
        assert config.system_instruction == "Hi"
