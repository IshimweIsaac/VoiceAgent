"""Gemini Live API session manager — one session per phone call.

Adapted from the Lucas project's ``LucasLive`` class pattern. Manages
a bidirectional audio + tool-call session with the Gemini Live API.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai import types

SEND_SAMPLE_RATE = 16000  # Gemini expects 16 kHz PCM input
RECEIVE_SAMPLE_RATE = 24000  # Gemini sends 24 kHz PCM output
CHUNK_SIZE = 1024
LIVE_MODEL = "models/gemini-3.1-flash-live-preview"
VOICE_NAME = "Charon"

logger = logging.getLogger(__name__)


class GeminiCallHandler:
    """Manages one Gemini Live session for one phone call.

    Usage::

        handler = GeminiCallHandler(api_key, business_id, prompt, schemas)
        await handler.connect()
        await handler.send_audio(pcm_16khz_bytes)
        async for event_type, payload in handler.receive():
            ...
        await handler.disconnect()
    """

    def __init__(
        self,
        api_key: str,
        business_id: int,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        """Store config and create the genai client.

        Args:
            api_key: Google Gemini API key.
            business_id: Business ID for scoping (used in logging).
            system_prompt: Business-specific system instruction.
            tool_schemas: List of trimmed tool schemas for function declarations.
        """
        self._business_id = business_id
        self._system_prompt = system_prompt
        self._tool_schemas = tool_schemas
        self._session: types.LiveSession | None = None

        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Gemini Live API.

        Creates the Live session with the pre-configured system prompt,
        voice settings, and tool declarations.
        """
        config = self._build_config()
        logger.info(
            "GeminiCallHandler[%d] connecting to %s ...",
            self._business_id,
            LIVE_MODEL,
        )
        self._session = await self._client.aio.live.connect(
            model=LIVE_MODEL,
            config=config,
        )
        logger.info("GeminiCallHandler[%d] session established", self._business_id)

    async def send_audio(self, pcm_16khz_bytes: bytes) -> None:
        """Send caller's audio to Gemini.

        Args:
            pcm_16khz_bytes: PCM s16le audio at 16 kHz.

        Raises:
            RuntimeError: If the session is not connected.
        """
        if self._session is None:
            raise RuntimeError("Cannot send audio — session not connected")

        try:
            await self._session.send_realtime_input(audio=pcm_16khz_bytes)
        except Exception as exc:
            logger.error(
                "GeminiCallHandler[%d] send_audio failed: %s",
                self._business_id,
                exc,
            )
            raise

    async def receive(
        self,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Receive messages from Gemini.

        Yields ``(type, data)`` tuples:

        * ``("audio", pcm_bytes)`` — PCM s16le audio at 24 kHz to play to caller.
        * ``("tool_call", function_call)`` — A tool function call from Gemini.
        * ``("turn_complete", None)`` — Gemini finished its speaking turn.

        Raises:
            RuntimeError: If the session is not connected.
        """
        if self._session is None:
            raise RuntimeError("Cannot receive — session not connected")

        async for response in self._session.receive():
            # Audio data from Gemini
            if response.data and response.data.data:
                yield ("audio", response.data.data)

            # Tool call from Gemini
            if response.tool_call:
                for fc in response.tool_call.function_calls:
                    yield ("tool_call", fc)

            # Turn complete signal
            if response.server_content and response.server_content.turn_complete:
                yield ("turn_complete", None)

    async def send_tool_response(self, responses: list[dict[str, Any]]) -> None:
        """Send tool call results back to Gemini.

        Args:
            responses: List of dicts with keys ``name``, ``response``
                       (and optionally ``id``) matching the function call.

        Raises:
            RuntimeError: If the session is not connected.
        """
        if self._session is None:
            raise RuntimeError("Cannot send tool response — session not connected")

        fn_responses = [
            types.FunctionResponse(
                id=resp.get("id", ""),
                name=resp["name"],
                response={"result": resp["response"]["result"]},
            )
            for resp in responses
        ]

        try:
            await self._session.send_tool_response(
                function_responses=fn_responses,
            )
        except Exception as exc:
            logger.error(
                "GeminiCallHandler[%d] send_tool_response failed: %s",
                self._business_id,
                exc,
            )
            raise

    async def disconnect(self) -> None:
        """Close the Gemini session gracefully."""
        if self._session is not None:
            try:
                await self._session.close()
                logger.info(
                    "GeminiCallHandler[%d] session closed",
                    self._business_id,
                )
            except Exception as exc:
                logger.warning(
                    "GeminiCallHandler[%d] error closing session: %s",
                    self._business_id,
                    exc,
                )
            finally:
                self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_config(self) -> types.LiveConnectConfig:
        """Build the LiveConnectConfig for this session.

        Includes system instruction, audio-only response modality,
        voice configuration, and tool declarations.
        """
        tool_declarations: list[dict[str, Any]] | None = None
        if self._tool_schemas:
            tool_declarations = [{"function_declarations": self._tool_schemas}]

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE_NAME,
                    ),
                ),
            ),
            system_instruction=self._system_prompt,
            tools=tool_declarations,
        )
