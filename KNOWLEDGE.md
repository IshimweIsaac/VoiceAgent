# VoiceAgent — Knowledge Base

## Architecture
- **Voice Pipeline**: Caller → Twilio (PSTN) → Twilio Media Streams (WebSocket, µ-law 8kHz) → FastAPI → Audio Converter → Gemini Live API (PCM 16/24kHz) → Tool Dispatch → Response
- **Tool Pattern**: Each tool in `voice_agent/tools/` exposes `TOOL_SCHEMA` dict + `async def handler(...)` function, auto-registered at import time
- **Multi-Tenant**: Business slug in WebSocket URL path routes calls to correct config

## Key Constants
| Constant | Value | File |
|----------|-------|------|
| `SEND_SAMPLE_RATE` | 16000 | gemini_client.py |
| `RECEIVE_SAMPLE_RATE` | 24000 | gemini_client.py |
| `LIVE_MODEL` | `models/gemini-3.1-flash-live-preview` | gemini_client.py |
| `VOICE_NAME` | `Charon` | gemini_client.py |
| `_MAX_TOOL_DESC` | 400 | tool_registry.py |
| `_MAX_PROP_DESC` | 140 | tool_registry.py |
| `_MAX_TOOLS` | 20 | tool_registry.py |

## Gotchas
- Python 3.14 removed `audioop` — implement G.711 µ-law codec using CPython reference algorithm
- `passlib` is unmaintained and incompatible with `bcrypt>=5.0` — use `bcrypt.hashpw()` directly
- Starlette 1.3.1 requires `TemplateResponse(request, name, context)` — first positional arg is the Request object
- Tool handlers must be async (call external APIs) — tool_registry.dispatch() handles both sync and async
- Gemini Live API sessions are per-call — create and tear down for every phone call

## Disoveries
- [2026-07-04] build discovered: Python 3.14 removed audioop — implemented µ-law codec manually
- [2026-07-04] build discovered: Google Gemini Flash Live (Charon voice) works well for phone receptionist use case
- [2026-07-04] build discovered: Twilio Media Streams sends bidirectional audio as base64 µ-law via WebSocket JSON messages

## Open Questions
- Should we switch from Google Calendar API to Calendly API for simpler booking?
- Should we add PostgreSQL support before production deployment?
