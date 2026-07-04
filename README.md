# VoiceAgent : AI Phone Receptionist

Standalone AI phone receptionist for small businesses. Handles inbound calls,
books appointments, answers business FAQs, and escalates to humans.

Built with FastAPI, Twilio Media Streams, and Google Gemini Live API.
172 tests passing. Built by Isaac Ishimwe.

:-

## Call Flow

Caller -> Twilio (PSTN) -> Media Streams WebSocket -> FastAPI -> Audio Converter
-> Gemini Live API -> Tool Dispatch -> Response -> Twilio -> Caller

Audio pipeline: Twilio u-law 8kHz -> PCM 8kHz -> resample 16kHz -> Gemini
Gemini -> PCM 24kHz -> resample 8kHz -> u-law -> Twilio

:-

## Features

- Inbound call handling via Twilio Media Streams WebSocket
- Natural voice AI using Gemini Live API (Charon voice, <500ms latency)
- Appointment booking via Google Calendar API with availability checking
- FAQ knowledge base (business owner uploads Q&A pairs)
- Human escalation (transfers call to real phone number)
- SMS confirmations via Twilio SMS
- Web dashboard for business configuration (FastAPI + Jinja2)
- Multi-tenant: multiple businesses on a single server
- Session-based auth with bcrypt
- Fernet encryption for OAuth tokens at rest

:-

## Stack

- Python 3.11+, FastAPI, Uvicorn
- Google Gemini Live API (gemini-3.1-flash-live-preview)
- Twilio (Media Streams, SMS, Voice API)
- SQLite via SQLAlchemy (async)
- Google Calendar API (OAuth 2.0)
- Jinja2 templates, CSS
- bcrypt, cryptography (Fernet)
- pytest-asyncio, httpx

## Development Tools

Cursor, Claude Code, Windsurf, OpenCode, VS Code, Git, pytest, Ruff

:-

## Tools

5 registered tools that the AI can call during a conversation:

- check_availability : Check available time slots for a given date
- book_appointment : Create a Google Calendar event and DB record
- lookup_faq : Search FAQ knowledge base with relevance scoring
- transfer_to_human : Return the business transfer number
- send_sms_confirmation : Send SMS via Twilio

:-

## Testing

pytest tests/ -x :tb=line -q

172 tests across 9 test modules. Coverage: 68% overall,
91-100% on core modules (models, config, audio converter, tool registry).

:-

github.com/IshimweIsaac/VoiceAgent
