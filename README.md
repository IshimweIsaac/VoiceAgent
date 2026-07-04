# VoiceAgent -- AI Phone Receptionist for Small Businesses

> A production-grade AI voice receptionist that answers inbound calls, books
> appointments, answers business FAQs, and escalates to humans -- powered by
> **Google Gemini Live API** and **Twilio**.

[![Tests](https://img.shields.io/badge/tests-172%20passing-brightgreen)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-68%25-yellowgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Demo

<!-- Replace with a screenshot, GIF, or hosted demo URL -->
*A live demo is coming soon. Reach out for a walkthrough.*

---

## Features

| Capability | Detail |
|---|---|
| **Inbound Call Handling** | Twilio Media Streams WebSocket streams real-time audio from PSTN calls |
| **Natural AI Voice** | Google Gemini Live API with Charon voice -- sub-500ms response latency |
| **Appointment Booking** | Google Calendar API integration with availability checking and event creation |
| **FAQ Knowledge Base** | Business owners upload Q&A pairs; the AI searches and answers from them |
| **Human Escalation** | Transfers calls to a real person when the caller asks or the AI cannot help |
| **SMS Confirmations** | Automatic appointment confirmation texts via Twilio SMS |
| **Web Dashboard** | FastAPI + Jinja2 dashboard for managing settings, FAQs, call history, and calendar |
| **Multi-Tenant** | Single server serves multiple businesses, each with isolated data |
| **Comprehensive Tests** | 172 tests with 68% code coverage across all modules |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Python 3.11+, FastAPI, Uvicorn |
| **AI Engine** | Google Gemini Live API (`gemini-3.1-flash-live-preview`) |
| **Voice** | Charon -- Gemini's natural, expressive voice |
| **Telephony** | Twilio (Media Streams, SMS, PSTN) |
| **Database** | SQLite via SQLAlchemy 2.0 (async) |
| **Scheduling** | Google Calendar API (OAuth 2.0, encrypted token storage) |
| **Audio Pipeline** | Custom µ-law (G.711) ↔ PCM resampling via NumPy |
| **Web Dashboard** | FastAPI + Jinja2 templates + CSS |
| **Auth** | bcrypt password hashing + session-based authentication |
| **Secrets** | Fernet (symmetric encryption) for OAuth tokens at rest |
| **Testing** | pytest, pytest-asyncio |

---

## Architecture

```

 Caller Twilio VoiceAgent Gemini Live
 (Phone) (PSTN) (FastAPI) API (Charon)


 Tool Registry

 * lookup_faq SQLite (FAQ KB)
 * check_availability Google Calendar
 * book_appointment Google Calendar
 * send_sms_confirmation Twilio SMS
 * transfer_to_human PSTN Transfer

```

### Call Flow

1. **Inbound Call** -- A customer dials the business's Twilio number.
2. **Twilio Webhook** -- Twilio POSTs to `/twilio/incoming`, which returns TwiML with a `<Connect><Stream>` directing the audio to VoiceAgent's WebSocket.
3. **WebSocket Connection** -- Twilio opens a Media Stream WebSocket at `/media/{business_slug}`. VoiceAgent looks up the business, creates a `CallManager`, and initializes a Gemini Live session.
4. **Bidirectional Audio** -- Caller audio flows: Twilio µ-law @ 8kHz → decode → resample to 16kHz PCM → Gemini Live API. Gemini responds with 24kHz PCM → resample to 8kHz → µ-law encode → Twilio.
5. **Tool Execution** -- When Gemini calls a tool (e.g., `book_appointment`), the `CallManager` dispatches it through the `ToolRegistry`. Results are streamed back to Gemini, which speaks them to the caller.
6. **Call End** -- On hangup, VoiceAgent records the transcript, outcome, and duration, then gracefully disconnects the Gemini session.

### Audio Pipeline

```
TWILIO → Server: µ-law @ 8kHz (base64) → PCM s16le @ 8kHz → resample → PCM @ 16kHz → Gemini
Server → TWILIO: Gemini PCM @ 24kHz → resample → PCM @ 8kHz → µ-law → base64 → Twilio media
```

The µ-law (G.711) codec is implemented from scratch using the standard algorithm -- no external audio libraries beyond NumPy for linear resampling.

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [Twilio account](https://www.twilio.com) with a phone number
- A [Google Gemini API key](https://aistudio.google.com/)
- A Google Cloud project with the Calendar API enabled (for appointment booking)
- [ngrok](https://ngrok.com/) (for local testing)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/IshimweIsaac/VoiceAgent.git
cd VoiceAgent

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see Configuration below)

# 5. Run the one-time setup script (optional, does steps 2-4)
bash setup.sh
```

### Configuration

Copy `.env.example` to `.env` and fill in all required values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | -- | Google Gemini API key |
| `TWILIO_ACCOUNT_SID` | Yes | -- | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Yes | -- | Twilio Auth Token |
| `SECRET_KEY` | Yes | -- | 64-char random session secret |
| `ENCRYPTION_KEY` | Yes | -- | Fernet key for OAuth token encryption |
| `GOOGLE_CLIENT_ID` | Calendar | -- | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Calendar | -- | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | Calendar | -- | OAuth callback URL |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server port |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./voice_agent.db` | Database connection |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DEBUG` | No | `false` | Enable debug mode |
| `MAX_CONCURRENT_CALLS` | No | `5` | Max simultaneous Gemini sessions |

Generate secure keys:

```bash
# SECRET_KEY (64-char URL-safe)
python -c "import secrets; print(secrets.token_urlsafe(64))"

# ENCRYPTION_KEY (32-byte base64 Fernet key)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Expose with ngrok

```bash
ngrok http 8000
```

Update your Twilio phone number's voice webhook in the [Twilio Console](https://console.twilio.com) to point to:

```
https://<your-ngrok-subdomain>.ngrok.app/twilio/incoming
```

---

## Project Structure

```
VoiceAgent/
 main.py # FastAPI entry point, lifespan, middleware
 setup.sh # One-click environment setup
 requirements.txt # Python dependencies
 .env.example # Environment variable template

 voice_agent/ # Core application library
 config.py # Pydantic Settings (env var loading)
 database.py # Async SQLAlchemy engine + session management
 models.py # ORM: Business, User, FAQ, Call, Appointment
 audio_converter.py # µ-law ↔ PCM conversion (G.711 algorithm)
 gemini_client.py # Gemini Live API session handler
 call_manager.py # Per-call lifecycle orchestrator
 twilio_handler.py # Twilio HTTP webhooks + Media Stream WebSocket
 twilio_client.py # Async Twilio REST wrapper (SMS, calls)
 google_calendar.py # Google Calendar OAuth + availability + events
 system_prompt.py # Business-specific prompt builder
 tool_registry.py # Tool registration and dispatch system

 tools/ # Individual tools (auto-register on import)
 appointment.py # check_availability, book_appointment
 faq.py # lookup_faq -- keyword-scored KB search
 sms.py # send_sms_confirmation
 transfer.py # transfer_to_human

 web/ # Business owner web dashboard
 auth.py # Session auth (login, register, logout)
 routes.py # Dashboard routes (settings, FAQs, calls)
 templates/ # Jinja2 HTML templates (9 pages)
 static/ # CSS stylesheets

 tests/ # Test suite
 test_api.py # API endpoint tests
 test_audio_converter.py# µ-law/PCM conversion edge cases
 test_call_manager.py # Call lifecycle orchestration
 test_gemini_client.py # Gemini session management
 test_models.py # ORM model validation
 test_system_prompt.py # Prompt building logic
 test_tool_registry.py # Tool registration + dispatch
 test_tools.py # Individual tool handlers

 scripts/ # Utility scripts
```

---

## API Reference

### HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/login` | Dashboard login page |
| `POST` | `/login` | Login form submission |
| `GET` | `/register` | Business registration page |
| `POST` | `/register` | Create new business account |
| `GET` | `/logout` | End session |
| `GET` | `/dashboard` | Dashboard overview with stats |
| `GET` | `/dashboard/settings` | Business settings form |
| `POST` | `/dashboard/settings` | Save settings |
| `GET` | `/dashboard/faqs` | FAQ management page |
| `POST` | `/dashboard/faqs/add` | Add FAQ entry |
| `POST` | `/dashboard/faqs/{id}/edit` | Edit FAQ entry |
| `POST` | `/dashboard/faqs/{id}/delete` | Delete FAQ entry |
| `GET` | `/dashboard/calls` | Call history (paginated) |
| `GET` | `/dashboard/calls/{id}` | Call detail + transcript |
| `GET` | `/dashboard/calendar/status` | Calendar connection status |
| `GET` | `/dashboard/calendar/auth` | Google OAuth consent screen |
| `GET` | `/dashboard/calendar/callback` | OAuth callback handler |
| `POST` | `/twilio/incoming` | **Twilio webhook** -- inbound call |
| `POST` | `/twilio/status` | **Twilio webhook** -- call status |

### WebSocket

| Path | Description |
|---|---|
| `/media/{business_slug}` | Twilio Media Stream -- bidirectional audio + tool calls |

---

## Tools

VoiceAgent exposes five tools that Gemini can call. Each tool is a self-contained module that auto-registers on import.

| Tool | Description | Parameters |
|---|---|---|
| `lookup_faq` | Search the FAQ knowledge base | `query` (string) |
| `check_availability` | Check open appointment slots | `date` (YYYY-MM-DD) |
| `book_appointment` | Book an appointment and create a calendar event | `date`, `time`, `customer_name`, `customer_phone`, `duration_minutes` |
| `send_sms_confirmation` | Send SMS after booking | `phone`, `message` |
| `transfer_to_human` | Escalate to a human operator | `reason` (optional) |

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v --tb=line

# Run with coverage report
coverage run -m pytest tests/ && coverage report
```

**Test Results:** 172 tests passing across 8 test files with 68% overall code coverage.

**Test Coverage by Module:**

| Module | Coverage |
|---|---|
| `call_manager.py` | 96% |
| `gemini_client.py` | 91% |
| `tool_registry.py` | 98% |
| `system_prompt.py` | 100% |
| `models.py` | 100% |
| `config.py` | 100% |
| `tools/` (combined) | 84% |
| `web/auth.py` | 95% |
| `web/routes.py` | 67% |
| `audio_converter.py` | N/A (manual audio) |
| **Overall** | **68%** |

---

## Extending

### Adding a New Tool

1. Create `voice_agent/tools/my_tool.py`
2. Define a `TOOL_SCHEMA` dict following the Gemini function declaration format
3. Write a handler accepting `(parameters, business_id, db_session, **kwargs)`
4. Call `register("tool_name", TOOL_SCHEMA, handler)` at module level
5. Import the module in `voice_agent/tools/__init__.py`

```python
TOOL_SCHEMA = {
 "name": "my_tool",
 "description": "What this tool does",
 "parameters": {
 "type": "OBJECT",
 "properties": {
 "param1": {"type": "STRING", "description": "..."},
 },
 "required": ["param1"],
 },
}

async def handler(parameters, business_id, db_session, **kwargs):
 # Your logic here
 return "Result string for Gemini to speak"

register("my_tool", TOOL_SCHEMA, handler)
```

---

## Deployment

### Local Testing (ngrok)

```bash
ngrok http 8000
```

Set your Twilio phone number's voice webhook URL to:
`https://<ngrok-subdomain>.ngrok.app/twilio/incoming`

### Production

Production deployments require:

- **ASGI Server**: Uvicorn with Gunicorn workers or Uvicorn workers for concurrency
- **Process Supervision**: systemd service for auto-restart
- **Reverse Proxy**: nginx with SSL termination
- **Database**: Consider migrating to PostgreSQL for higher reliability (swap the `DATABASE_URL`)

---

## Author

**Isaac Ishimwe** -- Kigali, Rwanda

- GitHub: [@IshimweIsaac](https://github.com/IshimweIsaac)

---

## License

MIT -- see [LICENSE](LICENSE) for details.