"""Application configuration — loads all env vars via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from environment variables (with .env file support)."""

    # --- Google Gemini API ---
    gemini_api_key: str

    # --- Twilio ---
    twilio_account_sid: str
    twilio_auth_token: str

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite+aiosqlite:///./voice_agent.db"
    secret_key: str
    encryption_key: str

    # --- Google OAuth (Calendar) ---
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""

    # --- Optional ---
    log_level: str = "INFO"
    debug: bool = False
    max_concurrent_calls: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
