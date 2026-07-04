"""Google Calendar integration — OAuth, availability checks, event CRUD.

Provides :class:`GoogleCalendarClient` for runtime calendar operations
and standalone :func:`get_auth_url` / :func:`handle_oauth_callback` for
the web dashboard OAuth flow.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from cryptography.fernet import Fernet
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from voice_agent.config import Settings
from voice_agent.models import Business

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_URI = "https://oauth2.googleapis.com/token"
_DEFAULT_CALENDAR_ID = "primary"

DAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def _build_oauth_client_config(settings: Settings) -> dict[str, Any]:
    """Build a Google OAuth client config dict from *settings*."""
    return {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": TOKEN_URI,
            "redirect_uris": [settings.google_redirect_uri],
        }
    }


def get_auth_url(settings: Settings) -> str:
    """Generate a Google OAuth consent URL for the dashboard.

    Args:
        settings: Application settings with Google OAuth credentials.

    Returns:
        The full Google consent URL the business owner visits.
    """
    flow = Flow.from_client_config(
        _build_oauth_client_config(settings),
        scopes=SCOPES,
        redirect_uri=settings.google_redirect_uri,
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return auth_url


async def handle_oauth_callback(
    code: str,
    business_id: int,
    db_session: AsyncSession,
    settings: Settings,
) -> bool:
    """Exchange an OAuth authorization code for tokens and store them encrypted.

    Args:
        code: The authorization code from Google's redirect.
        business_id: The business to attach credentials to.
        db_session: An active database session.
        settings: Application settings.

    Returns:
        True on success, False on failure.
    """
    flow = Flow.from_client_config(
        _build_oauth_client_config(settings),
        scopes=SCOPES,
        redirect_uri=settings.google_redirect_uri,
    )
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        logger.error(
            "OAuth token exchange failed for business %d: %s", business_id, exc
        )
        return False

    creds = flow.credentials
    fernet = Fernet(settings.encryption_key.encode())
    encrypted = fernet.encrypt(creds.to_json().encode()).decode()

    stmt = select(Business).where(Business.id == business_id)
    result = await db_session.execute(stmt)
    business = result.scalar_one_or_none()
    if business is None:
        logger.error("Business %d not found during OAuth callback", business_id)
        return False

    business.google_calendar_encrypted = encrypted
    await db_session.commit()
    logger.info("OAuth tokens stored for business %d", business_id)
    return True


class GoogleCalendarClient:
    """Manages Google Calendar operations for a single business.

    Usage::

        client = GoogleCalendarClient(business_id=1, db_session_factory=factory, settings=settings)
        slots = await client.check_availability("2026-07-10", duration_minutes=30)
        event_id = await client.create_event("2026-07-10", "14:00", 30, "John", "+12025551234")
    """

    def __init__(
        self,
        business_id: int,
        db_session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.business_id = business_id
        self._session_factory = db_session_factory
        self.settings = settings
        self._calendar_id: str = _DEFAULT_CALENDAR_ID

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------

    async def get_credentials(self) -> Credentials | None:
        """Load, decrypt, and return Google OAuth credentials for this business.

        Automatically refreshes expired tokens and persists the refreshed
        credentials.

        Returns:
            Credentials if available, None otherwise.
        """
        async with self._session_factory() as session:
            stmt = select(Business).where(Business.id == self.business_id)
            result = await session.execute(stmt)
            business = result.scalar_one_or_none()

        if business is None or not business.google_calendar_encrypted:
            return None

        self._calendar_id = business.google_calendar_id or _DEFAULT_CALENDAR_ID

        fernet = Fernet(self.settings.encryption_key.encode())
        try:
            creds_json = fernet.decrypt(
                business.google_calendar_encrypted.encode()
            ).decode()
            creds = Credentials.from_json(creds_json)
        except Exception as exc:
            logger.error(
                "Failed to decrypt calendar credentials for business %d: %s",
                self.business_id,
                exc,
            )
            return None

        # Refresh if expired and a refresh token is available
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleAuthRequest())
                # Persist refreshed credentials
                encrypted = fernet.encrypt(creds.to_json().encode()).decode()
                async with self._session_factory() as session:
                    stmt = select(Business).where(Business.id == self.business_id)
                    result = await session.execute(stmt)
                    biz = result.scalar_one_or_none()
                    if biz is not None:
                        biz.google_calendar_encrypted = encrypted
                        await session.commit()
                logger.info(
                    "Refreshed calendar credentials for business %d", self.business_id
                )
            except Exception as exc:
                logger.error("Failed to refresh calendar credentials: %s", exc)
                return None

        return creds

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    async def check_availability(
        self,
        date: str,
        duration_minutes: int = 30,
    ) -> list[dict[str, str]]:
        """Check available time slots for *date*.

        Queries Google Calendar for busy periods within the business's
        configured hours, then returns the remaining free windows.

        Args:
            date: Date string in ``YYYY-MM-DD`` format.
            duration_minutes: Minimum slot duration in minutes.

        Returns:
            List of ``{"start": "HH:MM", "end": "HH:MM"}`` available
            time windows. Empty list if the business is closed or fully
            booked, or if the calendar is not configured.
        """
        creds = await self.get_credentials()
        if creds is None:
            logger.warning("Calendar not configured for business %d", self.business_id)
            return []

        # Load business hours
        async with self._session_factory() as session:
            stmt = select(Business).where(Business.id == self.business_id)
            result = await session.execute(stmt)
            business = result.scalar_one_or_none()

        if business is None:
            return []

        hours = business.business_hours or {}
        tz_name = business.timezone or "UTC"

        # Determine day of week
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        day_name = DAY_NAMES[parsed_date.weekday()]
        day_hours = hours.get(day_name)

        if not day_hours or "open" not in day_hours or "close" not in day_hours:
            logger.info(
                "Business %d is closed on %s (%s)", self.business_id, date, day_name
            )
            return []

        open_time = day_hours["open"]
        close_time = day_hours["close"]

        import zoneinfo

        tz = zoneinfo.ZoneInfo(tz_name)
        day_start = datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            tzinfo=tz,
        )
        open_dt = day_start + timedelta(
            hours=int(open_time.split(":")[0]),
            minutes=int(open_time.split(":")[1]),
        )
        close_dt = day_start + timedelta(
            hours=int(close_time.split(":")[0]),
            minutes=int(close_time.split(":")[1]),
        )

        # Query Google Calendar for busy slots
        try:
            service = await asyncio_to_thread(
                build("calendar", "v3", credentials=creds)
            )
            events_result = await asyncio_to_thread(
                service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=open_dt.isoformat(),
                    timeMax=close_dt.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute
            )
        except Exception as exc:
            logger.error(
                "Google Calendar API error for business %d: %s", self.business_id, exc
            )
            return []

        busy_slots: list[dict[str, datetime]] = []
        for event in events_result.get("items", []):
            start_str = event["start"].get("dateTime", event["start"].get("date"))
            end_str = event["end"].get("dateTime", event["end"].get("date"))
            try:
                busy_start = datetime.fromisoformat(start_str)
                busy_end = datetime.fromisoformat(end_str)
                # Clamp to business hours
                if busy_end <= open_dt or busy_start >= close_dt:
                    continue
                busy_slots.append(
                    {
                        "start": max(busy_start, open_dt),
                        "end": min(busy_end, close_dt),
                    }
                )
            except (ValueError, TypeError):
                continue

        # Sort busy slots and merge overlaps
        busy_slots.sort(key=lambda x: x["start"])
        merged_busy: list[dict[str, datetime]] = []
        for slot in busy_slots:
            if merged_busy and slot["start"] <= merged_busy[-1]["end"]:
                merged_busy[-1]["end"] = max(merged_busy[-1]["end"], slot["end"])
            else:
                merged_busy.append(slot)

        # Generate free windows
        free_windows: list[dict[str, str]] = []
        cursor = open_dt
        for busy in merged_busy:
            if busy["start"] - cursor >= timedelta(minutes=duration_minutes):
                free_windows.append(
                    {
                        "start": cursor.strftime("%H:%M"),
                        "end": busy["start"].strftime("%H:%M"),
                    }
                )
            cursor = max(cursor, busy["end"])

        # Remaining time after last busy slot
        if close_dt - cursor >= timedelta(minutes=duration_minutes):
            free_windows.append(
                {
                    "start": cursor.strftime("%H:%M"),
                    "end": close_dt.strftime("%H:%M"),
                }
            )

        return free_windows

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    async def create_event(
        self,
        date: str,
        time: str,
        duration_minutes: int,
        customer_name: str,
        customer_phone: str,
        description: str = "",
    ) -> str:
        """Create a Google Calendar event and return its event ID.

        Args:
            date: ``YYYY-MM-DD``.
            time: ``HH:MM`` 24-hour format.
            duration_minutes: Length of the appointment in minutes.
            customer_name: Caller's full name.
            customer_phone: Caller's phone number (E.164).
            description: Optional notes.

        Returns:
            Google Calendar event ID string.

        Raises:
            RuntimeError: If credentials are not available.
            Exception: If the Calendar API call fails.
        """
        creds = await self.get_credentials()
        if creds is None:
            raise RuntimeError("Calendar not configured — cannot create events")

        # Load business for timezone
        async with self._session_factory() as session:
            stmt = select(Business).where(Business.id == self.business_id)
            result = await session.execute(stmt)
            business = result.scalar_one_or_none()

        tz_name = business.timezone if business and business.timezone else "UTC"
        import zoneinfo

        tz = zoneinfo.ZoneInfo(tz_name)
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        hour, minute = int(time.split(":")[0]), int(time.split(":")[1])
        start_dt = datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            hour,
            minute,
            tzinfo=tz,
        )
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        event_body: dict[str, Any] = {
            "summary": f"Appointment with {customer_name}",
            "description": (
                f"Customer: {customer_name}\n"
                f"Phone: {customer_phone}\n"
                f"Notes: {description or 'None'}"
            ),
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": tz_name,
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": tz_name,
            },
        }

        service = await asyncio_to_thread(build("calendar", "v3", credentials=creds))
        created = await asyncio_to_thread(
            service.events()
            .insert(
                calendarId=self._calendar_id,
                body=event_body,
            )
            .execute
        )
        event_id: str = created["id"]
        logger.info(
            "Created calendar event %s for business %d: %s at %s",
            event_id,
            self.business_id,
            date,
            time,
        )
        return event_id

    async def cancel_event(self, google_event_id: str) -> bool:
        """Cancel (delete) a Google Calendar event.

        Args:
            google_event_id: The event ID returned by :meth:`create_event`.

        Returns:
            True if the event was deleted, False on failure.
        """
        creds = await self.get_credentials()
        if creds is None:
            logger.warning("Cannot cancel event — calendar not configured")
            return False

        try:
            service = await asyncio_to_thread(
                build("calendar", "v3", credentials=creds)
            )
            await asyncio_to_thread(
                service.events()
                .delete(
                    calendarId=self._calendar_id,
                    eventId=google_event_id,
                )
                .execute
            )
            logger.info("Cancelled calendar event %s", google_event_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel event %s: %s", google_event_id, exc)
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def asyncio_to_thread(callable, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous callable in a thread via asyncio.

    The ``googleapiclient`` library is synchronous and would block the
    event loop if called directly.
    """
    import asyncio

    return await asyncio.to_thread(callable, *args, **kwargs)
