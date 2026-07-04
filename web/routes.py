"""Dashboard routes for the web interface.

All routes require authentication via get_current_business
(except login/register which are handled in auth.py).
"""

from __future__ import annotations

import json
import logging
from datetime import date

from fastapi import APIRouter, Depends, Form, Path, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voice_agent.config import Settings
from voice_agent.database import get_db
from voice_agent.models import Appointment, Business, Call, FAQ
from web.auth import flash_message, get_current_business, get_flash_messages

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory="web/templates")

PAGE_SIZE = 20


# ---------------------------------------------------------------------------
# Dashboard home
# ---------------------------------------------------------------------------


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_overview(
    request: Request,
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Show dashboard with stats overview (calls today, appointments, FAQ count)."""
    today = date.today()

    # Calls today
    result = await db.execute(
        select(func.count(Call.id)).where(
            Call.business_id == business.id,
            func.date(Call.created_at) == today,
        )
    )
    calls_today = result.scalar() or 0

    # Total appointments
    result = await db.execute(
        select(func.count(Appointment.id)).where(
            Appointment.business_id == business.id,
        )
    )
    appointments_booked = result.scalar() or 0

    # Enabled FAQ count
    result = await db.execute(
        select(func.count(FAQ.id)).where(
            FAQ.business_id == business.id,
            FAQ.enabled == True,  # noqa: E712
        )
    )
    faq_count = result.scalar() or 0

    # Calendar connection status
    calendar_connected = bool(business.google_calendar_encrypted)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "business": business,
            "calls_today": calls_today,
            "appointments_booked": appointments_booked,
            "faq_count": faq_count,
            "calendar_connected": calendar_connected,
            "messages": get_flash_messages(request),
        },
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@router.get("/dashboard/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    business: Business = Depends(get_current_business),
) -> HTMLResponse:
    """Show the business settings form."""
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "business": business,
            "messages": get_flash_messages(request),
        },
    )


@router.post("/dashboard/settings", response_class=HTMLResponse)
async def settings_save(
    request: Request,
    name: str = Form(...),
    greeting_message: str = Form(...),
    business_hours: str = Form("{}"),
    timezone: str = Form("America/New_York"),
    transfer_phone_number: str = Form(""),
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Save business settings."""
    # Parse business_hours JSON
    try:
        hours_dict = json.loads(business_hours) if business_hours else {}
    except json.JSONDecodeError:
        hours_dict = {}

    business.name = name.strip()
    business.greeting_message = greeting_message.strip()
    business.business_hours = hours_dict
    business.timezone = timezone.strip()
    business.transfer_phone_number = transfer_phone_number.strip()
    await db.flush()

    flash_message(request, "success", "Settings saved successfully.")
    return RedirectResponse("/dashboard/settings", status_code=303)


# ---------------------------------------------------------------------------
# FAQs
# ---------------------------------------------------------------------------


@router.get("/dashboard/faqs", response_class=HTMLResponse)
async def faqs_page(
    request: Request,
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Show FAQ list with add form."""
    result = await db.execute(
        select(FAQ)
        .where(FAQ.business_id == business.id)
        .order_by(FAQ.created_at.desc())
    )
    faqs = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "faqs.html",
        {
            "business": business,
            "faqs": faqs,
            "messages": get_flash_messages(request),
        },
    )


@router.post("/dashboard/faqs/add")
async def faq_add(
    request: Request,
    question: str = Form(...),
    answer: str = Form(...),
    category: str = Form("general"),
    keywords: str = Form(""),
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Add a new FAQ entry."""
    if not question.strip() or not answer.strip():
        flash_message(request, "error", "Question and answer are required.")
        return RedirectResponse("/dashboard/faqs", status_code=303)

    faq = FAQ(
        business_id=business.id,
        question=question.strip(),
        answer=answer.strip(),
        category=category.strip() or "general",
        keywords=keywords.strip(),
    )
    db.add(faq)
    await db.flush()
    flash_message(request, "success", "FAQ added successfully.")
    return RedirectResponse("/dashboard/faqs", status_code=303)


@router.post("/dashboard/faqs/{faq_id}/edit")
async def faq_edit(
    request: Request,
    faq_id: int = Path(...),
    question: str = Form(...),
    answer: str = Form(...),
    category: str = Form("general"),
    keywords: str = Form(""),
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Edit an existing FAQ entry."""
    result = await db.execute(
        select(FAQ).where(FAQ.id == faq_id, FAQ.business_id == business.id)
    )
    faq = result.scalar_one_or_none()
    if faq is None:
        flash_message(request, "error", "FAQ not found.")
        return RedirectResponse("/dashboard/faqs", status_code=303)

    faq.question = question.strip()
    faq.answer = answer.strip()
    faq.category = category.strip() or "general"
    faq.keywords = keywords.strip()
    await db.flush()
    flash_message(request, "success", "FAQ updated successfully.")
    return RedirectResponse("/dashboard/faqs", status_code=303)


@router.post("/dashboard/faqs/{faq_id}/delete")
async def faq_delete(
    request: Request,
    faq_id: int = Path(...),
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Delete a FAQ entry."""
    result = await db.execute(
        select(FAQ).where(FAQ.id == faq_id, FAQ.business_id == business.id)
    )
    faq = result.scalar_one_or_none()
    if faq is None:
        flash_message(request, "error", "FAQ not found.")
        return RedirectResponse("/dashboard/faqs", status_code=303)

    await db.delete(faq)
    await db.flush()
    flash_message(request, "success", "FAQ deleted successfully.")
    return RedirectResponse("/dashboard/faqs", status_code=303)


# ---------------------------------------------------------------------------
# Call History
# ---------------------------------------------------------------------------


@router.get("/dashboard/calls", response_class=HTMLResponse)
async def calls_page(
    request: Request,
    page: int = Query(1, ge=1),
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Show paginated call history."""
    # Total count
    result = await db.execute(
        select(func.count(Call.id)).where(Call.business_id == business.id)
    )
    total_calls = result.scalar() or 0

    total_pages = max(1, (total_calls + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    offset = (page - 1) * PAGE_SIZE

    result = await db.execute(
        select(Call)
        .where(Call.business_id == business.id)
        .order_by(Call.created_at.desc())
        .offset(offset)
        .limit(PAGE_SIZE)
    )
    calls = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "calls.html",
        {
            "business": business,
            "calls": calls,
            "page": page,
            "total_pages": total_pages,
            "total_calls": total_calls,
            "messages": get_flash_messages(request),
        },
    )


@router.get("/dashboard/calls/{call_id}", response_class=HTMLResponse)
async def call_detail_page(
    request: Request,
    call_id: int = Path(...),
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Show a single call's detail with full transcript."""
    result = await db.execute(
        select(Call).where(Call.id == call_id, Call.business_id == business.id)
    )
    call = result.scalar_one_or_none()
    if call is None:
        flash_message(request, "error", "Call not found.")
        return RedirectResponse("/dashboard/calls", status_code=303)

    return templates.TemplateResponse(
        request,
        "call_detail.html",
        {
            "business": business,
            "call": call,
            "messages": get_flash_messages(request),
        },
    )


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


@router.get("/dashboard/calendar/status", response_class=HTMLResponse)
async def calendar_status_page(
    request: Request,
    business: Business = Depends(get_current_business),
) -> HTMLResponse:
    """Show Google Calendar connection status."""
    connected = bool(business.google_calendar_encrypted)
    return templates.TemplateResponse(
        request,
        "calendar_status.html",
        {
            "business": business,
            "connected": connected,
            "messages": get_flash_messages(request),
        },
    )


@router.get("/dashboard/calendar/auth")
async def calendar_auth(
    request: Request,
    business: Business = Depends(get_current_business),
) -> RedirectResponse:
    """Redirect to Google OAuth consent screen to connect Google Calendar."""
    settings = Settings()  # type: ignore[call-arg]
    if not settings.google_client_id:
        flash_message(
            request,
            "error",
            "Google Calendar is not configured. Set GOOGLE_CLIENT_ID in .env.",
        )
        return RedirectResponse("/dashboard/calendar/status", status_code=303)

    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.google_redirect_uri],
            }
        },
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    flow.redirect_uri = settings.google_redirect_uri
    authorization_url, _ = flow.authorization_url(prompt="consent")
    return RedirectResponse(authorization_url, status_code=303)


@router.get("/dashboard/calendar/callback")
async def calendar_callback(
    request: Request,
    code: str = "",
    error: str = "",
    business: Business = Depends(get_current_business),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Handle the Google OAuth callback after user grants permission."""
    if error:
        flash_message(
            request, "error", f"Google Calendar authentication failed: {error}"
        )
        return RedirectResponse("/dashboard/calendar/status", status_code=303)

    if not code:
        flash_message(request, "error", "No authorization code received from Google.")
        return RedirectResponse("/dashboard/calendar/status", status_code=303)

    from cryptography.fernet import Fernet
    from google_auth_oauthlib.flow import Flow

    settings = Settings()  # type: ignore[call-arg]

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.google_redirect_uri],
            }
        },
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    flow.redirect_uri = settings.google_redirect_uri
    flow.fetch_token(code=code)

    # Encrypt and store credentials
    fernet = Fernet(settings.encryption_key.encode())
    encrypted = fernet.encrypt(flow.credentials.to_json().encode()).decode()
    business.google_calendar_encrypted = encrypted
    await db.flush()

    flash_message(request, "success", "Google Calendar connected successfully!")
    return RedirectResponse("/dashboard/calendar/status", status_code=303)
