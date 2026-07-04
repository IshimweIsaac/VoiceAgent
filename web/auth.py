"""Authentication routes for the web dashboard.

Provides registration, login, and logout functionality.
Uses session-based auth with bcrypt password hashing.
"""

from __future__ import annotations

import logging
import re
import time

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

import bcrypt as _bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from voice_agent.database import get_db
from voice_agent.models import Business, User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="web/templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert business name to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def flash_message(request: Request, type_: str, text: str) -> None:
    """Store a flash message in the session for display on the next page load."""
    messages: list[dict[str, str]] = request.session.get("messages", [])
    messages.append({"type": type_, "text": text})
    request.session["messages"] = messages


def get_flash_messages(request: Request) -> list[dict[str, str]]:
    """Retrieve and clear flash messages from the session."""
    return request.session.pop("messages", [])


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def get_current_business(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Business:
    """Extract the authenticated business from the session.

    Raises a 303 redirect to /login if the session is missing or invalid.
    """
    business_id = request.session.get("business_id")
    if not business_id:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    result = await db.execute(select(Business).where(Business.id == business_id))
    business = result.scalar_one_or_none()
    if business is None:
        request.session.clear()
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return business


# ---------------------------------------------------------------------------
# Routes — Login
# ---------------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Render the login form."""
    return templates.TemplateResponse(
        request,
        "login.html",
        {"messages": get_flash_messages(request)},
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Validate email/password and create a session."""
    result = await db.execute(select(User).where(User.email == email.strip().lower()))
    user = result.scalar_one_or_none()

    if user is None or not _bcrypt.checkpw(
        password.encode("utf-8"), user.password_hash.encode("utf-8")
    ):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"messages": [{"type": "error", "text": "Invalid email or password."}]},
            status_code=401,
        )

    request.session["business_id"] = user.business_id
    flash_message(request, "success", "Welcome back!")
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    """Clear the session and redirect to the login page."""
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ---------------------------------------------------------------------------
# Routes — Registration
# ---------------------------------------------------------------------------


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    """Render the registration form."""
    return templates.TemplateResponse(
        request,
        "register.html",
        {"messages": get_flash_messages(request)},
    )


@router.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    business_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    phone_number: str = Form(...),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Create a new Business and User account, then auto-login."""
    # Validate required fields
    business_name = business_name.strip()
    email = email.strip().lower()
    password = password.strip()
    phone_number = phone_number.strip()

    if not business_name or not email or not password:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"messages": [{"type": "error", "text": "All fields are required."}]},
            status_code=400,
        )

    if len(password) < 8:
        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "messages": [
                    {"type": "error", "text": "Password must be at least 8 characters."}
                ]
            },
            status_code=400,
        )

    # Check for duplicate email
    existing_user = await db.execute(select(User).where(User.email == email))
    if existing_user.scalar_one_or_none() is not None:
        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "messages": [
                    {
                        "type": "error",
                        "text": "An account with this email already exists.",
                    }
                ]
            },
            status_code=409,
        )

    # Create the business record
    base_slug = slugify(business_name)
    slug = base_slug
    existing_slug = await db.execute(select(Business).where(Business.slug == slug))
    if existing_slug.scalar_one_or_none() is not None:
        slug = f"{base_slug}-{int(time.time())}"

    business = Business(
        name=business_name,
        slug=slug,
        phone_number=phone_number,
    )
    db.add(business)
    await db.flush()

    # Create the user record
    user = User(
        business_id=business.id,
        email=email,
        password_hash=_bcrypt.hashpw(
            password.encode("utf-8"), _bcrypt.gensalt()
        ).decode("utf-8"),
    )
    db.add(user)
    await db.flush()

    # Auto-login
    request.session["business_id"] = business.id
    flash_message(
        request, "success", f"Welcome, {business_name}! Your account is ready."
    )
    return RedirectResponse("/dashboard", status_code=303)
