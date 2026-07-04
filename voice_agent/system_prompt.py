"""System prompt builder — generates business-specific prompts for Gemini.

Assembles the system prompt from business configuration and FAQ data.
"""

from __future__ import annotations

from typing import Any


def format_hours(business_hours: dict[str, Any] | None) -> str:
    """Format business hours dict into a human-readable string.

    Args:
        business_hours: Dict keyed by day name, each with "open" and "close".

    Returns:
        Formatted string like "Monday: 09:00 - 17:00".
    """
    if not business_hours:
        return "Not configured"

    lines: list[str] = []
    for day in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]:
        hours = business_hours.get(day)
        if hours and "open" in hours and "close" in hours:
            lines.append(f"{day.capitalize()}: {hours['open']} - {hours['close']}")
        else:
            lines.append(f"{day.capitalize()}: Closed")

    return "\n".join(lines)


def format_faqs_for_prompt(faqs: list[dict[str, Any]] | list[Any]) -> str:
    """Format FAQ entries into a compact string for the system prompt.

    Args:
        faqs: List of FAQ objects or dicts with "question" and "answer".

    Returns:
        Formatted FAQ string or "No FAQs configured." if empty.
    """
    if not faqs:
        return "No FAQs configured."

    lines: list[str] = []
    for i, faq in enumerate(faqs, start=1):
        if isinstance(faq, dict):
            question = faq.get("question", "")
            answer = faq.get("answer", "")
        else:
            question = getattr(faq, "question", "")
            answer = getattr(faq, "answer", "")
        lines.append(f"{i}. Q: {question}")
        lines.append(f"   A: {answer}")

    return "\n".join(lines)


def build_system_prompt(
    business_name: str,
    greeting_message: str,
    business_hours: dict[str, Any] | None,
    timezone: str,
    faqs: list[Any],
) -> str:
    """Build a business-specific system prompt for the Gemini Live session.

    Args:
        business_name: Name of the business (e.g. "Joe's Plumbing").
        greeting_message: AI greeting template (supports {business_name}).
        business_hours: Dict of business hours per day.
        timezone: IANA timezone string (e.g. "America/New_York").
        faqs: List of FAQ entries (objects with question/answer).

    Returns:
        Complete system prompt string for Gemini.
    """
    hours_str = format_hours(business_hours)
    faq_str = format_faqs_for_prompt(faqs)
    greeting = greeting_message.format(business_name=business_name)

    return f"""You are an AI voice receptionist for {business_name}.

BUSINESS DETAILS:
- Name: {business_name}
- Hours:
{hours_str}
- Timezone: {timezone}

GREETING:
{greeting}

CAPABILITIES:
You can:
1. Answer questions about the business — use the FAQ knowledge base
2. Book appointments — ask for date, time, name, and phone number
3. Transfer callers to a human — when they insist or you cannot help
4. Send SMS confirmations — after booking, confirm via SMS

RULES:
1. Always be polite, professional, and concise
2. Never make up information — use the FAQ tool for answers
3. If you cannot answer a question, offer to transfer to a human
4. When booking an appointment, confirm the details with the caller before creating
5. After booking, confirm via SMS
6. Keep responses brief and natural — this is a phone conversation
7. When the caller asks to speak to a human, transfer immediately
8. Never say "I'm an AI" or "I'm a virtual assistant" — just be helpful

FAQ KNOWLEDGE BASE:
{faq_str}

CONVERSATION FLOW:
- Start with the greeting
- Listen to the caller's needs
- Respond naturally, use tools as needed
- End the conversation gracefully"""
