"""FAQ lookup tool — searches the business FAQ knowledge base.

Registered tool: ``lookup_faq``
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from voice_agent.models import FAQ
from voice_agent.tool_registry import register

logger = logging.getLogger(__name__)

TOOL_SCHEMA: dict[str, Any] = {
    "name": "lookup_faq",
    "description": (
        "Search the FAQ knowledge base for answers to customer questions. "
        "Returns relevant Q&A matches."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The customer's question or keywords to search for",
            },
        },
        "required": ["query"],
    },
}


async def handler(
    parameters: dict[str, Any],
    business_id: int,
    db_session: Any,
    calendar_service: Any = None,
    twilio_client: Any = None,
    call_manager: Any = None,
) -> str:
    """Search the FAQ knowledge base for answers matching *query*.

    Args:
        parameters: Must contain ``query`` (search string).
        business_id: Business owning the FAQs.
        db_session: Async SQLAlchemy session.
        calendar_service: Unused — present for dispatch compatibility.
        twilio_client: Unused — present for dispatch compatibility.
        call_manager: Unused — present for dispatch compatibility.

    Returns:
        Formatted Q&A string for Gemini to speak, or a fallback message
        if no matches are found.
    """
    query = parameters.get("query", "").strip()
    if not query:
        return "Please provide a question to search for."

    try:
        stmt = (
            select(FAQ)
            .where(
                FAQ.business_id == business_id,
                FAQ.enabled.is_(True),
            )
            .order_by(FAQ.created_at.desc())
        )
        result = await db_session.execute(stmt)
        all_faqs = list(result.scalars().all())
    except Exception as exc:
        logger.error("FAQ lookup failed for business %d: %s", business_id, exc)
        return "I'm sorry, I'm having trouble looking up information right now."

    # Score FAQs by keyword presence (simple relevance ranking)
    query_lower = query.lower()
    query_terms = query_lower.split()

    scored: list[tuple[float, FAQ]] = []
    for faq in all_faqs:
        score = 0.0
        q_text = (faq.question or "").lower()
        a_text = (faq.answer or "").lower()
        kw_text = (faq.keywords or "").lower()

        # Exact phrase match in question (highest weight)
        if query_lower in q_text:
            score += 3.0

        # Exact phrase match in keywords
        if query_lower in kw_text:
            score += 2.0

        # Exact phrase match in answer
        if query_lower in a_text:
            score += 1.5

        # Individual term matches
        for term in query_terms:
            if term in q_text:
                score += 1.0
            if term in kw_text:
                score += 0.8
            if term in a_text:
                score += 0.5

        if score > 0:
            scored.append((score, faq))

    # Sort by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top 2 matches
    top_matches = scored[:2]
    if not top_matches:
        logger.info("No FAQ matches for business %d query: %s", business_id, query)
        return (
            "I don't have that information. Let me transfer you to someone "
            "who can help."
        )

    lines: list[str] = []
    for score, faq in top_matches:
        lines.append(f"Q: {faq.question}")
        lines.append(f"A: {faq.answer}")

    logger.info(
        "FAQ lookup for business %d: %d matches for '%s'",
        business_id,
        len(top_matches),
        query,
    )
    return "\n\n".join(lines)


# Auto-register at import time
register("lookup_faq", TOOL_SCHEMA, handler)
