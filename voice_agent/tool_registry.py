"""Tool Registry — single source of truth for VoiceAgent tool schemas and dispatch.

Every tool module registers itself via :func:`register`. The dispatch
function handles both sync and async handlers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# name → {"schema": dict, "handler": callable}
_TOOL_MAP: dict[str, dict[str, Any]] = {}

# Size limits for Gemini Live API compatibility
_MAX_TOOL_DESC: int = 400
_MAX_PROP_DESC: int = 140
_MAX_PROPERTIES: int = 28
_MAX_TOOLS: int = 20


def register(name: str, schema: dict[str, Any], handler: Callable[..., Any]) -> None:
    """Register a tool. Overwrites if *name* already exists (idempotent).

    Args:
        name: Tool name (must match schema["name"]).
        schema: Gemini function declaration dict.
        handler: Sync or async callable that executes the tool.
    """
    _TOOL_MAP[name] = {"schema": schema, "handler": handler}
    logger.debug("Registered tool: %s", name)


async def dispatch(name: str, parameters: dict[str, Any], **context: Any) -> str:
    """Look up *name* and call the handler, returning its string result.

    Supports both sync and async handlers.

    Args:
        name: Tool name to dispatch.
        parameters: Tool parameters dict.
        **context: Additional keyword arguments forwarded to the handler.

    Returns:
        Human-readable result string for Gemini to speak.

    Raises:
        KeyError: If *name* is not registered.
    """
    entry = _TOOL_MAP.get(name)
    if entry is None:
        raise KeyError(f"Unknown tool: {name}")

    handler = entry["handler"]
    if asyncio.iscoroutinefunction(handler):
        result = await handler(parameters=parameters, **context)
    else:
        result = handler(parameters=parameters, **context)
    return result or "Done."


def get_schemas() -> list[dict[str, Any]]:
    """Return all registered tool schemas for Gemini function declarations."""
    return [entry["schema"] for entry in _TOOL_MAP.values()]


def is_registered(name: str) -> bool:
    """Check if a tool is registered by name."""
    return name in _TOOL_MAP


def list_tool_names() -> list[str]:
    """Return sorted list of registered tool names."""
    return sorted(_TOOL_MAP.keys())


def trim_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Trim a single tool schema to fit Gemini Live API limits.

    - Top-level description trimmed to 400 characters.
    - Property descriptions trimmed to 140 characters.
    - Maximum 28 properties per tool.

    Args:
        schema: Original tool schema dict.

    Returns:
        Trimmed copy of the schema.
    """
    trimmed = dict(schema)

    desc = trimmed.get("description", "")
    if isinstance(desc, str) and len(desc) > _MAX_TOOL_DESC:
        trimmed["description"] = desc[:_MAX_TOOL_DESC] + "\u2026"

    params = trimmed.get("parameters", {})
    if not isinstance(params, dict):
        return trimmed

    props = params.get("properties", {})
    if not isinstance(props, dict):
        return trimmed

    for prop_schema in props.values():
        if not isinstance(prop_schema, dict):
            continue
        prop_desc = prop_schema.get("description", "")
        if isinstance(prop_desc, str) and len(prop_desc) > _MAX_PROP_DESC:
            prop_schema["description"] = prop_desc[:_MAX_PROP_DESC] + "\u2026"

    prop_items = list(props.items())
    if len(prop_items) > _MAX_PROPERTIES:
        kept = prop_items[:_MAX_PROPERTIES]
        trimmed["parameters"]["properties"] = dict(kept)

    return trimmed


def trim_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim all tool schemas and limit to 20 total.

    Args:
        schemas: List of tool schema dicts.

    Returns:
        Trimmed and limited list of schemas.
    """
    trimmed = [trim_schema(s) for s in schemas]

    if len(trimmed) > _MAX_TOOLS:
        logger.warning(
            "Trimming %d tools to %d (exceeds Live API limit)",
            len(trimmed),
            _MAX_TOOLS,
        )
        trimmed = trimmed[:_MAX_TOOLS]

    return trimmed
