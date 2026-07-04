"""Tests for voice_agent.tool_registry — register, dispatch, trim."""

from __future__ import annotations

import pytest

from voice_agent.tool_registry import (
    _MAX_PROPERTIES,
    _MAX_TOOLS,
    _TOOL_MAP,
    dispatch,
    get_schemas,
    is_registered,
    list_tool_names,
    register,
    trim_schema,
    trim_schemas,
)


# ---------------------------------------------------------------------------
# Register & Dispatch
# ---------------------------------------------------------------------------


class TestRegisterAndDispatch:
    """Tool registration and dispatch."""

    @pytest.mark.asyncio
    async def test_register_sync_handler_dispatch_returns_result(self):
        """Register a sync handler, dispatch it, verify result."""

        def greet(parameters, **context):
            name = parameters.get("name", "World")
            return f"Hello, {name}!"

        register("greet", {"name": "greet"}, greet)
        result = await dispatch("greet", {"name": "Test"})
        assert result == "Hello, Test!"

    @pytest.mark.asyncio
    async def test_register_async_handler_dispatch_returns_result(self):
        """Register an async handler, dispatch it, verify result."""

        async def async_greet(parameters, **context):
            name = parameters.get("name", "World")
            return f"Hi, {name}!"

        register("async_greet", {"name": "async_greet"}, async_greet)
        result = await dispatch("async_greet", {"name": "Async"})
        assert result == "Hi, Async!"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_raises_key_error(self):
        """Dispatching an unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown tool: nonexistent"):
            await dispatch("nonexistent", {})

    @pytest.mark.asyncio
    async def test_dispatch_with_context_passes_through(self):
        """The **context kwargs are forwarded to the handler."""

        def context_checker(parameters, **context):
            assert context.get("business_id") == 42
            assert context.get("call_sid") == "CA123"
            return f"business={context['business_id']}"

        register("ctx_check", {"name": "ctx_check"}, context_checker)
        result = await dispatch("ctx_check", {}, business_id=42, call_sid="CA123")
        assert result == "business=42"

    @pytest.mark.asyncio
    async def test_register_twice_overwrites_previous(self):
        """Re-registering the same name overwrites."""

        def handler_a(parameters, **context):
            return "first"

        def handler_b(parameters, **context):
            return "second"

        register("dup", {"name": "dup"}, handler_a)
        assert await dispatch("dup", {}) == "first"
        register("dup", {"name": "dup"}, handler_b)
        assert await dispatch("dup", {}) == "second"

    @pytest.mark.asyncio
    async def test_handler_returning_none_returns_done(self):
        """Handler returning None/empty gives 'Done.'"""

        def returns_none(parameters, **context):
            return None

        def returns_empty(parameters, **context):
            return ""

        register("none_handler", {"name": "none_handler"}, returns_none)
        register("empty_handler", {"name": "empty_handler"}, returns_empty)

        assert await dispatch("none_handler", {}) == "Done."
        assert await dispatch("empty_handler", {}) == "Done."


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


class TestQueryFunctions:
    """is_registered, list_tool_names, get_schemas."""

    def _register_test_tools(self):
        """Helper to register a few tools."""
        register("alpha", {"name": "alpha", "description": "First tool"}, lambda **kw: "a")
        register("beta", {"name": "beta", "description": "Second tool"}, lambda **kw: "b")
        register("gamma", {"name": "gamma", "description": "Third tool"}, lambda **kw: "c")

    def test_is_registered_returns_true_for_registered_tool(self):
        """Check registered tool."""
        register("exists", {"name": "exists"}, lambda **kw: "ok")
        assert is_registered("exists") is True

    def test_is_registered_returns_false_for_unknown_tool(self):
        """Check unregistered tool."""
        assert is_registered("does_not_exist") is False

    def test_list_tool_names_returns_sorted_names(self):
        """Returns sorted list of names."""
        self._register_test_tools()
        names = list_tool_names()
        assert names == ["alpha", "beta", "gamma"]

    def test_get_schemas_returns_all_schemas(self):
        """Returns list of schema dicts."""
        self._register_test_tools()
        schemas = get_schemas()
        assert len(schemas) == 3
        descriptions = {s["description"] for s in schemas}
        assert descriptions == {"First tool", "Second tool", "Third tool"}


# ---------------------------------------------------------------------------
# Schema trimming
# ---------------------------------------------------------------------------


class TestTrimSchema:
    """Individual schema trimming."""

    def test_trim_schema_trims_long_description(self):
        """Description > 400 chars gets truncated with ellipsis character."""
        schema = {
            "name": "long_desc",
            "description": "x" * 500,
            "parameters": {"type": "object", "properties": {}},
        }
        trimmed = trim_schema(schema)
        assert len(trimmed["description"]) == 401  # 400 chars + ellipsis
        assert trimmed["description"].endswith("\u2026")

    def test_trim_schema_trims_property_description(self):
        """Property description > 140 chars gets truncated."""
        schema = {
            "name": "prop_desc",
            "description": "A tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "y" * 200},
                },
            },
        }
        trimmed = trim_schema(schema)
        prop_desc = trimmed["parameters"]["properties"]["arg1"]["description"]
        assert len(prop_desc) == 141  # 140 chars + ellipsis
        assert prop_desc.endswith("\u2026")

    def test_trim_schema_limits_properties(self):
        """More than 28 properties are trimmed to 28."""
        props = {f"prop_{i}": {"type": "string"} for i in range(35)}
        schema = {
            "name": "many_props",
            "description": "A tool with many properties",
            "parameters": {
                "type": "object",
                "properties": props,
            },
        }
        trimmed = trim_schema(schema)
        assert len(trimmed["parameters"]["properties"]) == _MAX_PROPERTIES

    def test_trim_schema_handles_missing_parameters(self):
        """Schema without parameters returns as-is."""
        schema = {"name": "no_params", "description": "No parameters here"}
        trimmed = trim_schema(schema)
        assert trimmed == schema

    def test_trim_schema_handles_none_parameters(self):
        """Schema with parameters=None does not crash."""
        schema = {"name": "null_params", "description": "desc", "parameters": None}
        trimmed = trim_schema(schema)
        assert trimmed["description"] == "desc"

    def test_trim_schema_short_description_unchanged(self):
        """Short descriptions stay as-is."""
        schema = {
            "name": "short",
            "description": "Hello world",
            "parameters": {"type": "object", "properties": {}},
        }
        trimmed = trim_schema(schema)
        assert trimmed["description"] == "Hello world"

    def test_trim_schema_preserves_non_string_description(self):
        """Non-string description is passed through unchanged."""
        schema = {
            "name": "non_str",
            "description": 12345,
        }
        trimmed = trim_schema(schema)
        assert trimmed["description"] == 12345

    def test_trim_schema_preserves_short_prop_descriptions(self):
        """Property descriptions under 140 chars are not modified."""
        schema = {
            "name": "short_props",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short desc"},
                },
            },
        }
        trimmed = trim_schema(schema)
        assert trimmed["parameters"]["properties"]["name"]["description"] == "Short desc"

    def test_trim_schema_handles_non_dict_properties(self):
        """properties={} or non-dict properties doesn't crash."""
        schema = {
            "name": "empty_props",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
        trimmed = trim_schema(schema)
        assert trimmed["parameters"]["properties"] == {}

    def test_trim_schema_property_without_description_ok(self):
        """Property without description field is not modified."""
        schema = {
            "name": "no_prop_desc",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                },
            },
        }
        trimmed = trim_schema(schema)
        assert trimmed["parameters"]["properties"]["x"]["type"] == "integer"

    def test_trim_schema_non_dict_prop_value_skipped(self):
        """Property whose value is not a dict is skipped safely."""
        schema = {
            "name": "bad_prop_val",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": "just a string",
                    "y": {"type": "integer", "description": "fine"},
                },
            },
        }
        trimmed = trim_schema(schema)
        # The bad property should be preserved as-is (just skipped for description trimming)
        assert trimmed["parameters"]["properties"]["x"] == "just a string"
        assert trimmed["parameters"]["properties"]["y"]["description"] == "fine"


class TestTrimSchemas:
    """Multi-schema trimming."""

    def test_trim_schemas_limits_total_tools(self):
        """More than 20 tools are limited to 20."""
        schemas = []
        for i in range(25):
            schemas.append({"name": f"tool_{i}", "description": f"Tool {i}"})
        trimmed = trim_schemas(schemas)
        assert len(trimmed) == _MAX_TOOLS

    def test_trim_schemas_under_limit_preserves_all(self):
        """Fewer than 20 tools are all preserved."""
        schemas = [{"name": f"t{i}", "description": f"desc{i}"} for i in range(5)]
        trimmed = trim_schemas(schemas)
        assert len(trimmed) == 5

    def test_trim_schemas_empty_list(self):
        """Empty list returns empty list."""
        assert trim_schemas([]) == []

    def test_trim_schemas_applies_trim_schema_to_each(self):
        """Each schema gets individually trimmed (description limits)."""
        schemas = [
            {
                "name": "long",
                "description": "x" * 500,
            },
            {
                "name": "short",
                "description": "fine",
            },
        ]
        trimmed = trim_schemas(schemas)
        assert len(trimmed[0]["description"]) == 401
        assert trimmed[1]["description"] == "fine"
