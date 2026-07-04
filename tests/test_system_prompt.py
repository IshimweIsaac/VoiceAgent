"""Tests for voice_agent.system_prompt — prompt builder, hours formatter, FAQ formatter."""

from __future__ import annotations

import pytest

from voice_agent.system_prompt import (
    build_system_prompt,
    format_faqs_for_prompt,
    format_hours,
)

# ---------------------------------------------------------------------------
# format_hours
# ---------------------------------------------------------------------------


class TestFormatHours:
    """Business hours formatting."""

    def test_format_hours_configured_day_shows_hours(self):
        """A configured day shows 'Monday: 09:00 - 17:00'."""
        hours = {"monday": {"open": "09:00", "close": "17:00"}}
        result = format_hours(hours)
        assert "Monday: 09:00 - 17:00" in result

    def test_format_hours_closed_day_shows_closed(self):
        """A missing day shows 'Monday: Closed'."""
        hours = {"tuesday": {"open": "09:00", "close": "17:00"}}
        result = format_hours(hours)
        assert "Monday: Closed" in result
        assert "Tuesday: 09:00 - 17:00" in result

    def test_format_hours_none_returns_not_configured(self):
        """None input returns 'Not configured'."""
        assert format_hours(None) == "Not configured"

    def test_format_hours_empty_dict_returns_not_configured(self):
        """Empty dict returns 'Not configured' since it's falsy."""
        assert format_hours({}) == "Not configured"

    def test_format_hours_all_days_present(self):
        """All 7 days appear in output."""
        hours = {day: {"open": "09:00", "close": "17:00"} for day in [
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday",
        ]}
        result = format_hours(hours)
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            assert f"{day}: 09:00 - 17:00" in result

    def test_format_hours_mixed_open_closed(self):
        """Mix of open and closed days."""
        hours = {
            "monday": {"open": "09:00", "close": "17:00"},
            "wednesday": {"open": "10:00", "close": "18:00"},
        }
        result = format_hours(hours)
        assert "Monday: 09:00 - 17:00" in result
        assert "Tuesday: Closed" in result
        assert "Wednesday: 10:00 - 18:00" in result
        assert "Thursday: Closed" in result

    def test_format_hours_day_without_open_close_shows_closed(self):
        """Day entry missing 'open'/'close' keys shows as Closed."""
        hours = {"monday": {}}
        result = format_hours(hours)
        assert "Monday: Closed" in result

    def test_format_hours_day_with_partial_hours_shows_closed(self):
        """Day with only 'open' and no 'close' shows as Closed."""
        hours = {"monday": {"open": "09:00"}}
        result = format_hours(hours)
        assert "Monday: Closed" in result


# ---------------------------------------------------------------------------
# format_faqs_for_prompt
# ---------------------------------------------------------------------------


class TestFormatFAQs:
    """FAQ formatting."""

    def test_format_faqs_for_prompt_returns_expected_format(self):
        """Dict FAQs formatted as numbered Q/A pairs."""
        faqs = [
            {"question": "What are your hours?", "answer": "We are open 9-5."},
            {"question": "Where are you located?", "answer": "Downtown."},
        ]
        result = format_faqs_for_prompt(faqs)
        assert "1. Q: What are your hours?" in result
        assert "   A: We are open 9-5." in result
        assert "2. Q: Where are you located?" in result
        assert "   A: Downtown." in result

    def test_format_faqs_for_prompt_handles_empty_list(self):
        """Empty list returns 'No FAQs configured.'"""
        assert format_faqs_for_prompt([]) == "No FAQs configured."

    def test_format_faqs_for_prompt_handles_none(self):
        """None returns 'No FAQs configured.'"""
        assert format_faqs_for_prompt(None) == "No FAQs configured."

    def test_format_faqs_for_prompt_handles_model_objects(self):
        """FAQ model objects with .question/.answer work."""
        class FakeFAQ:
            def __init__(self, question, answer):
                self.question = question
                self.answer = answer

        faqs = [
            FakeFAQ("Q1?", "A1."),
            FakeFAQ("Q2?", "A2."),
        ]
        result = format_faqs_for_prompt(faqs)
        assert "1. Q: Q1?" in result
        assert "   A: A1." in result
        assert "2. Q: Q2?" in result
        assert "   A: A2." in result

    def test_format_faqs_for_prompt_mixed_types(self):
        """Mixed dict and object FAQs both work."""
        class Obj:
            def __init__(self, q, a):
                self.question = q
                self.answer = a

        faqs = [
            {"question": "Dict Q?", "answer": "Dict A."},
            Obj("Obj Q?", "Obj A."),
        ]
        result = format_faqs_for_prompt(faqs)
        assert "1. Q: Dict Q?" in result
        assert "2. Q: Obj Q?" in result

    def test_format_faqs_for_prompt_missing_question_or_answer(self):
        """FAQ missing question/answer fields doesn't crash."""
        faqs = [
            {"question": "Only question"},
            {"answer": "Only answer"},
            {},
        ]
        # No crash expected; missing fields show as empty
        result = format_faqs_for_prompt(faqs)
        assert "1. Q: Only question" in result
        assert "   A: " in result
        assert "2. Q: " in result

    def test_format_faqs_for_prompt_single_faq(self):
        """Single FAQ entry works (no number padding issues)."""
        faqs = [{"question": "Solo?", "answer": "Solo answer."}]
        result = format_faqs_for_prompt(faqs)
        assert result == "1. Q: Solo?\n   A: Solo answer."


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Full system prompt builder."""

    def test_build_system_prompt_contains_business_name(self):
        """The business name appears in output."""
        prompt = build_system_prompt(
            business_name="Joe's Plumbing",
            greeting_message="Hello!",
            business_hours={"monday": {"open": "09:00", "close": "17:00"}},
            timezone="America/New_York",
            faqs=[],
        )
        assert "Joe's Plumbing" in prompt

    def test_build_system_prompt_contains_greeting_with_formatted_name(self):
        """{business_name} placeholder is replaced in greeting."""
        prompt = build_system_prompt(
            business_name="Acme Corp",
            greeting_message="Welcome to {business_name}!",
            business_hours={},
            timezone="UTC",
            faqs=[],
        )
        assert "Welcome to Acme Corp!" in prompt
        assert "{business_name}" not in prompt

    def test_build_system_prompt_includes_faqs(self):
        """FAQ content appears in prompt."""
        faqs = [{"question": "Q1?", "answer": "A1."}]
        prompt = build_system_prompt(
            business_name="Biz",
            greeting_message="Hi",
            business_hours={},
            timezone="UTC",
            faqs=faqs,
        )
        assert "Q1?" in prompt
        assert "A1." in prompt

    def test_build_system_prompt_handles_empty_faqs(self):
        """No FAQs → 'No FAQs configured.' message."""
        prompt = build_system_prompt(
            business_name="Biz",
            greeting_message="Hi",
            business_hours={},
            timezone="UTC",
            faqs=[],
        )
        assert "No FAQs configured." in prompt

    def test_build_system_prompt_handles_none_hours(self):
        """None hours → 'Not configured' message."""
        prompt = build_system_prompt(
            business_name="Biz",
            greeting_message="Hi",
            business_hours=None,
            timezone="UTC",
            faqs=[],
        )
        assert "Not configured" in prompt

    def test_build_system_prompt_includes_timezone(self):
        """Timezone string appears in prompt."""
        prompt = build_system_prompt(
            business_name="Biz",
            greeting_message="Hi",
            business_hours={},
            timezone="America/Chicago",
            faqs=[],
        )
        assert "America/Chicago" in prompt

    def test_build_system_prompt_includes_hours(self):
        """Business hours are included."""
        hours = {
            "monday": {"open": "08:00", "close": "18:00"},
            "tuesday": {"open": "08:00", "close": "18:00"},
        }
        prompt = build_system_prompt(
            business_name="Shop",
            greeting_message="Hello",
            business_hours=hours,
            timezone="America/New_York",
            faqs=[],
        )
        assert "Monday: 08:00 - 18:00" in prompt
        assert "Tuesday: 08:00 - 18:00" in prompt

    def test_build_system_prompt_contains_rules_section(self):
        """The prompt contains the rules/conversation-flow sections."""
        prompt = build_system_prompt(
            business_name="Test",
            greeting_message="Hi",
            business_hours=None,
            timezone="UTC",
            faqs=[],
        )
        assert "RULES:" in prompt
        assert "CONVERSATION FLOW:" in prompt
        assert "CAPABILITIES:" in prompt

    def test_build_system_prompt_no_placeholder_left_unfilled(self):
        """No raw placeholders remain in output."""
        prompt = build_system_prompt(
            business_name="MyBiz",
            greeting_message="Hello from {business_name}.",
            business_hours={},
            timezone="Asia/Kolkata",
            faqs=[],
        )
        assert "{" not in prompt or "{business_name}" not in prompt
