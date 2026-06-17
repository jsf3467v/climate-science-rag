"""Tests for LaTeX to text placeholder scrubbing.

scrub removes the placeholder text pylatexenc leaves behind and keeps the real
content, including math subscripts like CO_2 that must stay untouched.
"""
from normalize import scrub


def test_removes_citation_and_graphics_placeholders():
    assert scrub("Warming <cit.> increases levels <graphics> here.") == "Warming increases levels here."


def test_preserves_subscript_content():
    assert "CO_2" in scrub("emissions of CO_2 rise")


def test_removes_rule_lines():
    assert "====" not in scrub("text ==== more")


def test_collapses_space_before_punctuation():
    assert scrub("spaced  punctuation , here .") == "spaced punctuation, here."
