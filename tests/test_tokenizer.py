"""Tests for the shared tokenizer.

The same tokenizer runs at index time and at query time. So, it is part of the
measured numbers. These check lowercasing, stopword removal, a two character
floor, hyphen retention, and the exclusion of tokens that begin with a digit.
"""
from tokenizer import tokens


def test_lowercases_and_drops_stopwords():
    assert tokens("The Warming of the Ocean") == ["warming", "ocean"]


def test_keeps_hyphenated_terms():
    assert "sea-level" in tokens("sea-level rise")


def test_drops_short_and_digit_leading_tokens():
    out = tokens("a I of 3.5 sea-level")
    assert "3.5" not in out and "i" not in out and "a" not in out
    assert out == ["sea-level"]


def test_empty_text():
    assert tokens("") == []
