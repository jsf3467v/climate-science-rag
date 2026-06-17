"""Tests for the shared JSON extractor.
"""
from jsonblock import json_block, span


def test_plain_object():
    assert json_block('prefix {"a": 1} suffix') == {"a": 1}


def test_array_root_in_code_fence():
    text = '```json\n[{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]\n```'
    assert json_block(text) == [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]


def test_empty_object_is_none():
    # An empty object means there is no usable verdict, so json_block returns
    # None instead of an empty dict. Callers then read it as a parse failure
    # and the cache does not store it for later runs.
    assert json_block("{}") is None
    assert json_block("[]") is None


def test_skips_latex_braces_and_finds_trailing_json():
    # The judge wrote reasoning with LaTeX before the real verdict.
    text = r'Reasoning with $\frac{1}{2}$ and {not: valid} then {"score": 0.5}'
    assert json_block(text) == {"score": 0.5}


def test_keeps_last_parseable_region():
    assert json_block('first {"score": 0.1} then {"score": 0.9}') == {"score": 0.9}


def test_truncated_json_is_none():
    assert json_block('the answer is {"score": 0.5') is None


def test_no_json_is_none():
    assert json_block("no json here at all") is None
    assert json_block("") is None
    assert json_block(None) is None


def test_escaped_quotes_inside_string():
    assert json_block(r'{"a": "she said \"hi\""}') == {"a": 'she said "hi"'}


def test_nested_object():
    assert json_block('head {"a": {"b": 2}} tail') == {"a": {"b": 2}}


def test_span_matches_balanced_close():
    s = '{"a": {"b": 2}}'
    assert span(s, 0) == len(s)
