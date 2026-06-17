"""Tests for the Tier 2 judge response parsing.
"""
from synthesis_eval import flag, flags, clip


def test_flag_mixed_types():
    assert flag(True) == 1
    assert flag(0) == 0
    assert flag("relevant") == 1
    assert flag("0") == 0          # must not count as relevant
    assert flag("maybe") is None


def test_flags_normalizes_sequences():
    assert flags([1, 0, "1", "no"]) == [1, 0, 1, 0]
    assert flags({"relevant": [1, 1, 0]}) == [1, 1, 0]


def test_flags_empty_or_invalid_is_none():
    assert flags({"relevant": []}) is None
    assert flags("not a sequence") is None


def test_clip_bounds_to_unit_interval():
    assert clip(1.4) == 1.0
    assert clip(-0.2) == 0.0
    assert clip("x") == 0.0
    assert clip(0.5) == 0.5
