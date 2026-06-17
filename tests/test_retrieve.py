"""Tests for the deterministic retrieval helpers.

top_indices and diversify are deterministic ranking helpers with no model or
disk access. They support the measured numbers, so their behavior is checked
here. The diversity cap is the per-paper logic used by the search() CLI.
"""
import numpy as np

from retrieve import top_indices, diversify


def test_top_indices_orders_by_score_desc():
    assert top_indices(np.array([0.1, 0.9, 0.5, 0.3]), 2) == [1, 2]


def test_top_indices_clamps_n_to_length():
    assert top_indices(np.array([0.2, 0.8]), 5) == [1, 0]


def test_diversify_caps_one_per_paper():
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
    chunks = [{"arxiv_id": "A"}, {"arxiv_id": "A"}, {"arxiv_id": "B"}, {"arxiv_id": "C"}]
    assert diversify(ranked, chunks, per_paper=1, top_k=3) == [(0, 0.9), (2, 0.7), (3, 0.6)]


def test_diversify_respects_top_k_and_cap():
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
    chunks = [{"arxiv_id": "A"}, {"arxiv_id": "A"}, {"arxiv_id": "B"}, {"arxiv_id": "C"}]
    assert diversify(ranked, chunks, per_paper=2, top_k=3) == [(0, 0.9), (1, 0.8), (2, 0.7)]
