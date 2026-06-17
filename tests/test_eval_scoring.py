"""Tests for the Tier 1 scorer.

reciprocal_rank and scores define recall@k and MRR at chunk and paper level. The
embedder ablation reuses this exact scorer, so its rows compare to the lexical
baseline. That makes these the most reused numbers in the project.
"""
from types import SimpleNamespace

from evaluation import reciprocal_rank, scores

CHUNKS = [
    {"chunk_id": "c0", "arxiv_id": "p0"},
    {"chunk_id": "c1", "arxiv_id": "p0"},
    {"chunk_id": "c2", "arxiv_id": "p1"},
]


def bundle():
    return SimpleNamespace(chunks=CHUNKS)


def test_reciprocal_rank_hit_and_miss():
    assert reciprocal_rank(["x", "y", "z"], "y") == 0.5
    assert reciprocal_rank(["x", "y"], "q") == 0.0


def test_scores_full_hit():
    question = {"source_chunk_id": "c2", "arxiv_id": "p1"}
    assert scores([2, 0, 1], bundle(), question, k=3) == {
        "recall_chunk": 1.0, "recall_paper": 1.0, "mrr_chunk": 1.0, "mrr_paper": 1.0,
    }


def test_scores_wrong_chunk_right_paper():
    # The single reference caveat from the README. A sibling chunk of the right
    # paper misses on chunk recall but hits on paper recall.
    question = {"source_chunk_id": "c2", "arxiv_id": "p0"}
    row = scores([0, 1], bundle(), question, k=3)
    assert row["recall_chunk"] == 0.0 and row["recall_paper"] == 1.0


def test_scores_respects_k_cutoff():
    question = {"source_chunk_id": "c2", "arxiv_id": "p1"}
    # The gold chunk sits at rank 3 but k=2 excludes it.
    assert scores([0, 1, 2], bundle(), question, k=2)["recall_chunk"] == 0.0
