"""Tests for the HTTP routes.

The retrieval and synthesis paths are replaced with stubs, so these tests check
the contract the service exposes rather than the measured numbers, which the
retrieval and evaluation tests already cover. No network or model call is made.
"""
import pytest
from fastapi.testclient import TestClient

import api
from schemas import QUERY_CHAR_CAP

ROW = {
    "chunk_id": "c1", "arxiv_id": "2401.00001", "title": "Sea level projections",
    "section": "Results", "abs_url": "https://arxiv.org/abs/2401.00001",
    "text": "Projections diverge above two degrees of warming.", "score": 1.5,
}


class Bundle:
    chunks = [ROW]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "search", lambda query, bundle, cfg: [ROW])
    monkeypatch.setattr(api, "context", lambda query, bundle, cl, r, h: [ROW])
    monkeypatch.setattr(api, "answer", lambda query, chunks, cl, cfg: "Warming drives it [0].")
    monkeypatch.setattr(api, "index_artifacts", lambda cfg: Bundle())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with TestClient(api.app) as running:
        yield running


def test_health_reports_loaded_corpus(client):
    body = client.get("/health").json()
    assert body["ready"] is True and body["chunks"] == 1


def test_search_returns_passages(client):
    body = client.post("/search", json={"query": "sea level"}).json()
    assert body["passages"][0]["arxiv_id"] == "2401.00001"


def test_answer_marks_covered_question(client):
    body = client.post("/answer", json={"query": "sea level"}).json()
    assert body["covered"] is True and body["passages"]


def test_answer_marks_uncovered_question(client, monkeypatch):
    monkeypatch.setattr(api, "answer", lambda query, chunks, cl, cfg: api.DECLINED)
    assert client.post("/answer", json={"query": "sea level"}).json()["covered"] is False


def test_query_over_cap_rejected(client):
    assert client.post("/search", json={"query": "x" * (QUERY_CHAR_CAP + 1)}).status_code == 422


def test_empty_query_rejected(client):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_missing_key_returns_service_unavailable(client, monkeypatch):
    monkeypatch.setitem(api.STATE, "client", None)
    assert client.post("/answer", json={"query": "sea level"}).status_code == 503