"""Request and response models for the retrieval API.

The response models mirror result_row() in retrieve.py, so a client reads the
same fields the command line interface prints. The query cap is declared here
as a field constraint rather than checked inside a handler.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

QUERY_CHAR_CAP = 300


class Question(BaseModel):
    query: str = Field(min_length=1, max_length=QUERY_CHAR_CAP)


class Passage(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str | None = None
    section: str | None = None
    abs_url: str | None = None
    text: str
    score: float | None = None


class Matches(BaseModel):
    query: str
    passages: list[Passage]


class Grounded(BaseModel):
    query: str
    answer: str
    covered: bool
    passages: list[Passage]


class Status(BaseModel):
    ready: bool
    chunks: int
    model_available: bool