"""Shared tokenizer for the climate-arXiv RAG project (src/).

A single tokenizer is used both at index time for the corpus and at search time for the query,
ensuring that the BM25 index and the query share a consistent view of the text. This tokenizer
resides in its own module and is never executed as a standalone script, so components like indexer, retrieve,
rerank, and eval all import the same function. This approach prevents inconsistencies that could arise if different tokenization methods were used.

"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[^\W\d_][\w\-]*", re.UNICODE)
STOP = ENGLISH_STOP_WORDS
MIN_TOKEN_LEN = 2


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower())
            if t not in STOP and len(t) >= MIN_TOKEN_LEN]