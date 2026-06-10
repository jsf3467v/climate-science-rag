"""Retrieval for the climate arXiv RAG project (src/).

This file loads the BM25 index built by indexer.py, scores a query against it, and returns
the top chunks with provenance for the reranking and synthesis stages. No
pretrained model is used here. The LSA arm and RRF fusion were removed after
evaluation showed fusion performed worse than BM25 alone.

Run from src/ so config and the tokenizer import resolve. For example

    python retrieve.py sea level rise under warming
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from config import RetrieveConfig as Config
from tokenizer import tokens


@dataclass
class Bundle:
    bm25: object
    chunks: list[dict]


def jsonl_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def index_artifacts(cfg: Config) -> Bundle:
    chunks = jsonl_rows(cfg.chunks_path)
    bm25 = joblib.load(cfg.index_dir / "bm25.joblib")
    if len(chunks) != bm25.corpus_size:
        raise SystemExit(f"alignment: {len(chunks)} chunks, {bm25.corpus_size} bm25 docs; rebuild index")
    return Bundle(bm25, chunks)


def top_indices(scores: np.ndarray, n: int) -> list[int]:
    n = min(n, len(scores))
    part = np.argpartition(scores, -n)[-n:]
    return part[np.argsort(scores[part])[::-1]].tolist()


def bm25_scores(query: str, bundle: Bundle) -> np.ndarray:
    return np.asarray(bundle.bm25.get_scores(tokens(query)))


def bm25_ranks(query: str, bundle: Bundle, n: int) -> list[int]:
    return top_indices(bm25_scores(query, bundle), n)


def diversify(ranked: list[tuple[int, float]], chunks: list[dict], per_paper: int, top_k: int) -> list[tuple[int, float]]:
    seen: dict[str, int] = collections.Counter()
    out = []
    for idx, score in ranked:
        paper = chunks[idx]["arxiv_id"]
        if seen[paper] >= per_paper:
            continue
        seen[paper] += 1
        out.append((idx, score))
        if len(out) >= top_k:
            break
    return out


def result_row(chunk: dict, score: float) -> dict:
    return {
        "score": round(score, 6),
        "chunk_id": chunk["chunk_id"],
        "arxiv_id": chunk["arxiv_id"],
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "source": chunk.get("source"),
        "abs_url": chunk.get("abs_url"),
        "text": chunk["text"],
    }


def candidate(chunk: dict) -> dict:
    """The minimal fields a model needs to rank or read a chunk: id, title, text."""
    return {"chunk_id": chunk["chunk_id"], "title": chunk.get("title"), "text": chunk["text"]}


def search(query: str, bundle: Bundle, cfg: Config) -> list[dict]:
    scores = bm25_scores(query, bundle)
    ranked = [(i, float(scores[i])) for i in top_indices(scores, cfg.pool)]
    kept = diversify(ranked, bundle.chunks, cfg.per_paper, cfg.top_k)
    return [result_row(bundle.chunks[i], score) for i, score in kept]


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the climate corpus (BM25).")
    ap.add_argument("query", nargs="+", help="query terms")
    cfg = Config()
    bundle = index_artifacts(cfg)
    for r in search(" ".join(ap.parse_args().query), bundle, cfg):
        print(f"{r['score']:.4f}  {r['arxiv_id']}  [{r['section'] or 'lead'}]  {(r['title'] or '')[:60]}")
        print(f"        {r['text'][:160].strip()}")


if __name__ == "__main__":
    main()