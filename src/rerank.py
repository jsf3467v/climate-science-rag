"""LLM reranker for the climate-arXiv RAG project (src/).

This is the initial stage where the language model begins. Retrieval (BM25)
provides a pool of candidates; this step prompts Claude to reorder them by
relevance to the query, using a listwise approach in a single call, retaining
only the top results. These orderings are stored on disk based on query, candidate set,
prompt, and snippet length, ensuring that a crashed or repeated evaluation run does not
incur duplicate costs and that prompt edits invalidate the cache properly. No local
models are used here; only the API model is involved.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python rerank.py sea level rise under warming
"""

from __future__ import annotations

import argparse
import hashlib

from anthropic import Anthropic

from config import RerankerConfig as Config, HydeConfig, RetrieveConfig, cached
from hyde import hyde_query
from jsonblock import json_block
from retrieve import Bundle, bm25_ranks, candidate, index_artifacts

SYSTEM = ("You are a relevance judge for a climate-science retrieval system. "
          "Rank passages by how directly they answer the query.")


def prompt(query: str, candidates: list[dict], cfg: Config) -> str:
    blocks = [f"[{i}] {c.get('title') or ''}\n{c['text'][:cfg.snippet_chars]}"
              for i, c in enumerate(candidates)]
    listing = "\n\n".join(blocks)
    return (f"Query: {query}\n\nPassages:\n{listing}\n\n"
            f"Return only a JSON array, most relevant first, one object per passage: "
            f'[{{"index": <int>, "score": <float 0-1>}}]. No prose.')


def ranking(text: str, n: int) -> list[tuple[int, float]]:
    rows = json_block(text)
    if isinstance(rows, list):
        ranked = [(int(r["index"]), float(r["score"])) for r in rows
                  if isinstance(r, dict) and 0 <= int(r["index"]) < n]
    else:
        ranked = []
    seen = {i for i, _ in ranked}
    ranked += [(i, 0.0) for i in range(n) if i not in seen]
    return ranked


def claude_ranking(query: str, candidates: list[dict], client: Anthropic, cfg: Config) -> list[tuple[int, float]]:
    msg = client.messages.create(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(query, candidates, cfg)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return ranking(text, len(candidates))


def cache_key(query: str, candidates: list[dict], cfg: Config) -> str:
    # Temperature and max_tokens are not in this key, so clear rerank_cache.db if you change either.
    ids = ",".join(c["chunk_id"] for c in candidates)
    payload = f"{cfg.model}|{cfg.snippet_chars}|{SYSTEM}|{query}|{ids}"
    return hashlib.sha1(payload.encode()).hexdigest()


def cached_ranking(query: str, candidates: list[dict], client: Anthropic, cfg: Config) -> list[tuple[int, float]]:
    if not cfg.cache:
        return claude_ranking(query, candidates, client, cfg)
    ranked = cached(cfg.cache_path, cache_key(query, candidates, cfg),
                    lambda: claude_ranking(query, candidates, client, cfg))
    return [tuple(pair) for pair in ranked]


def reranked(query: str, bundle: Bundle, client: Anthropic,
             rcfg: Config, hcfg: HydeConfig) -> list[tuple[int, float]]:
    """Canonical retrieval-and-rerank: HyDE-expanded BM25 pool, Claude listwise rerank.
    Returns (chunk_index, score) over bundle.chunks, most relevant first. Eval,
    synthesis, and this CLI all call it, so the measured path is the shipped path."""
    pool = bm25_ranks(hyde_query(query, client, hcfg), bundle, rcfg.candidates)
    ranked = cached_ranking(query, [candidate(bundle.chunks[i]) for i in pool], client, rcfg)
    return [(pool[local], score) for local, score in ranked]


def rerank(query: str, bundle: Bundle, client: Anthropic, rcfg: Config, hcfg: HydeConfig) -> list[dict]:
    ordered = reranked(query, bundle, client, rcfg, hcfg)[:rcfg.top_k]
    return [{**bundle.chunks[i], "rerank_score": score} for i, score in ordered]


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerank retrieval candidates with Claude.")
    ap.add_argument("query", nargs="+", help="query terms")
    cfg = Config()
    client = Anthropic()
    bundle = index_artifacts(RetrieveConfig())
    for r in rerank(" ".join(ap.parse_args().query), bundle, client, cfg, HydeConfig()):
        print(f"{r['rerank_score']:.2f}  {r['arxiv_id']}  [{r['section'] or 'lead'}]  {(r['title'] or '')[:60]}")
        print(f"        {r['text'][:160].strip()}")


if __name__ == "__main__":
    main()