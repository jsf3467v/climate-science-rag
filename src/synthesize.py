"""Answers synthesis for the climate-arXiv RAG project (src/).

Builds the retrieval pipeline by combining the HyDE-expanded query, BM25, and Claude rerank into a context set.
It then prompts Claude to answer using only this context and to cite passage numbers. The answer is grounded through
design. The prompt disallows outside knowledge and requests an explicit "not in the provided context" if the answer
isn't found, ensuring Tier-2 faithfulness has a clear basis for scoring. Answers are stored on disk by query, context, and
prompt. The process uses only the API as the model.


    pip install anthropic rank_bm25 scikit-learn
    export ANTHROPIC_API_KEY=...
    python synthesize.py what limits sea level projections under warming
"""

from __future__ import annotations

import argparse
import hashlib

from anthropic import Anthropic

from config import SynthConfig as Config, HydeConfig, RerankerConfig, RetrieveConfig, cached
from rerank import reranked
from retrieve import Bundle, index_artifacts

SYSTEM = ("You are a climate-science research assistant. Answer the question using only the "
          "numbered passages provided, and cite the passages you use as [n]. If the passages do "
          "not contain the answer, reply exactly: the provided context does not answer this "
          "question. Do not use outside knowledge.")


def context(query: str, bundle: Bundle, client: Anthropic,
            rcfg: RerankerConfig, hcfg: HydeConfig) -> list[dict]:
    ordered = reranked(query, bundle, client, rcfg, hcfg)[:rcfg.top_k]
    return [bundle.chunks[i] for i, _ in ordered]


def prompt(query: str, chunks: list[dict], cfg: Config) -> str:
    blocks = [f"[{i}] {c.get('title') or ''}\n{c['text'][:cfg.snippet_chars]}"
              for i, c in enumerate(chunks)]
    return f"Question: {query}\n\nPassages:\n" + "\n\n".join(blocks) + "\n\nAnswer:"


def claude_answer(query: str, chunks: list[dict], client: Anthropic, cfg: Config) -> str:
    msg = client.messages.create(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(query, chunks, cfg)}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def cache_key(query: str, chunks: list[dict], cfg: Config) -> str:
    # The key assumes temperature and max_tokens are fixed. Change either and clear synth_cache.db first, or stale answers are served.
    ids = ",".join(c["chunk_id"] for c in chunks)
    payload = f"{cfg.model}|{cfg.temperature}|{cfg.max_tokens}|{cfg.snippet_chars}|{SYSTEM}|{query}|{ids}"
    return hashlib.sha1(payload.encode()).hexdigest()


def answer(query: str, chunks: list[dict], client: Anthropic, cfg: Config) -> str:
    if not cfg.cache:
        return claude_answer(query, chunks, client, cfg)
    return cached(cfg.cache_path, cache_key(query, chunks, cfg),
                  lambda: claude_answer(query, chunks, client, cfg))


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer a query from reranked context (RAG synthesis).")
    ap.add_argument("query", nargs="+", help="query terms")
    cfg = Config()
    client = Anthropic()
    bundle = index_artifacts(RetrieveConfig())
    query = " ".join(ap.parse_args().query)
    chunks = context(query, bundle, client, RerankerConfig(), HydeConfig())
    print(answer(query, chunks, client, cfg))
    print("\nsources:")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c['arxiv_id']}  {(c.get('title') or '')[:70]}")


if __name__ == "__main__":
    main()