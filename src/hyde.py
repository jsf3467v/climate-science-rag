"""HyDE query expansion for the climate-arXiv RAG project (src/).

This process bridges the vocabulary gap between a paraphrased question and the corpus. Claude generates a brief,
plausible climate-science passage that can answer the query. Retrieval then uses this passage, typically with the
original query prepended instead of just the question. It searches for domain-specific
terms and synonyms within the passage, rather than relying solely on the question's exact wording.
This helps address issues such as low-overlap questions identified in diagnostics. Passages are then cached on disk by query
and prompt, allowing recovery during repeated or interrupted runs. The prompt edits effectively clear the cache. The entire
process operates only via the API, with no local models involved.



    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python hyde.py what limits sea level projections under warming
"""

from __future__ import annotations

import argparse
import hashlib

from anthropic import Anthropic

from config import HydeConfig as Config, cached

SYSTEM = ("You are a climate scientist. Given a question, write one dense, "
          "factual paragraph, in the style of a climate-science paper, that "
          "would answer it. Use precise domain terminology and likely synonyms. "
          "State it plainly: no hedging, no first person, no preamble. If unsure, "
          "write the most plausible passage anyway -- it is used only to match "
          "text, never shown to a reader.")


def prompt(query: str, cfg: Config) -> str:
    return f"Question: {query}\n\nWrite the passage in about {cfg.max_words} words."


def claude_passage(query: str, client: Anthropic, cfg: Config) -> str:
    msg = client.messages.create(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(query, cfg)}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def cache_key(query: str, cfg: Config) -> str:
    # Temperature and max_tokens are part of the key, so changing either invalidates the cache on its own.
    payload = f"{cfg.model}|{cfg.temperature}|{cfg.max_tokens}|{cfg.max_words}|{SYSTEM}|{query}"
    return hashlib.sha1(payload.encode()).hexdigest()


def cached_passage(query: str, client: Anthropic, cfg: Config) -> str:
    if not cfg.cache:
        return claude_passage(query, client, cfg)
    return cached(cfg.cache_path, cache_key(query, cfg),
                  lambda: claude_passage(query, client, cfg))


def hyde_query(query: str, client: Anthropic, cfg: Config) -> str:
    passage = cached_passage(query, client, cfg)
    return f"{query} {passage}" if cfg.echo_query else passage


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand a query into a hypothetical passage (HyDE).")
    ap.add_argument("query", nargs="+", help="query terms")
    cfg = Config()
    client = Anthropic()
    query = " ".join(ap.parse_args().query)
    passage = cached_passage(query, client, cfg)
    print(f"query  : {query}\n\npassage:\n{passage}\n\nsearch text:\n{hyde_query(query, client, cfg)}")


if __name__ == "__main__":
    main()