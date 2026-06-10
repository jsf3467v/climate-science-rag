"""Tier-1 retrieval evaluation for the climate-arXiv RAG project (evaluation/).

Scores three configurations. BM25 only, BM25-over-HyDE, and HyDE+Claude-rerank—on the question set,
comparing each to the ground-truth source chunk from which the question originated. Metrics include
recall@k and MRR at both chunk and paper levels, all objective and deterministic, providing a clear
assessment of whether HyDE and the reranker add value. The per-paper diversity cap is intentionally
disabled here to prevent masking same-paper hits; the focus is on ranking n quality. The HyDE and the
reranker reuse their on-disk caches, so rerunning incurs no additional cost. The setup is outside src/, so src/ is added to the path.


    export ANTHROPIC_API_KEY=...
    python evaluation.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import EvalConfig as Config, HydeConfig, RerankerConfig, RetrieveConfig   # noqa: E402
from retrieve import bm25_ranks, candidate, index_artifacts               # noqa: E402
from rerank import cached_ranking, reranked                               # noqa: E402
from hyde import hyde_query                                               # noqa: E402

METRICS = ("recall_chunk", "recall_paper", "mrr_chunk", "mrr_paper")


def question_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def hyde_ranks(query: str, bundle, client: Anthropic, ecfg: Config, hcfg: HydeConfig) -> list[int]:
    return bm25_ranks(hyde_query(query, client, hcfg), bundle, ecfg.pool)


def rerank_order(query: str, bundle, client: Anthropic, rcfg: RerankerConfig, hcfg: HydeConfig) -> list[int]:
    return [i for i, _ in reranked(query, bundle, client, rcfg, hcfg)]


def plain_rerank_order(query: str, bundle, client: Anthropic, rcfg: RerankerConfig) -> list[int]:
    """Rerank the bare-BM25 pool, no HyDE, to isolate HyDE's contribution to the candidate set."""
    pool = bm25_ranks(query, bundle, rcfg.candidates)
    ranked = cached_ranking(query, [candidate(bundle.chunks[i]) for i in pool], client, rcfg)
    return [pool[local] for local, _ in ranked]


def order(config: str, query: str, bundle, client: Anthropic, ecfg: Config,
          rcfg: RerankerConfig, hcfg: HydeConfig) -> list[int]:
    if config == "bm25":
        return bm25_ranks(query, bundle, ecfg.pool)
    if config == "hyde":
        return hyde_ranks(query, bundle, client, ecfg, hcfg)
    if config == "rerank_nohyde":
        return plain_rerank_order(query, bundle, client, rcfg)
    return rerank_order(query, bundle, client, rcfg, hcfg)


def reciprocal_rank(ids: list, target) -> float:
    for rank, value in enumerate(ids, start=1):
        if value == target:
            return 1.0 / rank
    return 0.0


def scores(indices: list[int], bundle, question: dict, k: int) -> dict:
    chunk_ids = [bundle.chunks[i]["chunk_id"] for i in indices[:k]]
    papers = [bundle.chunks[i]["arxiv_id"] for i in indices[:k]]
    src_chunk, src_paper = question["source_chunk_id"], question["arxiv_id"]
    return {"recall_chunk": float(src_chunk in chunk_ids),
            "recall_paper": float(src_paper in papers),
            "mrr_chunk": reciprocal_rank(chunk_ids, src_chunk),
            "mrr_paper": reciprocal_rank(papers, src_paper)}


def report(agg: dict, n: int, ecfg: Config) -> None:
    head = "config".ljust(10) + "".join(m.rjust(14) for m in METRICS)
    print(f"n={n}  k={ecfg.k}\n{head}")
    summary = {}
    for config in ecfg.configs:
        means = {m: agg[config][m] / n for m in METRICS}
        summary[config] = {m: round(v, 4) for m, v in means.items()}
        print(config.ljust(10) + "".join(f"{means[m]:14.3f}" for m in METRICS))
    ecfg.summary_path.write_text(json.dumps({"n": n, "k": ecfg.k, "configs": summary}, indent=2))


def main() -> None:
    ecfg = Config()
    rcfg = RerankerConfig()
    hcfg = HydeConfig()
    client = Anthropic()
    bundle = index_artifacts(RetrieveConfig())
    rows = question_rows(ecfg.questions_path)
    if not rows:
        raise SystemExit(f"no questions at {ecfg.questions_path}")
    agg = {c: collections.defaultdict(float) for c in ecfg.configs}
    with ecfg.results_path.open("w") as out:
        for question in rows:
            for config in ecfg.configs:
                indices = order(config, question["question"], bundle, client, ecfg, rcfg, hcfg)
                row = scores(indices, bundle, question, ecfg.k)
                out.write(json.dumps({"question_id": question["question_id"], "config": config, **row}) + "\n")
                for metric, value in row.items():
                    agg[config][metric] += value
    report(agg, len(rows), ecfg)


if __name__ == "__main__":
    main()