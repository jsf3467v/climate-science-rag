"""Dense-vs-lexical pipeline comparison for the climate-arXiv RAG project (ablation/).

Scores four methods on the same question set and with the same evaluator, enabling direct comparison: bm25 (lexical baseline), dense (FAISS
retrieval), rerank (the shipped lexical HyDE+rerank pipeline), and dense_rerank (FAISS pool fed into the Claude reranker).


    export ANTHROPIC_API_KEY=...
    python ablation/dense_eval.py --device mps --model thenlper/gte-large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
ABLATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

from config import EvalConfig, RetrieveConfig, RerankerConfig, HydeConfig    # noqa: E402
from retrieve import bm25_ranks, candidate, index_artifacts                  # noqa: E402
from rerank import cached_ranking, reranked                                  # noqa: E402
from evaluation import question_rows                                         # noqa: E402  shared scorer lives in arm_scores
from embedder_ablation import AblationConfig as Config, slug, arm_scores, report  # noqa: E402
from dense_index import index_path                                          # noqa: E402


def faiss_ranks(query: str, index: faiss.Index, model: SentenceTransformer, n: int) -> list[int]:
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    return index.search(vec, n)[1][0].tolist()


def dense_rerank_order(query: str, index, model, bundle, client: Anthropic, rcfg: RerankerConfig) -> list[int]:
    pool = faiss_ranks(query, index, model, rcfg.candidates)
    ranked = cached_ranking(query, [candidate(bundle.chunks[i]) for i in pool], client, rcfg)
    return [pool[local] for local, _ in ranked]


def lexical_rerank_order(query: str, bundle, client: Anthropic, rcfg: RerankerConfig, hcfg: HydeConfig) -> list[int]:
    return [i for i, _ in reranked(query, bundle, client, rcfg, hcfg)]


def arms(index, model, bundle, client: Anthropic, ecfg: EvalConfig, rcfg: RerankerConfig, hcfg: HydeConfig) -> dict:
    return {
        "bm25": lambda q: bm25_ranks(q, bundle, ecfg.pool),
        "dense": lambda q: faiss_ranks(q, index, model, ecfg.pool),
        "rerank": lambda q: lexical_rerank_order(q, bundle, client, rcfg, hcfg),
        "dense_rerank": lambda q: dense_rerank_order(q, index, model, bundle, client, rcfg),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare dense and lexical retrieval pipelines on one scorer.")
    ap.add_argument("--device", default=Config.device, help="cpu, mps, or cuda")
    ap.add_argument("--model", default=Config.model, help="sentence-transformer model id")
    args = ap.parse_args()
    cfg = Config(device=args.device, model=args.model)
    path = index_path(cfg)
    if not path.exists():
        raise SystemExit(f"no dense index at {path}; run dense_index.py --model {cfg.model} first")

    ecfg, rcfg, hcfg = EvalConfig(), RerankerConfig(), HydeConfig()
    bundle = index_artifacts(RetrieveConfig())
    index = faiss.read_index(str(path))
    if index.ntotal != len(bundle.chunks):
        raise SystemExit(f"alignment: {index.ntotal} vectors, {len(bundle.chunks)} chunks; rebuild with dense_index.py --refresh")
    rows = question_rows(ecfg.questions_path)
    if not rows:
        raise SystemExit(f"no questions at {ecfg.questions_path}")

    client = Anthropic(max_retries=8)
    model = SentenceTransformer(cfg.model, device=cfg.device)
    results = {name: arm_scores(rank, rows, bundle, ecfg.k)
               for name, rank in arms(index, model, bundle, client, ecfg, rcfg, hcfg).items()}
    report(results, len(rows), ecfg.k, cfg.model, ABLATION / f"dense_comparison_{slug(cfg.model)}.json")


if __name__ == "__main__":
    main()