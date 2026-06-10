"""Dense-embedder retrieval ablation for the climate-arXiv RAG project (ablation/).

This measures the cost of the constraint "no pretrained models except the LLM' at the
retrieval layer. A pretrained sentence-transformer is evaluated using the same
question set, ground-truth source chunk, and recall and MRR metrics employed by the production
lexical evaluation, making the BM25 and embedder rows directly comparable. The embedder exists
only here, not in src/, keeping the shipped system purely lexical. It is judged without external
input and without an API; the only expense is local encoding, which is saved to disk in blocks
within a per-model subfolder. This setup allows for recovery after crashes and ensures that swapping models does not mix vectors.


    pip install -r requirements.txt -r ablations/requirements-ablation.txt
    python ablation/embedder_ablation.py --device mps
    python ablation/embedder_ablation.py --device mps --model thenlper/gte-large
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

from config import EvalConfig, RetrieveConfig                   # noqa: E402
from retrieve import bm25_ranks, index_artifacts, top_indices   # noqa: E402
from evaluation import METRICS, question_rows, scores           # noqa: E402  reuse the exact scorer so rows compare

ABLATION_DIR = ROOT / "ablation"


@dataclass
class AblationConfig:
    model: str = "thenlper/gte-large"
    device: str = "cpu"        # portable default; pass --device mps or cuda to speed up encoding
    batch: int = 256           # sentence-transformer encode batch
    block: int = 20000         # chunks per save point, so a crashed encode resumes
    cache_dir: Path = field(default_factory=lambda: ABLATION_DIR / "embedder_cache")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def cached_embeddings(texts: list[str], model: SentenceTransformer, cfg: AblationConfig) -> np.ndarray:
    store = cfg.cache_dir / slug(cfg.model)
    store.mkdir(parents=True, exist_ok=True)
    saved = sorted(store.glob("block_*.npy"))
    for begin in range(len(saved) * cfg.block, len(texts), cfg.block):
        vectors = model.encode(texts[begin:begin + cfg.block], batch_size=cfg.batch,
                               normalize_embeddings=True, convert_to_numpy=True)
        np.save(store / f"block_{begin:08d}.npy", vectors)
        print(f"  embedded {min(begin + cfg.block, len(texts))}/{len(texts)}")
    return np.vstack([np.load(p) for p in sorted(store.glob("block_*.npy"))])


def dense_ranks(query: str, vectors: np.ndarray, model: SentenceTransformer, n: int) -> list[int]:
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return top_indices(vectors @ q, n)


def arm_scores(rank: Callable[[str], list[int]], rows: list[dict], bundle, k: int) -> dict:
    agg: dict = collections.defaultdict(float)
    for question in rows:
        row = scores(rank(question["question"]), bundle, question, k)
        for metric in METRICS:
            agg[metric] += row[metric]
    return {metric: round(agg[metric] / len(rows), 4) for metric in METRICS}


def report(results: dict, n: int, k: int, model: str, path: Path) -> None:
    head = "config".ljust(10) + "".join(m.rjust(14) for m in METRICS)
    print(f"n={n}  k={k}  model={model}\n{head}")
    for name, means in results.items():
        print(name.ljust(10) + "".join(f"{means[m]:14.3f}" for m in METRICS))
    path.write_text(json.dumps({"n": n, "k": k, "model": model, "configs": results}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense-embedder retrieval ablation against the lexical baseline.")
    ap.add_argument("--device", default=AblationConfig.device, help="cpu, mps, or cuda")
    ap.add_argument("--model", default=AblationConfig.model, help="sentence-transformer model id")
    args = ap.parse_args()
    cfg = AblationConfig(device=args.device, model=args.model)
    ecfg = EvalConfig()
    bundle = index_artifacts(RetrieveConfig())
    rows = question_rows(ecfg.questions_path)
    if not rows:
        raise SystemExit(f"no questions at {ecfg.questions_path}")
    model = SentenceTransformer(cfg.model, device=cfg.device)
    vectors = cached_embeddings([c["text"] for c in bundle.chunks], model, cfg)
    if len(vectors) != len(bundle.chunks):
        raise SystemExit(f"alignment: {len(vectors)} vectors, {len(bundle.chunks)} chunks; clear {cfg.cache_dir / slug(cfg.model)}")
    arms = {"bm25": lambda q: bm25_ranks(q, bundle, ecfg.pool),
            "embedder": lambda q: dense_ranks(q, vectors, model, ecfg.pool)}
    results = {name: arm_scores(rank, rows, bundle, ecfg.k) for name, rank in arms.items()}
    report(results, len(rows), ecfg.k, cfg.model, ABLATION_DIR / f"embedder_summary_{slug(cfg.model)}.json")


if __name__ == "__main__":
    main()