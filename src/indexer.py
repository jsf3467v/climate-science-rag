"""
Builds one from-scratch lexical index over the chunked corpus, no pretrained
models. This is a BM25 (rank_bm25) over tokenized chunks.

Reads ../data/chunks_norm.jsonl (row order is the canonical chunk order that
retrieval relies on) and writes index/bm25.joblib. The artifact is written once
and skipped if present, so a crash resumes; --refresh rebuilds.

    pip install rank_bm25 scikit-learn
    python indexer.py              # --refresh rebuilds the index
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from rank_bm25 import BM25Okapi

from config import IndexerConfig as Config
from tokenizer import tokens


def chunk_texts(path: Path) -> list[str]:
    with path.open() as fh:
        return [json.loads(line)["text"] for line in fh]


def bm25_index(texts: list[str], cfg: Config) -> BM25Okapi:
    corpus = [tokens(t) for t in texts]
    return BM25Okapi(corpus, k1=cfg.bm25_k1, b=cfg.bm25_b)


def manifest(texts: list[str], cfg: Config) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"bm25_k1": cfg.bm25_k1, "bm25_b": cfg.bm25_b,
                   "chunks_path": str(cfg.chunks_path)},
        "counts": {"chunks": len(texts)},
        "artifacts": ["bm25.joblib"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the BM25 index over the chunked corpus.")
    ap.add_argument("--refresh", action="store_true", help="rebuild the index artifact")
    cfg = Config()
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = cfg.index_dir / "bm25.joblib"
    if ap.parse_args().refresh:
        bm25_path.unlink(missing_ok=True)

    texts = chunk_texts(cfg.chunks_path)
    print(f"chunks: {len(texts)}")

    if not bm25_path.exists():
        joblib.dump(bm25_index(texts, cfg), bm25_path)
        print("bm25: built")

    (cfg.index_dir / "index_manifest.json").write_text(json.dumps(manifest(texts, cfg), indent=2))
    print(f"index -> {cfg.index_dir.name}/")


if __name__ == "__main__":
    main()