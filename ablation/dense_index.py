"""FAISS dense index for the climate-arXiv RAG dense-retrieval comparison (ablation/).

The Mirrors indexer.py on the dense side encodes the chunked corpus using a
pretrained sentence-transformer and constructs an exact inner-product FAISS index with
normalized vectors (where inner product on normalized vectors equals cosine similarity).
The embedder, being pretrained, resides outside src/ to keep the shipped system lexical.
Encoding reuses the blocked, cached encoder from embedder_ablation, allowing a
crashed run to resume from the last saved block, with vectors shared across the
single-stage ablation. The index is generated once and is skipped if it already exists.


    pip install -r requirements.txt -r ablations/requirements-ablation.txt
    python ablation/dense_index.py --device mps
    python ablation/dense_index.py --device mps --model thenlper/gte-large --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
ABLATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import RetrieveConfig                                            # noqa: E402
from retrieve import index_artifacts                                        # noqa: E402
from embedder_ablation import AblationConfig as Config, slug, cached_embeddings  # noqa: E402


def index_path(cfg: Config) -> Path:
    return ABLATION / f"dense_{slug(cfg.model)}.faiss"


def faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])   # exact cosine on normalized vectors
    index.add(vectors)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FAISS dense index over the chunked corpus.")
    ap.add_argument("--device", default=Config.device, help="cpu, mps, or cuda")
    ap.add_argument("--model", default=Config.model, help="sentence-transformer model id")
    ap.add_argument("--refresh", action="store_true", help="rebuild the index artifact")
    args = ap.parse_args()
    cfg = Config(device=args.device, model=args.model)
    path = index_path(cfg)
    if args.refresh:
        path.unlink(missing_ok=True)

    bundle = index_artifacts(RetrieveConfig())
    if path.exists():
        print(f"index present: {path.name}  ({faiss.read_index(str(path)).ntotal} vectors)")
        return

    model = SentenceTransformer(cfg.model, device=cfg.device)
    vectors = cached_embeddings([c["text"] for c in bundle.chunks], model, cfg)
    if len(vectors) != len(bundle.chunks):
        raise SystemExit(f"alignment: {len(vectors)} vectors, {len(bundle.chunks)} chunks; clear {cfg.cache_dir / slug(cfg.model)}")
    faiss.write_index(faiss_index(np.ascontiguousarray(vectors, dtype="float32")), str(path))
    print(f"dense index: {len(vectors)} vectors  dim {vectors.shape[1]}  -> {path.name}")


if __name__ == "__main__":
    main()