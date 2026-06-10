"""Reference-chunk filter for the climate-arXiv RAG project (src/).

This file removes chunks resembling reference lists or data tables, such as those containing LaTeX \section{References} or \begin{thebibliography},
especially from PDF fallback papers where the entire bibliography was stored in one body, which can clutter retrieval. During processing,
data/chunks.jsonl is streamed to identify and discard these chunks while keeping chunks.jsonl unchanged, writing the survivors to
data/chunks_clean.jsonl and outputting removed segments to data/dropped.jsonl for review. The filter uses prose density as its primary
discriminator rather than citation or digit count alone, where genuine prose maintains a normal stopword fraction, regardless of citations or quotes.
Whereas reference lists and numeric tables lack connectivity. Consequently, chunks that resemble prose are kept automatically, while those with
low prose density are tested against the citation and table rules.

    python clean.py
"""

from __future__ import annotations

import json
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import CleanConfig as Config

CITE_YEAR_RE = re.compile(r"(?:19|20)\d{2}[a-z]?\s*\)")
WORD_RE = re.compile(r"[a-z]+")
STOP = ENGLISH_STOP_WORDS


def stopword_fraction(text: str) -> float:
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(w in STOP for w in words) / len(words)


def reference_like(text: str, cfg: Config) -> bool:
    if stopword_fraction(text) >= cfg.prose_floor:
        return False
    low = text.lower()
    dois = low.count("doi")
    etal = low.count("et al")
    cites = dois + etal + len(CITE_YEAR_RE.findall(text))
    if dois >= cfg.doi_max or etal >= cfg.etal_max or cites >= cfg.cite_max:
        return True
    tokens = text.split()
    if not tokens:
        return False
    numeric = sum(any(c.isdigit() for c in t) for t in tokens)
    return numeric / len(tokens) > cfg.numeric_ratio


def main() -> None:
    cfg = Config()
    kept = dropped = 0
    with cfg.chunks_path.open() as src, cfg.clean_path.open("w") as keep, cfg.dropped_path.open("w") as drop:
        for line in src:
            if reference_like(json.loads(line)["text"], cfg):
                drop.write(line)
                dropped += 1
            else:
                keep.write(line)
                kept += 1
    total = kept + dropped
    pct = 100 * dropped / total if total else 0.0
    print(f"chunks: {total}  kept: {kept}  dropped: {dropped} ({pct:.1f}%)  -> chunks_clean.jsonl")


if __name__ == "__main__":
    main()