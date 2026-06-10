"""Text normalization for the climate-arXiv RAG project (src/).

The LaTeX-to-text conversion in fulltext.py leaves placeholder cruft in the
chunk text -- "<cit.>" for \\cite, a spaced "<graphics>" stub for figures,
section-symbol runs, and "====" rule lines -- uniform noise the reranker and
synthesis stages would otherwise read. This pass rewrites the text field of each
chunk, leaving real content (including math subscripts like CO_2) untouched, and
writes chunks_norm.jsonl. It runs after clean.py, so the filtered corpus is
preserved one-to-one; only the text is scrubbed.

    python normalize.py
"""

from __future__ import annotations

import json
import re

from config import NormalizeConfig as Config

CIT_RE = re.compile(r"<\s*cit\.?\s*>")
GRAPHICS_RE = re.compile(r"<\s*g\s*r\s*a\s*p\s*h\s*i\s*c\s*s\s*>")
SECTION_RE = re.compile("\u00a7(?:\\s*\\.\\s*\u00a7)*")
RULE_RE = re.compile(r"={4,}")
SPACE_PUNCT_RE = re.compile(r"\s+([.,;:])")
WS_RE = re.compile(r"\s+")


def scrub(text: str) -> str:
    for pattern in (CIT_RE, GRAPHICS_RE, SECTION_RE, RULE_RE):
        text = pattern.sub(" ", text)
    text = SPACE_PUNCT_RE.sub(r"\1", text)
    return WS_RE.sub(" ", text).strip()


def main() -> None:
    cfg = Config()
    total = changed = 0
    with cfg.chunks_path.open() as src, cfg.out_path.open("w") as dst:
        for line in src:
            row = json.loads(line)
            scrubbed = scrub(row["text"])
            changed += scrubbed != row["text"]
            row["text"] = scrubbed
            dst.write(json.dumps(row) + "\n")
            total += 1
    print(f"chunks: {total}  changed: {changed}  -> {cfg.out_path.name}")


if __name__ == "__main__":
    main()