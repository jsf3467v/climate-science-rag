"""Stage 2 full-text fetch and chunking for the climate-arXiv RAG project (src/).

For each selected paper it pulls the LaTeX source from arXiv, recovers the
section structure from the markup, and converts each section to text; if the
source is missing or unparsable it falls back to the PDF via PyMuPDF as a
single body. Sections are then split into overlapping word windows, each chunk
carrying provenance (arXiv id, section, indices, source, license).

Reads selected.jsonl from data/ and appends to data/chunks.jsonl. Chunking is
per-paper and append-flushed, so a crash resumes from the last finished paper.
Fetching is deliberately serial with backoff to respect arXiv; no model is used
anywhere here.

    pip install pylatexenc pymupdf requests
    python fulltext.py             # --refresh re-chunks from scratch
"""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import logging
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import fitz
import requests
from pylatexenc.latex2text import LatexNodes2Text

from config import FulltextConfig as Config, rooted

logging.getLogger("pylatexenc").setLevel(logging.ERROR)
fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)
SESSION = requests.Session()
SECTION_RE = re.compile(r"\\(?:sub)?section\*?\{([^}]*)\}")
NOISE = ("references", "bibliography", "acknowledg")
L2T = LatexNodes2Text()


def jsonl_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def chunked_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["arxiv_id"] for line in path.open()}


def payload(url: str, cfg: Config) -> bytes | None:
    wait = cfg.backoff_start
    for _ in range(cfg.backoff_tries):
        try:
            resp = SESSION.get(url, headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout)
            if resp.status_code in (429, 503):
                time.sleep(wait)
                wait = min(wait * 2, cfg.backoff_max)
                continue
            return resp.content if resp.status_code == 200 else None
        except requests.RequestException:
            time.sleep(wait)
            wait = min(wait * 2, cfg.backoff_max)
    return None


def main_tex(blob: bytes) -> str | None:
    if blob[:4] == b"%PDF":
        return None
    try:
        tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
        texts = [tar.extractfile(m).read().decode("utf-8", "ignore")
                 for m in tar.getmembers() if m.isfile() and m.name.endswith(".tex")]
    except tarfile.TarError:
        try:
            texts = [gzip.decompress(blob).decode("utf-8", "ignore")]
        except OSError:
            return None
    if not texts:
        return None
    return max(texts, key=lambda t: t.count("\\section") + 100 * ("\\documentclass" in t))


def tex_body(tex: str) -> str:
    start = tex.find("\\begin{document}")
    if start == -1:
        return tex
    end = tex.rfind("\\end{document}")
    return tex[start + len("\\begin{document}"):end if end != -1 else len(tex)]


def section_spans(body: str) -> list[tuple[str, str]]:
    marks = list(SECTION_RE.finditer(body))
    spans = []
    head = body[:marks[0].start()] if marks else body
    if head.strip():
        spans.append(("", head))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        spans.append((m.group(1), body[m.end():end]))
    return spans


def plain(raw: str) -> str:
    try:
        return L2T.latex_to_text(raw).strip()
    except Exception:
        return raw.strip()


def noise_section(title: str) -> bool:
    low = title.lower()
    return any(n in low for n in NOISE)


def latex_sections(rec: dict, cfg: Config) -> list[tuple[str, str]] | None:
    blob = payload(cfg.eprint_url.format(id=rec["arxiv_id"]), cfg)
    if blob is None:
        return None
    tex = main_tex(blob)
    if tex is None:
        return None
    spans = section_spans(tex_body(tex))
    out = [(title, plain(raw)) for title, raw in spans if not noise_section(title)]
    return [(t, txt) for t, txt in out if txt] or None


def pdf_sections(rec: dict, cfg: Config) -> list[tuple[str, str]] | None:
    blob = payload(cfg.pdf_url.format(id=rec["arxiv_id"]), cfg)
    if blob is None or blob[:4] != b"%PDF":
        return None
    try:
        doc = fitz.open(stream=blob, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc).strip()
    except Exception:
        return None
    return [("body", text)] if text else None


def paper_sections(rec: dict, cfg: Config) -> tuple[list[tuple[str, str]], str]:
    secs = latex_sections(rec, cfg)
    if secs:
        return secs, "latex"
    secs = pdf_sections(rec, cfg)
    if secs:
        return secs, "pdf"
    return [], "failed"


def chunk_row(rec: dict, title: str, s_idx: int, c_idx: int, piece: list[str], kind: str) -> dict:
    sid = rec["arxiv_id"]
    return {
        "chunk_id": f"{sid}:{s_idx}:{c_idx}",
        "arxiv_id": sid,
        "title": rec.get("title"),
        "section": title,
        "section_index": s_idx,
        "chunk_index": c_idx,
        "text": " ".join(piece),
        "n_words": len(piece),
        "source": kind,
        "license": rec.get("license"),
        "abs_url": rec.get("abs_url"),
    }


def chunks(sections: list[tuple[str, str]], rec: dict, cfg: Config, kind: str) -> list[dict]:
    step = max(1, cfg.chunk_words - cfg.overlap_words)
    out = []
    for s_idx, (title, text) in enumerate(sections):
        words = text.split()
        c_idx = 0
        for start in range(0, len(words), step):
            piece = words[start:start + cfg.chunk_words]
            if len(piece) < cfg.min_words and start > 0:
                break
            out.append(chunk_row(rec, title, s_idx, c_idx, piece, kind))
            c_idx += 1
    return out


def chunk_stats(path: Path) -> dict:
    source = {}
    total = 0
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            source[row["arxiv_id"]] = row["source"]
            total += 1
    kinds = list(source.values())
    return {"papers": len(source), "chunks": total,
            "latex": kinds.count("latex"), "pdf": kinds.count("pdf")}


def manifest(rows: list[dict], stats: dict, cfg: Config) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"chunk_words": cfg.chunk_words, "overlap_words": cfg.overlap_words,
                   "min_words": cfg.min_words, "selected_path": rooted(cfg.selected_path)},
        "counts": {"papers_selected": len(rows), "papers_chunked": stats["papers"],
                   "failed": len(rows) - stats["papers"], "chunks": stats["chunks"],
                   "latex": stats["latex"], "pdf": stats["pdf"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch full text and chunk the selected corpus.")
    ap.add_argument("--refresh", action="store_true", help="re-chunk from scratch (clear chunks.jsonl)")
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if ap.parse_args().refresh:
        cfg.chunks_path.unlink(missing_ok=True)

    rows = jsonl_rows(cfg.selected_path)
    done = chunked_ids(cfg.chunks_path)
    pending = [r for r in rows if r["arxiv_id"] not in done]
    print(f"papers: {len(rows)}  done: {len(done)}  pending: {len(pending)}")

    kinds = collections.Counter()
    with cfg.chunks_path.open("a") as fh:
        for i, rec in enumerate(pending, 1):
            sections, kind = paper_sections(rec, cfg)
            kinds[kind] += 1
            for ch in chunks(sections, rec, cfg, kind):
                fh.write(json.dumps(ch) + "\n")
            fh.flush()
            if i % cfg.progress_every == 0:
                print(f"  {i}/{len(pending)}  latex={kinds['latex']} pdf={kinds['pdf']} failed={kinds['failed']}")
            time.sleep(cfg.delay)

    stats = chunk_stats(cfg.chunks_path)
    (cfg.out_dir / "chunks_manifest.json").write_text(json.dumps(manifest(rows, stats, cfg), indent=2))
    print(f"chunks: {stats['chunks']}  papers_chunked: {stats['papers']}  failed: {len(rows) - stats['papers']}  ->  {cfg.out_dir.name}/")


if __name__ == "__main__":
    main()