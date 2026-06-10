"""Corpus building for the climate-arXiv RAG project (src/).

arXiv's live search API limits bulk metadata retrieval, especially tightened in early 2026.
Instead, a static arXiv metadata snapshot from Hugging Face is used, avoiding rate limits, ensuring
reproducibility, and including each paper's license for future hosting decisions. Semantic Scholar enhances
this data with citation counts, fields of study, and TLDR summaries. A keyword, citation, and recency
filter then selects specific IDs. The snapshot is obtained via the datasets library and cached locally on the first
run using Parquet format, which is much smaller than the raw JSON. During enrichment, data is appended to enriched.jsonl
in batches and throttling is managed so that a crash can resume the process. Outputs are stored in data/, and the full text
is retrieved in the next stage.


    pip install datasets requests
    export S2_API_KEY=...          # optional; raises the Semantic Scholar rate limit
    python corpus.py               # --refresh re-scans the snapshot and clears caches
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from datasets import load_dataset

from config import CorpusConfig as Config, rooted

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "title,year,citationCount,fieldsOfStudy,externalIds,tldr"
S2_BATCH_SIZE = 100               # endpoint accepts up to 500;


def jsonl_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def jsonl_dump(rows: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def paper_date(rec: dict) -> str:
    stamp = rec.get("update_date")
    if stamp is None:
        return ""
    return stamp.isoformat()[:10] if hasattr(stamp, "isoformat") else str(stamp)[:10]


def paper_match(rec: dict, cfg: Config, lo_year: int, hi_year: int) -> bool:
    cats = rec.get("categories", "").split()
    if set(cats).isdisjoint(cfg.categories):
        return False
    if cats and any(cats[0].startswith(x) for x in cfg.exclude_primary):
        return False
    day = paper_date(rec)
    year = int(day[:4]) if day[:4].isdigit() else 0
    if not lo_year <= year <= hi_year:
        return False
    text = f"{rec.get('title') or ''} {rec.get('abstract') or ''}".lower()
    return not cfg.keywords or any(k in text for k in cfg.keywords)


def snapshot_row(rec: dict) -> dict:
    sid = rec["id"]
    cats = rec.get("categories", "").split()
    return {
        "arxiv_id": sid,
        "title": (rec.get("title") or "").strip(),
        "abstract": (rec.get("abstract") or "").strip(),
        "authors": rec.get("authors"),
        "primary_category": cats[0] if cats else None,
        "categories": cats,
        "published": paper_date(rec) or None,
        "license": rec.get("license"),
        "abs_url": f"https://arxiv.org/abs/{sid}",
        "pdf_url": f"https://arxiv.org/pdf/{sid}",
    }


def candidates(cfg: Config, out_path: Path) -> list[dict]:
    lo_year, hi_year = int(cfg.date_start[:4]), int(cfg.date_end[:4])
    data = load_dataset(cfg.snapshot_repo, split="train")
    with out_path.open("w") as dst:
        for rec in data:
            if paper_match(rec, cfg, lo_year, hi_year):
                dst.write(json.dumps(snapshot_row(rec)) + "\n")
    out_path.with_suffix(".done").touch()
    return jsonl_rows(out_path)


def scholar_batch(ids: list[str], headers: dict) -> list:
    body = {"ids": [f"ARXIV:{i}" for i in ids]}
    wait = 2.0
    for _ in range(4):
        try:
            resp = requests.post(S2_BATCH_URL, params={"fields": S2_FIELDS},
                                 json=body, headers=headers, timeout=30)
            if resp.status_code == 429:
                time.sleep(wait)
                wait = min(wait * 2, 30.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            time.sleep(wait)
            wait = min(wait * 2, 30.0)
    return [None] * len(ids)


def scholar_fields(rows: list[dict], enriched_path: Path) -> list[dict]:
    done = {r["arxiv_id"] for r in jsonl_rows(enriched_path)} if enriched_path.exists() else set()
    pending = [r for r in rows if r["arxiv_id"] not in done]
    key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}
    delay = 1.1 if key else 3.5
    with enriched_path.open("a") as fh:
        for start in range(0, len(pending), S2_BATCH_SIZE):
            chunk = pending[start:start + S2_BATCH_SIZE]
            papers = scholar_batch([r["arxiv_id"] for r in chunk], headers)
            for rec, paper in zip(chunk, papers):
                tldr = (paper or {}).get("tldr") or {}
                rec["s2_found"] = paper is not None
                rec["s2_citations"] = (paper or {}).get("citationCount", 0)
                rec["s2_fields"] = (paper or {}).get("fieldsOfStudy") or []
                rec["s2_tldr"] = tldr.get("text")
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            time.sleep(delay)
    cache = {r["arxiv_id"]: r for r in jsonl_rows(enriched_path)}
    return [cache[r["arxiv_id"]] for r in rows if r["arxiv_id"] in cache]


def gate(rec: dict, cfg: Config, cutoff_year: int) -> bool:
    if cfg.min_citations <= 0:
        return True
    year = int(rec["published"][:4]) if rec.get("published") else 0
    return rec.get("s2_citations", 0) >= cfg.min_citations or year >= cutoff_year


def selection(rows: list[dict], cfg: Config) -> list[dict]:
    cutoff_year = datetime.now(timezone.utc).year - cfg.recency_grace_years
    kept = [r for r in rows if gate(r, cfg, cutoff_year)]
    kept.sort(key=lambda r: (r.get("s2_citations", 0), r.get("published") or ""), reverse=True)
    return kept[:cfg.target_size]


def manifest(candidate_rows: list[dict], enriched: list[dict], selected: list[dict], cfg: Config) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(cfg), "out_dir": rooted(cfg.out_dir)},
        "counts": {
            "candidates": len(candidate_rows),
            "enriched": len(enriched),
            "selected": len(selected),
            "scholar_found": sum(1 for r in enriched if r.get("s2_found")),
        },
        "note": "IDs, metadata, and license only; full text comes from the fetch stage.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the climate-arXiv corpus (IDs and metadata).")
    ap.add_argument("--refresh", action="store_true", help="re-scan the snapshot and clear all caches")
    ap.add_argument("--rescope", action="store_true", help="re-scan candidates but reuse the enrichment cache")
    args = ap.parse_args()

    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = cfg.out_dir / "candidates.jsonl"
    enriched_path = cfg.out_dir / "enriched.jsonl"
    done_marker = candidates_path.with_suffix(".done")

    if args.refresh:
        for p in (candidates_path, enriched_path, done_marker):
            p.unlink(missing_ok=True)
    elif args.rescope:
        for p in (candidates_path, done_marker):
            p.unlink(missing_ok=True)

    rows = jsonl_rows(candidates_path) if done_marker.exists() else candidates(cfg, candidates_path)
    print(f"candidates: {len(rows)}")

    enriched = scholar_fields(rows, enriched_path)
    print(f"enriched: {len(enriched)}  scholar_found: {sum(1 for r in enriched if r.get('s2_found'))}")

    selected = selection(enriched, cfg)
    jsonl_dump(selected, cfg.out_dir / "selected.jsonl")
    (cfg.out_dir / "selected_ids.txt").write_text("\n".join(r["arxiv_id"] for r in selected) + "\n")
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest(rows, enriched, selected, cfg), indent=2))
    print(f"selected: {len(selected)}  ->  {cfg.out_dir.name}/")


if __name__ == "__main__":
    main()