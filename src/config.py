"""Shared configuration and disk cache for the climate-arXiv RAG project (src/).

This file contains the repository path anchors, each stage's Config, and the on-disk cache that the
model stages access. Paths are resolved relative to this file's location, ensuring they remain valid
regardless of how or where the project is checked out or which script imports them. Each
script imports its specific config, typically aliased as Config, such as:

    from config import IndexerConfig as Config

The cache consists of one SQLite file per stage (hyde, rerank, synth, judge). The keys are based on the content hashes that the call
sites generate. A model result is saved immediately upon creation, allowing a crash to restart from the last cached call without reprocessing it.

"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX = ROOT / "index"
EVAL = ROOT / "evaluation"


def rooted(path: Path) -> str:
    """A path written relative to ROOT, so manifests never serialize an absolute, machine-specific path."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


def connection(path: Path) -> sqlite3.Connection:
    """A fresh cache connection with its one table ensured. Opening per call keeps
    the cache correct across threads, including the app's worker threads, and the
    cost is trivial next to a model call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return conn


def cached(path: Path, key: str, miss):
    """The stored value for key, or miss() computed once, stored, and returned.
    miss() runs with no connection open, so a slow model call never holds a lock.
    Empty or failed results are not stored, so they recompute next run rather than
    freezing a bad value. Values round-trip through JSON, preserving str/list/dict."""
    conn = connection(path)
    try:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if row:
        return json.loads(row[0])
    value = miss()
    if value:
        conn = connection(path)
        try:
            conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))
            conn.commit()
        finally:
            conn.close()
    return value


@dataclass
class CorpusConfig:
    categories: tuple[str, ...] = ("physics.ao-ph", "physics.geo-ph")
    exclude_primary: tuple[str, ...] = ("astro-ph", "physics.space-ph")
    date_start: str = "2015-01-01"
    date_end: str = "2026-01-01"
    keywords: tuple[str, ...] = (
        "climate", "warming", "atmospher", "ocean", "carbon", "emission",
        "precipitation", "temperature", "sea level", "greenhouse",
        "aerosol", "monsoon", "cyclone", "reanalysis",
    )
    min_citations: int = 3            # 0 disables the citation gate
    recency_grace_years: int = 2      # recent papers bypass the citation gate
    target_size: int = 3000
    snapshot_repo: str = "librarian-bots/arxiv-metadata-snapshot"
    out_dir: Path = field(default_factory=lambda: DATA)


@dataclass
class FulltextConfig:
    chunk_words: int = 250
    overlap_words: int = 40
    min_words: int = 30
    delay: float = 1.0
    timeout: int = 30
    backoff_start: float = 5.0
    backoff_max: float = 120.0
    backoff_tries: int = 6
    progress_every: int = 250
    eprint_url: str = "https://arxiv.org/e-print/{id}"
    pdf_url: str = "https://arxiv.org/pdf/{id}"
    user_agent: str = "climate-rag-portfolio/1.0"
    out_dir: Path = field(default_factory=lambda: DATA)
    selected_path: Path = field(default_factory=lambda: DATA / "selected.jsonl")
    chunks_path: Path = field(default_factory=lambda: DATA / "chunks.jsonl")


@dataclass
class CleanConfig:
    prose_floor: float = 0.32   # >= this stopword fraction -> prose, kept regardless
    doi_max: int = 3            # >= this many DOI mentions -> reference list
    etal_max: int = 4           # >= this many "et al" -> reference list
    cite_max: int = 6           # >= this many citation markers (doi + et al + "(year)") -> reference list
    numeric_ratio: float = 0.5  # > this fraction of tokens carrying a digit -> data table
    out_dir: Path = field(default_factory=lambda: DATA)
    chunks_path: Path = field(default_factory=lambda: DATA / "chunks.jsonl")
    clean_path: Path = field(default_factory=lambda: DATA / "chunks_clean.jsonl")
    dropped_path: Path = field(default_factory=lambda: DATA / "dropped.jsonl")


@dataclass
class NormalizeConfig:
    chunks_path: Path = field(default_factory=lambda: DATA / "chunks_clean.jsonl")
    out_path: Path = field(default_factory=lambda: DATA / "chunks_norm.jsonl")


@dataclass
class IndexerConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    chunks_path: Path = field(default_factory=lambda: DATA / "chunks_norm.jsonl")
    index_dir: Path = field(default_factory=lambda: INDEX)


@dataclass
class RetrieveConfig:
    pool: int = 100        # BM25 candidates pulled before diversity + truncation
    per_paper: int = 2     # max chunks from one paper in the final results
    top_k: int = 10        # results returned
    index_dir: Path = field(default_factory=lambda: INDEX)
    chunks_path: Path = field(default_factory=lambda: DATA / "chunks_norm.jsonl")


@dataclass
class RerankerConfig:
    model: str = "claude-sonnet-4-6"   # set to whatever model you have access to
    temperature: float = 0.0           # deterministic reranking
    candidates: int = 50               # pulled from retrieval, then reordered (was 25; widened to expose deeper reachable gold)
    top_k: int = 10                    # returned after reranking
    snippet_chars: int = 600           # candidate text shown to the model
    max_tokens: int = 2048             # one JSON object per candidate -> must scale with `candidates` (50 ~ 1.3k tokens; 1024 truncated the array)
    cache: bool = True
    cache_path: Path = field(default_factory=lambda: DATA / "rerank_cache.db")


@dataclass
class HydeConfig:
    model: str = "claude-sonnet-4-6"   # set to whatever model you have access to
    temperature: float = 0.0           # deterministic expansion
    max_tokens: int = 256
    max_words: int = 120               # target passage length, asked in the prompt
    echo_query: bool = True            # prepend the original query to the passage
    cache: bool = True
    cache_path: Path = field(default_factory=lambda: DATA / "hyde_cache.db")


@dataclass
class EvalConfig:
    gen_model: str = "claude-sonnet-4-6"   # writes the question set
    n_questions: int = 150                 # target sample; ~4 fail generation -> 146 realized in questions.jsonl
    per_paper: int = 1                     # cap per paper -> coverage, not domination
    snippet_chars: int = 1200              # chunk text shown to the generator
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42
    k: int = 10                            # recall@k / MRR@k cutoff
    pool: int = 100                        # BM25 candidates before truncation
    configs: tuple[str, ...] = ("bm25", "hyde", "rerank", "rerank_nohyde")   # rerank_nohyde isolates HyDE's pool contribution
    chunks_path: Path = field(default_factory=lambda: DATA / "chunks_norm.jsonl")
    out_dir: Path = field(default_factory=lambda: EVAL)
    questions_path: Path = field(default_factory=lambda: EVAL / "questions.jsonl")
    results_path: Path = field(default_factory=lambda: EVAL / "results.jsonl")
    summary_path: Path = field(default_factory=lambda: EVAL / "summary.json")


@dataclass
class SynthConfig:
    model: str = "claude-sonnet-4-6"   # generation model - same family as HyDE/rerank
    temperature: float = 0.0
    max_tokens: int = 700
    snippet_chars: int = 1200          # chunk text shown in the context
    cache: bool = True
    cache_path: Path = field(default_factory=lambda: DATA / "synth_cache.db")


@dataclass
class JudgeConfig:
    model: str = "claude-opus-4-6"
    temperature: float = 0.0
    max_tokens: int = 2048
    snippet_chars: int = 1200          # passage text shown to the judge
    cache: bool = True
    cache_path: Path = field(default_factory=lambda: DATA / "judge_cache.db")
    questions_path: Path = field(default_factory=lambda: EVAL / "questions.jsonl")
    results_path: Path = field(default_factory=lambda: EVAL / "tier2_results.jsonl")
    summary_path: Path = field(default_factory=lambda: EVAL / "tier2_summary.json")
