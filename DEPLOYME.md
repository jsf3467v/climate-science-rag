# Deploying the climate-arXiv RAG System

This serves the **measured** pipeline: `app.py` answers through
`synthesize.context()` and `answer()` — the same HyDE → BM25 → Claude-rerank →
grounded-synthesis path the evaluation scored — so the demo and the reported
numbers are the same system. No retrieval logic is re-implemented in the app.

The Space card metadata (title, `sdk`, `sdk_version`, `app_file`) lives in the
YAML front matter at the top of `README.md`. Set `sdk_version` there to the
Gradio version you install (`pip show gradio`).

## What the Space needs at run time

Two prebuilt artifacts, referenced by paths relative to the repo root:

- `index/bm25.joblib` — the BM25 index
- `data/chunks_norm.jsonl` — the normalized chunks (row order must match the index)

Everything else is code in `src/`, including `app.py`, which lives alongside the
modules it imports. The corpus and full-text stages are **not** needed to serve;
build the artifacts once, locally, and commit them. The README front matter sets
`app_file: src/app.py` so the Space launches the app from `src/`.

## One-time local build

```bash
pip install -r requirements.txt -r requirements-build.txt
export ANTHROPIC_API_KEY=...          # needed for eval and serving
export S2_API_KEY=...                 # optional, raises the Semantic Scholar limit

cd src
python corpus.py                      # selected.jsonl + manifest
python fulltext.py                    # chunks.jsonl
python clean.py                       # chunks_clean.jsonl
python normalize.py                   # chunks_norm.jsonl
python indexer.py                     # index/bm25.joblib
```

The corpus stage reads a live metadata snapshot and live citation counts and
uses a date-based recency cutoff, so it is not reproducible across time. Build it
once and keep the artifacts; do not re-run it unless you intend a new corpus
version (and then pin the snapshot revision and freeze the cutoff first).

## Commit layout

```
requirements.txt
README.md
src/            app.py config.py tokenizer.py jsonblock.py retrieve.py hyde.py rerank.py synthesize.py ...
index/          bm25.joblib
data/           chunks_norm.jsonl
```

`bm25.joblib` and `chunks_norm.jsonl` are large; track them with Git LFS:

```bash
git lfs install
git lfs track "index/bm25.joblib" "data/chunks_norm.jsonl"
git add .gitattributes requirements.txt README.md src index data
```

## Space settings

In **Settings → Variables and secrets**, add `ANTHROPIC_API_KEY` as a
**secret**. The app reads it from the environment; it is never entered in the UI.
The Space card takes an `emoji:` field; it is omitted from the front matter on
purpose — add one in the settings UI if you want a thumbnail.

## Cost and memory

- `QUERY_CHAR_CAP` (300) rejects pasted documents before any model call.
- `SESSION_QUERY_CAP` (20) caps paid calls per browser session. It is a soft cap
  that resets on reload — for a hard ceiling, set a spend limit on the Anthropic
  account. Both knobs are constants at the top of `app.py`.
- HyDE, rerank, and synthesis responses are cached on disk by query, so repeated
  questions cost nothing after the first.
- The BM25 index plus chunks load to about 2 GB resident (measured; the ~325 MB
  on disk expands as Python objects), and request handling adds little on top.
  The free CPU Basic tier (2 vCPU, 16 GB RAM) has ample headroom.

## Run locally

```bash
export ANTHROPIC_API_KEY=...
python src/app.py    # serves on http://127.0.0.1:7860
```
```
```
```