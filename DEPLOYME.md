# Deploying the climate-arXiv RAG System

This serves as a guide to run the model. Two surfaces are available. `app.py` is
the Gradio demo, and `src/api.py` is the programmatic service. Both answer through
`synthesize.context()` and `answer()`, the same HyDE-expanded BM25, Claude rerank,
and grounded synthesis path the evaluation scored, so both surfaces and the reported
numbers are the same system. No retrieval logic is re-implemented in either one.

Either surface runs locally, and the API also runs as a container. Hosting the
Gradio demo on Hugging Face is a separate step covered at the end, and it is
optional.

## What the system needs at run time

Two prebuilt artifacts are referenced by paths relative to the repo root, and both
surfaces read the same two.

- `index/bm25.joblib`, the BM25 index
- `data/chunks_norm.jsonl`, the normalized chunks (row order must match the index)

Everything else is code in `src/`, including `app.py`, which lives alongside the
modules it imports. The corpus and full-text stages are **not** needed to serve;
build the artifacts once, locally, and commit them.

## One-time local build

```bash
pip install -r requirements.txt -r requirements-build.txt
export ANTHROPIC_API_KEY=...          # needed for evaluation and serving
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

After a fresh clone, pull the artifacts before running anything, since Git LFS
leaves pointer files in their place.

```bash
git lfs pull
```

## Running the Gradio app

```bash
export ANTHROPIC_API_KEY=...
python src/app.py    # serves on http://127.0.0.1:7860
```

## Serving the API

The service in `src/api.py` reads the same two artifacts as the Gradio app and
loads them once at startup rather than per request.

```bash
pip install -r requirements-api.txt
export ANTHROPIC_API_KEY=...
uvicorn api:app --app-dir src        # serves on http://127.0.0.1:8000
```

Confirm the corpus loaded before anything else. A missing Git LFS pull is the
usual cause of a healthy service that returns nothing.

```bash
curl localhost:8000/health           # expect chunks: 118903
```

`GET /health` reports readiness and touches no model. `POST /search` runs BM25
retrieval alone and costs nothing per call. `POST /answer` runs the full reranked
path and returns the answer, its cited passages, and a `covered` flag that is
false when the system declines. The generated specification is at
`/openapi.json`, with an interactive interface at `/docs`.

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "what limits sea level projections under warming"}'
```

Route tests stub the retrieval and synthesis paths, so they make no network call
and take about a second.

```bash
python -m pytest tests/test_api.py -q
```

## Serving the API in a container

`.dockerignore` keeps `index/` and `data/` out of the build, since baking 325 MB
of artifacts into the image would slow every rebuild for no gain. The compose
file mounts them read-only at run time instead, so the image stays small and the
artifacts stay in one place on disk.

```bash
export ANTHROPIC_API_KEY=...
docker compose up api                # serves on http://127.0.0.1:8000
docker compose run --rm test         # route tests inside the container
```

Two failure modes are worth knowing. A `chunks` count of zero at `/health` means
the volume mounts did not find the artifacts, usually because the command ran
outside the repo root. A container that exits during startup usually means the
memory ceiling is too low, since the index expands to about 2 GB resident. Docker
Desktop ships with enough headroom by default, though a reduced limit will not
hold it.

The image builds for the host architecture. Add `--platform linux/amd64` to the
build only when the target runs on Intel hardware.

## Commit layout

```
requirements.txt
requirements-api.txt
README.md
Dockerfile docker-compose.yml .dockerignore
src/            app.py api.py schemas.py config.py tokenizer.py retrieve.py hyde.py rerank.py synthesize.py ...
tests/          test_api.py ...
index/          bm25.joblib
data/           chunks_norm.jsonl
```

`bm25.joblib` and `chunks_norm.jsonl` are large, so track them with Git LFS.

```bash
git lfs install
git lfs track "index/bm25.joblib" "data/chunks_norm.jsonl"
git add .gitattributes requirements.txt requirements-api.txt README.md \
        Dockerfile docker-compose.yml .dockerignore src tests index data
```

## Hosting the Gradio demo on Hugging Face

This step is optional, and the repository is not currently deployed as a Space.
Hugging Face now requires a PRO subscription to host a Gradio Space on the free
CPU tier, so creating one returns a payment error until that subscription is
active. Everything above runs without it.

The Space card metadata (title, `sdk`, `sdk_version`, `app_file`) lives in the
YAML front matter at the top of `README.md`. Set `sdk_version` there to the
Gradio version you install (`pip show gradio`), and note that Hugging Face
installs Gradio from that field rather than from `requirements.txt`, so a
mismatch between the two is the most common first-build failure. The front matter
sets `app_file: src/app.py` so the Space launches the app from `src/`.

A Space installs from `requirements.txt` alone, so `requirements-api.txt` never
reaches it and the API dependencies stay out of the deployed image. The
`Dockerfile` at the root is likewise ignored, since Hugging Face builds from a
Dockerfile only when the front matter declares `sdk: docker`.

Push to the Space as a second remote, which leaves the GitHub remote untouched.

```bash
hf auth login
hf repos create climate-science-rag --repo-type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/<user>/climate-science-rag
git push space main
```

In **Settings**, under **Variables and secrets**, add `ANTHROPIC_API_KEY` as a
**secret**. The app reads it from the environment, and it is never entered in the
interface. The Space card takes an `emoji:` field. It is omitted from the front matter on
purpose, so add one in the settings page if you want a thumbnail.

## Cost and memory

- `QUERY_CHAR_CAP` (300) rejects pasted documents before any model call.
- `SESSION_QUERY_CAP` (20) caps paid calls per browser session. It is a soft cap
  that resets on reload. For a hard ceiling, set a spend limit on the Anthropic
  account and leave auto-reload disabled, which caps total spend at the credits
  already purchased. Both knobs are constants at the top of `app.py`.
- HyDE, rerank, and synthesis responses are cached on disk by query, so repeated
  questions cost nothing after the first.
- The BM25 index plus chunks load to about 2 GB resident (measured; the
  325 MB on disk expands as Python objects), and request handling adds little on
  top. A CPU tier with 16 GB of memory has ample headroom.
