"""HTTP service over the measured retrieval pipeline.

Three routes. /health reports readiness without touching the paid model.
/search runs BM25 retrieval only, so it costs nothing per call. /answer runs the
full reranked path and grounded synthesis, which is the path the evaluation
measured. The index is read once at startup and shared across requests.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HydeConfig, RerankerConfig, RetrieveConfig, SynthConfig  # noqa: E402
from retrieve import index_artifacts, search  # noqa: E402
from schemas import Grounded, Matches, Passage, Question, Status  # noqa: E402
from synthesize import SYSTEM, answer, context  # noqa: E402

DECLINED = SYSTEM.split("reply exactly: ")[1].split(". Do not")[0].strip()

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bundle"] = index_artifacts(RetrieveConfig())
    STATE["client"] = Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    STATE["retrieve"] = RetrieveConfig()
    STATE["rerank"] = RerankerConfig()
    STATE["hyde"] = HydeConfig()
    STATE["synth"] = SynthConfig()
    yield
    STATE.clear()


app = FastAPI(
    title="Climate arXiv retrieval",
    version="1.0.0",
    summary="Grounded answers over a climate-science arXiv corpus, cited to source passages.",
    lifespan=lifespan,
)


def passage(row: dict) -> Passage:
    return Passage(**{k: row.get(k) for k in Passage.model_fields})


def client_or_error() -> Anthropic:
    client = STATE.get("client")
    if client is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set.")
    return client


@app.get("/health", response_model=Status)
def health() -> Status:
    bundle = STATE.get("bundle")
    return Status(
        ready=bundle is not None,
        chunks=len(bundle.chunks) if bundle else 0,
        model_available=STATE.get("client") is not None,
    )


@app.post("/search", response_model=Matches)
def matches(question: Question) -> Matches:
    rows = search(question.query, STATE["bundle"], STATE["retrieve"])
    return Matches(query=question.query, passages=[passage(r) for r in rows])


@app.post("/answer", response_model=Grounded)
def grounded(question: Question) -> Grounded:
    client = client_or_error()
    try:
        chunks = context(question.query, STATE["bundle"], client,
                         STATE["rerank"], STATE["hyde"])
        text = answer(question.query, chunks, client, STATE["synth"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream model call failed: {exc}") from exc
    return Grounded(
        query=question.query,
        answer=text,
        covered=DECLINED.lower() not in text.lower(),
        passages=[passage(c) for c in chunks],
    )