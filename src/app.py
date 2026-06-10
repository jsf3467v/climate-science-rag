"""Gradio app for the climate-arXiv RAG system (repo root).

The system serves the measured pipeline, which includes HyDE-expanded BM25, Claude rerank, and grounded
synthesis with citations—behind a single question box. It calls
synthesize.context() and answer(), following the exact process used during evaluation, so users see exactly
what was measured. The index and chunks are loaded once at startup.
The Anthropic key is retrieved from the environment (set as a Space secret, not entered in the UI).
A session-based counter and a length limit restrict usage of the paid
model; any query that exceeds a guard threshold is prevented from reaching the API.


    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HydeConfig, RerankerConfig, RetrieveConfig, SynthConfig   # noqa: E402
from retrieve import index_artifacts                                         # noqa: E402
from synthesize import answer, context                                       # noqa: E402

QUERY_CHAR_CAP = 300       # one focused question, not a pasted document
SESSION_QUERY_CAP = 20     # paid calls per browser session before a reload is required

CLIENT = Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
BUNDLE = index_artifacts(RetrieveConfig())
RCFG, HCFG, SCFG = RerankerConfig(), HydeConfig(), SynthConfig()

INTRO = ("# Climate-science arXiv RAG\n"
         "Grounded answers with citations over about 3,000 climate papers. "
         "Retrieval is BM25 with HyDE query expansion and a Claude reranker; "
         "answers cite only the retrieved passages and say so plainly when the "
         "corpus does not cover the question.")


def sources(chunks: list[dict]) -> str:
    lines = [f"[{i}] {c.get('title') or c['arxiv_id']} - {c.get('abs_url') or c['arxiv_id']}"
             for i, c in enumerate(chunks)]
    return "**Sources**\n\n" + "\n\n".join(lines)


def reply(query: str, used: int):
    query = (query or "").strip()
    if CLIENT is None:
        return "Set ANTHROPIC_API_KEY as a Space secret to enable answers.", "", used
    if not query:
        return "Ask a question about the climate-science corpus.", "", used
    if len(query) > QUERY_CHAR_CAP:
        return f"Keep the question under {QUERY_CHAR_CAP} characters.", "", used
    if used >= SESSION_QUERY_CAP:
        return "Session limit reached. Reload the page to start a new session.", "", used
    try:
        chunks = context(query, BUNDLE, CLIENT, RCFG, HCFG)
        text = answer(query, chunks, CLIENT, SCFG)
    except Exception as exc:
        return f"The model call did not complete: {exc}", "", used
    return text, sources(chunks), used + 1


def page() -> gr.Blocks:
    with gr.Blocks(title="Climate arXiv RAG") as demo:
        gr.Markdown(INTRO)
        used = gr.State(0)
        box = gr.Textbox(label="Question",
                         placeholder="what limits sea level projections under warming")
        ask = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        src = gr.Markdown()
        ask.click(reply, [box, used], [out, src, used])
        box.submit(reply, [box, used], [out, src, used])
    return demo


demo = page()

if __name__ == "__main__":
    demo.launch()