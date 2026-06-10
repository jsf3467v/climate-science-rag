"""Question-set generator for the climate-arXiv RAG evaluation (evaluation/).

Samples the chunked corpus stratified by section type and capped per paper
(seeded), then has Claude generate one abstracted question per sampled chunk,
where the answer resides within that chunk, along with a brief reference answer. Each entry records its
ground-truth source chunk which the retrieval metrics evaluate against and the lexical overlap between
the question and that chunk, ensuring that the keyword-matching
edge for synthetic questions given to BM25 is transparent. It writes to questions.jsonl, appending
new entries and skipping chunks already processed so a crash doesn't require reprocessing. The file is
stored outside src/, and it adds src/ to the path for importing shared config and tokenizer.


    export ANTHROPIC_API_KEY=...
    python questions.py            # --n overrides the question count
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import EvalConfig as Config   # noqa: E402  (needs src/ on the path, set above)
from jsonblock import json_block          # noqa: E402
from tokenizer import tokens              # noqa: E402

SECTION_TYPES = {
    "introduction": "intro", "background": "intro",
    "method": "methods", "data": "methods", "experiment": "methods",
    "result": "results", "finding": "results",
    "discussion": "discussion", "conclusion": "discussion",
}
SYSTEM = ("You write evaluation questions for a climate-science retrieval system. "
          "Ask about the scientific content -- a finding, method, mechanism, or claim "
          "-- never about funding, authorship, affiliations, or acknowledgments. "
          "Each question must be answerable from the passage alone, phrased "
          "conceptually, without copying its distinctive terms, numbers, or phrasing.")


def section_type(title: str) -> str:
    low = title.lower()
    for key, kind in SECTION_TYPES.items():
        if key in low:
            return kind
    return "other"


def corpus_chunks(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def next_chunk(bucket: list[dict], seen_paper: collections.Counter, cap: int) -> dict | None:
    while bucket:
        chunk = bucket.pop()
        if seen_paper[chunk["arxiv_id"]] < cap:
            seen_paper[chunk["arxiv_id"]] += 1
            return chunk
    return None


def sample_chunks(chunks: list[dict], cfg: Config) -> list[dict]:
    rng = random.Random(cfg.seed)
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for chunk in chunks:
        buckets[section_type(chunk.get("section") or "")].append(chunk)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    types = sorted(buckets)
    seen_paper: collections.Counter = collections.Counter()
    chosen: list[dict] = []
    while len(chosen) < cfg.n_questions and any(buckets.values()):
        for kind in types:
            picked = next_chunk(buckets[kind], seen_paper, cfg.per_paper)
            if picked is not None:
                chosen.append(picked)
            if len(chosen) >= cfg.n_questions:
                break
    return chosen


def overlap(question: str, chunk_text: str) -> float:
    q = set(tokens(question))
    return len(q & set(tokens(chunk_text))) / len(q) if q else 0.0


def prompt(chunk: dict, cfg: Config) -> str:
    return (f"Passage (from \"{chunk.get('title') or ''}\"):\n"
            f"{chunk['text'][:cfg.snippet_chars]}\n\n"
            f"Write one question a researcher might ask whose answer is in this passage, "
            f"and a one- to two-sentence reference answer. Abstract the wording: do not "
            f"reuse the passage's distinctive terms or numbers. "
            f'Return only JSON: {{"question": "...", "answer": "..."}}.')


def parsed(text: str) -> dict | None:
    obj = json_block(text)
    if not isinstance(obj, dict):
        return None
    try:
        question, answer = obj["question"].strip(), obj["answer"].strip()
    except (KeyError, AttributeError):
        return None
    return {"question": question, "answer": answer} if question and answer else None


def generated(chunk: dict, client: Anthropic, cfg: Config) -> dict | None:
    msg = client.messages.create(
        model=cfg.gen_model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(chunk, cfg)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return parsed(text)


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["source_chunk_id"] for line in path.open()}


def row(chunk: dict, qa: dict) -> dict:
    return {
        "question_id": chunk["chunk_id"],
        "question": qa["question"],
        "reference_answer": qa["answer"],
        "source_chunk_id": chunk["chunk_id"],
        "arxiv_id": chunk["arxiv_id"],
        "section": chunk.get("section"),
        "section_type": section_type(chunk.get("section") or ""),
        "overlap": round(overlap(qa["question"], chunk["text"]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the RAG evaluation question set.")
    ap.add_argument("--n", type=int, help="override the question count")
    args = ap.parse_args()
    cfg = Config()
    if args.n:
        cfg.n_questions = args.n
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    client = Anthropic()
    done = done_ids(cfg.questions_path)
    pending = [c for c in sample_chunks(corpus_chunks(cfg.chunks_path), cfg)
               if c["chunk_id"] not in done]
    written = 0
    with cfg.questions_path.open("a") as fh:
        for chunk in pending:
            qa = generated(chunk, client, cfg)
            if qa is None:
                continue
            fh.write(json.dumps(row(chunk, qa)) + "\n")
            fh.flush()
            written += 1
    print(f"questions: {len(done) + written}  new: {written}  -> {cfg.questions_path.name}")


if __name__ == "__main__":
    main()