"""Tier 2 judged evaluation for the climate arXiv RAG project (evaluation/).

The system generates an answer for each question using the winning retrieval pipeline's
context and then evaluates it with an independent judge model based on three RAGAS-style
metrics. These metrics include faithfulness (measuring the proportion of the answer's
claims supported by the context), answer relevance (assessing how well the answer addresses the question), and
context precision (determining the proportion of retrieved passages that are relevant). The
judge must be different from the generation model, which is enforced below, to prevent it
from grading its own work. Embedding-based RAGAS is avoided because it
requires a pretrained embedder. Since everything except the LLM is built from scratch, each metric is
scored by an LLM. Judge calls are cached to allow resumption after crashes. It operates outside src/, so src/ is added to the path.


    export ANTHROPIC_API_KEY=...
    python synthesis_eval.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import JudgeConfig as Config, SynthConfig, RerankerConfig, HydeConfig, RetrieveConfig, cached  # noqa: E402
from jsonblock import json_block                # noqa: E402
from retrieve import index_artifacts, jsonl_rows   # noqa: E402
from synthesize import answer, context             # noqa: E402

METRICS = ("faithfulness", "answer_relevance", "context_precision")

FAITH_SYS = ("You are a strict evaluator. Given passages and an answer, judge what fraction of the "
             "answer's factual claims are directly supported by the passages. An answer that "
             "declines because the context lacks the information makes no claims; score it 1.0. "
             'Return only JSON {"score": <float 0-1>}.')
REL_SYS = ("You are a strict evaluator. Judge how fully the answer addresses the question, ignoring "
           'whether it is correct. Return only JSON {"score": <float 0-1>}.')
PREC_SYS = ("You are a strict evaluator. For each numbered passage, judge whether it is relevant to "
            'answering the question. Return only JSON {"relevant": [<0 or 1>, ...]} in passage order.')


def passages(chunks: list[dict], jcfg: Config) -> str:
    return "\n\n".join(f"[{i}] {c['text'][:jcfg.snippet_chars]}" for i, c in enumerate(chunks))


def clip(value) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def verdict(system: str, user: str, client: Anthropic, jcfg: Config) -> dict | None:
    msg = client.messages.create(model=jcfg.model, max_tokens=jcfg.max_tokens,
                                 temperature=jcfg.temperature, system=system,
                                 messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    return json_block(text)


def judgment(system: str, user: str, client: Anthropic, jcfg: Config) -> dict | None:
    if not jcfg.cache:
        return verdict(system, user, client, jcfg)
    # The key assumes temperature and max_tokens are fixed. Change either and clear judge_cache.db first, or stale verdicts are served.
    key = hashlib.sha1(f"{jcfg.model}|{system}|{user}".encode()).hexdigest()
    return cached(jcfg.cache_path, key, lambda: verdict(system, user, client, jcfg))


def faithfulness(ans: str, chunks: list[dict], client: Anthropic, jcfg: Config) -> float | None:
    user = f"Passages:\n{passages(chunks, jcfg)}\n\nAnswer:\n{ans}"
    obj = judgment(FAITH_SYS, user, client, jcfg)
    return clip(obj["score"]) if obj and "score" in obj else None


def answer_relevance(question: str, ans: str, client: Anthropic, jcfg: Config) -> float | None:
    user = f"Question: {question}\n\nAnswer:\n{ans}"
    obj = judgment(REL_SYS, user, client, jcfg)
    return clip(obj["score"]) if obj and "score" in obj else None


def flag(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "relevant"}:
            return 1
        if token in {"0", "false", "no", "irrelevant"}:
            return 0
        return None
    if isinstance(value, dict):
        for key in ("relevant", "relevance", "verdict", "label", "score"):
            if key in value:
                return flag(value[key])
    return None


def flags(obj) -> list[int] | None:
    if isinstance(obj, dict):
        seq = obj.get("relevant")
        if not isinstance(seq, list):
            seq = next((v for v in obj.values() if isinstance(v, list)), None)
    elif isinstance(obj, list):
        seq = obj
    else:
        seq = None
    if not seq:
        return None
    bits = [flag(x) for x in seq]
    return [b for b in bits if b is not None] or None


def context_precision(question: str, chunks: list[dict], client: Anthropic, jcfg: Config) -> float | None:
    user = f"Question: {question}\n\nPassages:\n{passages(chunks, jcfg)}"
    obj = judgment(PREC_SYS, user, client, jcfg)
    bits = flags(obj)
    if not bits:
        return None
    return sum(bits) / len(bits)


def cell(question: dict, bundle, client: Anthropic, scfg: SynthConfig,
         rcfg: RerankerConfig, hcfg: HydeConfig, jcfg: Config) -> dict:
    q = question["question"]
    chunks = context(q, bundle, client, rcfg, hcfg)
    ans = answer(q, chunks, client, scfg)
    return {"faithfulness": faithfulness(ans, chunks, client, jcfg),
            "answer_relevance": answer_relevance(q, ans, client, jcfg),
            "context_precision": context_precision(q, chunks, client, jcfg)}


def report(agg: dict, cnt: dict, fails: dict, n: int, jcfg: Config) -> None:
    metrics = {m: round(agg[m] / cnt[m], 4) if cnt[m] else 0.0 for m in METRICS}
    print("n=%d  judge=%s\n%s" % (n, jcfg.model, "".join(m.rjust(20) for m in METRICS)))
    print("".join(f"{metrics[m]:20.3f}" for m in METRICS))
    if any(fails.values()):
        print("parse failures: " + ", ".join(f"{m}={fails[m]}" for m in METRICS if fails[m]))
    jcfg.summary_path.write_text(json.dumps(
        {"n": n, "judge": jcfg.model, "metrics": metrics,
         "scored": {m: cnt[m] for m in METRICS},
         "parse_failures": {m: fails[m] for m in METRICS}}, indent=2))


def main() -> None:
    jcfg = Config()
    scfg = SynthConfig()
    if jcfg.model == scfg.model:
        raise SystemExit(f"judge model {jcfg.model} must differ from the generation model {scfg.model}")
    rcfg, hcfg = RerankerConfig(), HydeConfig()
    client = Anthropic()
    bundle = index_artifacts(RetrieveConfig())
    rows = jsonl_rows(jcfg.questions_path)
    if not rows:
        raise SystemExit(f"no questions at {jcfg.questions_path}")
    agg: dict[str, float] = collections.defaultdict(float)
    cnt: dict[str, int] = collections.defaultdict(int)
    fails: dict[str, int] = collections.defaultdict(int)
    with jcfg.results_path.open("w") as out:
        for question in rows:
            row = cell(question, bundle, client, scfg, rcfg, hcfg, jcfg)
            out.write(json.dumps({"question_id": question["question_id"], **row}) + "\n")
            for metric, value in row.items():
                if value is None:
                    fails[metric] += 1
                else:
                    agg[metric] += value
                    cnt[metric] += 1
    report(agg, cnt, fails, len(rows), jcfg)


if __name__ == "__main__":
    main()