"""Paired significance for Tier-1 retrieval. Reads per-question results.jsonl and
reports, for the comparisons the README makes, the mean difference with a 95%
paired-bootstrap CI and an exact McNemar p-value on the binary recall metrics.
When questions.jsonl is present, the HyDE contribution is also split by overlap
band. Writes significance.json plus a compact console table.

    python significance.py
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
METRICS = (("recall_chunk", True), ("recall_paper", True), ("mrr_chunk", False), ("mrr_paper", False))
PAIRS = (("bm25", "rerank"), ("rerank_nohyde", "rerank"), ("bm25", "hyde"))


def results(path):
    out = collections.defaultdict(dict)
    for line in path.open():
        r = json.loads(line)
        out[r["config"]][r["question_id"]] = r
    return out


def overlaps(path):
    return {json.loads(l)["question_id"]: json.loads(l)["overlap"] for l in path.open()}


def band(value):
    return "low" if value < 0.30 else ("high" if value >= 0.60 else "mid")


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired binary outcomes."""
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def interval(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def figures(a, b, binary):
    delta, lo, hi = interval(a, b)
    out = {"a": round(float(a.mean()), 4), "b": round(float(b.mean()), 4),
           "delta": round(delta, 4), "ci": [round(lo, 4), round(hi, 4)]}
    if binary:
        out["mcnemar_p"] = round(mcnemar(a, b), 4)
    return out


def comparison(res, ca, cb, qids, label):
    rec = {"a": ca, "b": cb, "band": label, "n": len(qids)}
    for metric, binary in METRICS:
        a = np.array([res[ca][q][metric] for q in qids])
        b = np.array([res[cb][q][metric] for q in qids])
        rec[metric] = figures(a, b, binary)
    return rec


def table(out):
    print(f"n={out['n']}  {out['n_boot']} paired bootstrap, exact McNemar on recall\n")
    print(f"{'comparison':24}{'band':5}{'metric':14}{'delta':>8}{'95% ci':>19}{'p':>9}")
    for c in out["comparisons"]:
        name = f"{c['b']} vs {c['a']}"
        for metric in ("recall_chunk", "recall_paper"):
            f = c[metric]
            ci = f"[{f['ci'][0]:+.3f}, {f['ci'][1]:+.3f}]"
            print(f"{name:24}{c['band']:5}{metric:14}{f['delta']:>+8.3f}{ci:>19}{f['mcnemar_p']:>9.4f}")


def report():
    res = results(BASE / "results.jsonl")
    qids = sorted(res["bm25"])
    out = {"n": len(qids), "k": 10, "n_boot": 10000, "seed": 0, "comparisons": []}
    for ca, cb in PAIRS:
        out["comparisons"].append(comparison(res, ca, cb, qids, "all"))
    qpath = BASE / "questions.jsonl"
    if qpath.exists():
        over = overlaps(qpath)
        for b in ("low", "mid", "high"):
            ids = [q for q in qids if band(over[q]) == b]
            out["comparisons"].append(comparison(res, "rerank_nohyde", "rerank", ids, b))
    dest = BASE / "significance.json"
    dest.write_text(json.dumps(out, indent=2))
    table(out)
    print(f"\nwrote {dest.name}")


if __name__ == "__main__":
    report()