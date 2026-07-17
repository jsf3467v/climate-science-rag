---
title: Climate arXiv RAG
sdk: gradio
sdk_version: 5.9.1
app_file: src/app.py
pinned: false
---

[![CI](https://github.com/jsf3467v/climate-science-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/climate-science-rag/actions/workflows/ci.yml)

# Climate Science arXiv RAG

The system answers climate science questions using only evidence it retrieves from a corpus of approximately 3,000 arXiv papers, never from the language model's own training. Each claim in an answer comes from a specific retrieved passage and is cited to it, so a reader can trace every statement back to its source paper. When the corpus holds no passage that addresses a question, the system declines rather than fills the gap from memory. Restricting every answer to cited source evidence in this way is what grounded means here, and the retrieval and reranking stages described below exist to place the most relevant passages in front of the model before it writes.

The system runs as an interactive chat interface built with Gradio, where a user asks a climate science question in plain language, and the interface drives the full pipeline described below.

The primary limitation is that only the language model is pretrained; the system does not include dense embedders or cross encoders. Retrieval depends solely on lexical methods like BM25, while the language model manages query expansion, reranking, synthesis, and evaluation. An ablation study below compares this approach to a robust pretrained embedder, showing that the reranker nearly recovers the embedder's advantage. Consequently, this constraint results in a lightweight, CPU-only, and easily inspectable retrieval system with minimal measured cost.

See `DEPLOYME.md` for build, deploy, and local run instructions, including how to launch the chat interface.

## Architecture

The pipeline consists of a series of single-purpose stages. The costly tasks, such as downloads and model calls, are cached or checkpointed, allowing a stop and rerun to resume from where it left off instead of repeating a run.

1. **Corpus** (`corpus.py`). Enumerates climate papers from a static arXiv metadata snapshot, which avoids live API rate limits and carries a clear license. It enriches them with Semantic Scholar citations and fields, then filters by keyword and by a citation and recency gate down to 3,000 papers.

2. **Full text and chunking** (`fulltext.py`). Fetches LaTeX source and recovers section structure, falling back to the PDF when source is missing. Sections are split into 250-word windows with 40-word overlap, and each chunk carries provenance. Approximately 119k chunks survive cleaning.

3. **Cleaning and normalization** (`clean.py`, `normalize.py`). A prose density gate drops reference lists and data table chunks first, while keeping genuine prose regardless of how many citations it contains. The remainder is filtered by citation structure, such as DOIs, et al., and parenthetical years, and by a numeric ratio test. Normalization then scrubs LaTeX-to-text placeholder artifacts and leaves real content untouched.

4. **Index and retrieval** (`indexer.py`, `retrieve.py`). A single BM25 index over a shared tokenizer, with no embeddings.

5. **Query expansion** (`hyde.py`). HyDE has the model write a short, plausible answer passage. Retrieval runs on that passage, prepended with the query, so lexical matching reaches domain terms that a paraphrased question never names.

6. **Reranking** (`rerank.py`). The model reorders the BM25-over-HyDE candidate pool listwise in a single call.

7. **Synthesis** (`synthesize.py`). The model answers from the reranked passages only, cites them by number, and declines when the answer is absent.

8. **Evaluation** (`evaluation/`). Tier 1 is judge-free and deterministic, computing recall and MRR at cutoff $k$ against each question's source chunk. Tier 2 uses an independent judge model that scores faithfulness, answer relevance, and context precision. The reranked path that deploys is the exact path the evaluation scores.

## Dependencies

- **Python**, CPU-only. No GPU anywhere in the served stack.

- **rank_bm25** for lexical retrieval, **scikit-learn** for the shared tokenizer stopword set, and **joblib** to persist the index.

- **pylatexenc** and **PyMuPDF** for LaTeX and PDF full-text extraction.

- **Hugging Face `datasets`** for the arXiv metadata snapshot and the **Semantic Scholar** API for citation metadata.

- **Anthropic Claude**. `claude-sonnet-4-6` for HyDE, reranking, synthesis, and writing the eval questions, and `claude-opus-4-6` as the independent Tier 2 judge, enforced to differ from the generation model so it never grades its own work.

- **Gradio** for the interactive chat app.

- **sentence-transformers** and **FAISS** appear only in the ablation (`ablation/`) to measure the cost of the lexical-only constraint, and neither enters the served system.

## Results

Tier 1 retrieval, $n=146$ questions, cutoff $k=10$.

| Configuration        | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| -------------------- | ----------------- | ----------------- | -------------- | -------------- |
| BM25                 | 0.384             | 0.651             | 0.251          | 0.477          |
| with HyDE            | 0.384             | 0.685             | 0.225          | 0.448          |
| with HyDE and rerank | 0.555             | 0.808             | 0.418          | 0.720          |
| rerank, no HyDE      | 0.527             | 0.822             | 0.396          | 0.712          |

The served path is BM25 over HyDE followed by the listwise reranker, shown in the row marked with HyDE and rerank, and it is the exact path the evaluation scores. The final row reranks a plain BM25 pool with HyDE removed, included to test whether query expansion contributes.

Chunk-level recall is a strict single-reference metric. Each question is scored against the one chunk it was written from, out of roughly 119k, so retrieving an equally good neighbor still counts as a miss. The paper-level numbers credit any chunk from the correct paper and give a fairer read of retrieval quality. The paper-level MRR shows that when the right paper is found, it usually sits at rank one or two.

The gain is concentrated in the reranker. Moving from BM25 to the reranked path raises chunk recall from 0.384 to 0.555 and paper MRR from 0.477 to 0.720, a step well beyond the roughly 0.08 confidence interval at this sample size. The reranker is the component that produces the result.

### Does HyDE contribute

Overall, HyDE and the no-HyDE rerank path are tied, with scores of 0.555 versus 0.527 for chunk recall and 0.808 versus 0.822 for paper recall, both within noise levels. An initial review of these combined figures suggested that HyDE could be eliminated. However, analyzing questions based on the lexical overlap between the question and its source chunk reveals that this conclusion is incorrect, as the overall similarity masks two opposing effects. Low-overlap questions, with less than 0.30 overlap and constituting a third of the set at 54 out of 146, represent the failure mode HyDE was designed to address. Conversely, high-overlap questions, totaling 14 out of 146, already share vocabulary with the corpus and do not require additional help.

Recall by overlap band follows for the reranked paths, given as chunk recall then paper recall.

| Configuration        | low overlap (n=54) | mid overlap (n=78) | high overlap (n=14) |
| -------------------- | ------------------ | ------------------ | ------------------- |
| BM25                 | 0.093 / 0.500      | 0.487 / 0.705      | 0.929 / 0.929       |
| with HyDE and rerank | 0.352 / 0.815      | 0.654 / 0.795      | 0.786 / 0.857       |
| rerank, no HyDE      | 0.185 / 0.685      | 0.692 / 0.885      | 0.929 / 1.000       |

On the paraphrased band, HyDE nearly doubles reranked chunk recall, from 0.185 to 0.352, and raises paper recall by thirteen points, from 0.685 to 0.815. It performs slightly worse on high-overlap questions, where the expansion only adds plausible competitors to an already well-matched query. The overall neutrality is the average of this targeted gain and the small loss. HyDE therefore proves effective in the exact failure mode it was designed for and remains valuable. Routing HyDE by estimated query overlap, so that it applies only to low-overlap queries, is a logical refinement.

### Dense retrieval ablation, the cost of the constraint

An ablation evaluates the cost of the lexical-only constraint during retrieval. A pretrained sentence transformer is indexed in FAISS and scored against BM25 with the same questions, source chunks, and recall and MRR metrics, which allows a direct comparison of rows. The embedder exists only within the ablation (`ablation/`) and is not part of the deployed system. Two models represent two honest scenarios. The first is a small, general model with a short context window, and the second is a more powerful model that processes the full chunk.

| Configuration        | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| -------------------- | ----------------- | ----------------- | -------------- | -------------- |
| BM25                 | 0.384             | 0.651             | 0.251          | 0.477          |
| `all-MiniLM-L6-v2`   | 0.336             | 0.664             | 0.183          | 0.404          |
| `thenlper/gte-large` | 0.397             | 0.747             | 0.253          | 0.538          |

The small model does not outperform BM25 and falls short on three of the four metrics. This is a close miss because the model is general rather than scientific, and its 256-token window truncates a 250-word segment, resulting in less processed text than BM25. The more advanced model reads the entire chunk and surpasses BM25 at the paper level, increasing recall from 0.651 to 0.747 and MRR from 0.477 to 0.538, while remaining tied at the chunk level. Its primary strength is topical relevance, as a dense embedder more effectively identifies the correct paper, and it is already competitive for exact passage matching on lexical terms. Therefore, at the initial retrieval stage, this constraint lowers paper recall by about ten points compared to a strong embedder.

The single-stage gap explains the whole story, as the served pipeline also performs reranking. By fixing the reranker and changing only the first stage, it is evident that the embedder is added once reranking is included.

| Configuration               | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --------------------------- | ----------------- | ----------------- | -------------- | -------------- |
| lexical and rerank (served) | 0.555             | 0.808             | 0.418          | 0.720          |
| `gte-large` and rerank      | 0.520             | 0.836             | 0.396          | 0.735          |

The results indicate that all differences are within the 0.08 confidence interval, showing the two methods are statistically similar. The reranked dense pool slightly surpasses at the paper level with scores of 0.836 compared to 0.808, whereas the reranked lexical pool has a small edge at the chunk level with 0.555 versus 0.520. This echoes the earlier topically versus exact-match distinction seen in single-stage results, now after reranking. The ten-point recall lead of the embedder over BM25 at the paper level narrows to about three points once reranking is applied to both pools, and this difference is within noise levels. The reranker, which operates on a lexical pool, nearly restores the embedder’s advantage, explaining why the system remains lexical with minimal extra cost. Using a reranked dense first stage provides a slight gain in paper retrieval and is a reasonable choice if constraints are loosened, though it doesn't represent a significant victory.

A weak embedder cannot be improved solely through reranking. Re-ranking the MiniLM pool achieves only a 0.452 chunk recall and a 0.781 paper recall, both lower than the lexical pipeline's performance. Since the reranker can only reorder the pool it receives, it cannot retrieve correct results that were never presented initially. While it can enhance a strong candidate set, it cannot generate recall from a weak one.

### Tier 2 judged synthesis

$n=146$, judge `claude-opus-4-6`.

| Metric            | Score | Scored over |
| ----------------- | ----- | ----------- |
| Faithfulness      | 0.947 | 146         |
| Answer relevance  | 0.777 | 143         |
| Context precision | 0.608 | 143         |

Among the 146 evaluated answers, three relevance and three precision judgments were unparseable, which was distributed across four questions. These are excluded and tallied separately rather than scored as 0.0. The average relevance and precision scores therefore cover 143 responses, while faithfulness is averaged over all 146. Parse failures account for approximately 1.4% of the judge assessments.

Three main findings were discovered. Faithfulness remains consistently high, with nearly identical scores of 0.948 when the correct paper is retrieved and 0.946 when it is not, indicating that the system tends to ground or decline rather than hallucinate, even in retrieval failures. Answer relevance is slightly higher at 0.791 when the correct paper is fetched, versus 0.713, a modest difference. Context precision stands at 0.608, which is conservative, since many passages the judge marked irrelevant still originate from the correct paper.

## Reproducibility

Reproducibility is categorized into three clear tiers because the deterministic core of a RAG pipeline is reproducible, whereas the model itself is not. It is more important to specify which tier is being claimed than to claim the highest one.

- **Bitwise**. Identical bytes are produced at every stage. From a fixed corpus and configuration, the deterministic core is bitwise reproducible. This core covers chunking, cleaning, normalization, BM25 indexing, lexical retrieval, and the judge-free Tier 1 scoring. Rebuilding the index and obtaining the same BM25 retrieval results verifies it. The model-dependent parts, namely HyDE, reranking, synthesis, and the Tier 2 judge, are bitwise reproducible only when on-disk caches store every model response keyed by model, prompt, and inputs. The cache acts as a frozen snapshot, so reruns replay identically, and the cached outputs behave as committed inputs rather than freshly generated ones.

- **Statistical**. A rerun falls within the expected noise range. This is a weaker fallback for the model layer when caches are missing, because the API is not bitwise deterministic even at temperature 0, and models may change or be retired. Clearing the cache and rerunning shifted the model-dependent numbers by 0.01-0.05, within the stated interval, while the deterministic BM25 rows remained identical.

- **Provenance**. The conditions that produced a result are recorded and verifiable, even when a deprecated model prevents recomputation. For anything that relies on language models, this is the realistic standard, and it is more honest than treating the model as deterministic.

The project aims for bitwise reproducibility in the deterministic basis and employs provenance and cache-backed replay for model-dependent results. The only part that currently falls short is corpus construction, which reads from a live snapshot and citations with a date cutoff, as mentioned in the limitations below. Achieving bitwise reproducibility would involve fixing the snapshot revision, freezing citation enrichment, and setting a cutoff date.

## Limitations and future work

- **Synthetic, single-model-family evaluation.** The evaluation questions are generated by `claude-sonnet-4-6`, which also performs HyDE, reranking, and synthesis. The Tier 2 judge uses a different model, `claude-opus-4-6`, and the overlap between each question and its source chunk is measured so that the keyword edge passed to BM25 is transparent rather than hidden. Because the benchmark is synthetic and partly authored by the system's own model family, it has real validity limits, and human-curated questions would strengthen it considerably.

- **Faithfulness measures groundedness, not truth.** The 0.947 faithfulness score is the proportion of answer claims supported by retrieved passages, not a check of their scientific correctness. No domain expert validation was performed, so a confidently cited but incorrect passage can still score as faithful. The score reflects resistance to hallucination, not factual accuracy.

- **No dense retrieval, by design.** A strong embedder beats BM25 by about ten points in paper recall at the single retrieval stage. Once pools are reranked, this gap narrows to about three points and falls within the noise. The lexical pipeline is better at the chunk level, so adding a dense first stage before reranking would give only marginal paper-level gains and remains an optional extension.

- **HyDE is a targeted lever applied unconditionally.** HyDE markedly improves reranked chunk recall on roughly one third of questions, almost doubling it, while slightly hurting high-overlap questions. Applying it conditionally on estimated query overlap would keep the benefit and cut unnecessary cost.

- **Evaluation set size.** Both tiers now cover the full set of 146 questions, with confidence intervals of roughly 0.08. A larger set would improve stratification, since the high-overlap band holds only 14 questions.

- **Single-shot, not agentic.** The system retrieves and answers in one pass, without decomposing multi-step questions, calling tools, or self-critique. Each of these is a possible future enhancement.

- **Presentation versus measurement.** The current display does not account for per-paper diversity, so the same paper can appear several times. Simple deduplication would improve presentation without changing scoring.

- **Corpus is a point-in-time snapshot.** The corpus is captured live with a date-based recency cutoff, which makes its construction non-reproducible. A versioned corpus would lock the snapshot and cutoff.

- **Residual data noise.** Some acknowledgment text under malformed sections can survive cleaning, and about 1% of chunks are short fragments. A body-level acknowledgment filter and stricter chunking would address both.

## References

1. Es, Shahul, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2023. "RAGAS, Automated Evaluation of Retrieval Augmented Generation." arXiv preprint 2309.15217.

2. Gao, Luyu, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. "Precise Zero-Shot Dense Retrieval without Relevance Labels." In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1, Long Papers), 1762 to 1777. ACL Anthology.

3. Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks." In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 9459 to 9474.

4. Robertson, Stephen, and Hugo Zaragoza. 2009. "The Probabilistic Relevance Framework, BM25 and Beyond." Foundations and Trends in Information Retrieval 3 (4), 333 to 389.

5. Sun, Weiwei, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. "Is ChatGPT Good at Search? Investigating Large Language Models as Re Ranking Agents." In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 14918 to 14937. ACL Anthology.

6. Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. "Judging LLM as a Judge with MT Bench and Chatbot Arena." In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, 46595 to 46623.

