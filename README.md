---
title: Climate arXiv RAG
sdk: gradio
sdk_version: "6.18.0"
app_file: src/app.py
pinned: false
---

[![CI](https://github.com/jsf3467v/climate-science-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/climate-science-rag/actions/workflows/ci.yml)

# Climate Science arXiv RAG

This project answers climate science questions over a corpus of roughly 3,000 arXiv papers and grounds every answer in the passages it retrieves. A question is initially expanded into a hypothetical answer, which is then matched against a lexical index. It is reranked using a language model and answered solely based on the retrieved text, including inline citations and an explicit refusal if the corpus lacks the requested information.

The entire process is accessible through an interactive Gradio chat, allowing users to ask questions and receive either a cited response or a refusal.

The main constraint is that only the language model is pretrained, resulting in a system without dense embedders or cross-encoders. The retrieval relies solely on lexical methods such as BM25, while the language model handles query expansion, reranking, synthesis, and evaluation. An ablation study shown below compares this setup to a strong pretrained embedder and demonstrates that the reranker nearly recovers the embedder's advantages. As a result, the lexical-only approach provides a lightweight, CPU-only, inspectable system with minimal additional cost.

Build, deploy, and local run instructions, including how to launch the chat, are in `DEPLOYME.md`.

## Architecture

The pipeline runs as a sequence of single-purpose stages, with the more expensive downloads and model calls cached or checkpointed so that, in case of an interruption, the model resumes rather than restarts.

The `corpus.py` script builds a collection of climate papers from a static snapshot of arXiv metadata, avoiding live API rate limits and ensuring clear licensing. It enriches each paper with Semantic Scholar citations and fields, then filters down to approximately 3,000 papers using keywords, citation metrics, and recency criteria. The `fulltext.py` script retrieves the LaTeX source for these papers and reconstructs their section structure, resorting to PDFs when source files are unavailable. It then divides each section into 250-word segments with a 40-word overlap, preserving provenance for each chunk. After cleaning, around 119,000 chunks remain.

Cleaning and normalization (`clean.py`, `normalize.py`) remove reference lists and data-table chunks, applying a prose-density gate first so that genuine prose is kept whatever its citation count, then filtering on citation structure such as DOIs, `et al`, and parenthetical years together with a numeric-ratio test on what remains. Normalization scrubs the LaTeX-to-text placeholder artifacts while leaving real content untouched.

The retrieval core (`indexer.py`, `retrieve.py`) is a single BM25 index over a shared tokenizer, with no embeddings anywhere. Query expansion (`hyde.py`) uses HyDE, where the model writes a short, plausible answer passage and retrieval runs on that passage prepended to the query, so lexical matching can reach domain terms a paraphrased question never names. Reranking (`rerank.py`) then has the model reorder the BM25-over-HyDE candidate pool listwise in a single call, and synthesis (`synthesize.py`) answers from those reranked passages alone, cites them by number, and declines when the answer is not present.

The evaluation lives in evaluation/ and separates the quality of what is retrieved from the quality of what is written. A deterministic, judge-free process manages retrieval by measuring recall@k and MRR based on the original source chunk for each question. Separately, an independent judge model assesses the synthesis by scoring faithfulness, relevance of the answer, and context accuracy. Since the path being scored is the same path that deploys, the numbers describe the served system rather than serving as a stand-in for it.

## Dependencies

The served stack includes only Python and CPU, with no GPU used at any stage. Lexical retrieval is performed with `rank_bm25`, while `scikit-learn` provides the stopword tokenizer set. The index is saved using `joblib`. Full-text extraction uses `pylatexenc` and `PyMuPDF`. The arXiv metadata snapshot is fetched via Hugging Face `datasets`, and citation data is obtained through the Semantic Scholar API. The language model is Anthropic Claude: `claude-sonnet-4-6` handles HyDE, reranking, synthesis, and writing evaluation questions, while `claude-opus-4-6` acts as the independent Tier 2 judge, which must differ from the generation model to avoid grading its own work. Gradio offers the chat interface. The only embedding dependencies, `sentence-transformers` and FAISS, are used only in the ablation and are not part of the served system.

## Results

The first table reports Tier 1 retrieval over $n = 146$ questions at a cutoff of $k = 10$.

| Configuration | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --- | --- | --- | --- | --- |
| BM25 | 0.384 | 0.651 | 0.251 | 0.477 |
| with HyDE | 0.384 | 0.685 | 0.225 | 0.448 |
| with HyDE and rerank | 0.555 | 0.808 | 0.418 | 0.720 |
| rerank, no HyDE | 0.527 | 0.822 | 0.396 | 0.712 |

The served path uses BM25 over HyDE, followed by a listwise reranker. It includes the row marked with HyDE and rerank, which corresponds exactly to the evaluation scores. The last row reranks a plain BM25 pool without HyDE, testing if query expansion provides a real benefit.

Chunk-level recall is a strict single-reference metric, as each question is scored against the one chunk it originated from out of about 119,000. Even a near-perfect neighbor is considered a miss. The paper-level metrics, which count any chunk from the correct paper, give a fairer assessment of retrieval quality. The paper-level MRR indicates that once the correct paper is retrieved, it is usually ranked first or second.

The main improvements come from the reranker. Switching from BM25 to the reranked path raises chunk recall from 0.384 to 0.555 and paper MRR from 0.477 to 0.720. These improvements are significantly above the approximate 0.08 confidence interval at this sample size, indicating that the reranker is the key component driving the results.

### Does HyDE earn its keep

Overall, HyDE and the no-HyDE rerank path are comparable, with $0.555$ versus $0.527$ in chunk recall and $0.808$ versus $0.822$ in paper recall, both within the margin of error. An earlier interpretation of these aggregate figures suggested that HyDE could be eliminated, but dividing the questions based on the lexical overlap between a question and its source chunk reveals this is incorrect. The average tie in the aggregate results masks two opposing effects. The questions with low overlap—those below $0.30$, accounting for about a third of the set with 54 of 146 questions—are the primary challenge HyDE was designed to address. In contrast, the 14 questions with high overlap already share vocabulary with the corpus and do not require additional help.

The next table reports recall by overlap band for the reranked paths, written as chunk recall then paper recall.

| Configuration | low overlap (n=54) | mid overlap (n=78) | high overlap (n=14) |
| --- | --- | --- | --- |
| BM25 | 0.093 / 0.500 | 0.487 / 0.705 | 0.929 / 0.929 |
| with HyDE and rerank | 0.352 / 0.815 | 0.654 / 0.795 | 0.786 / 0.857 |
| rerank, no HyDE | 0.185 / 0.685 | 0.692 / 0.885 | 0.929 / 1.000 |

On the paraphrased band, HyDE nearly doubles reranked chunk recall, from $0.185$ to $0.352$, and improves paper recall by thirteen points, from $0.685$ to $0.815$. It becomes slightly negative on high-overlap questions, where expansion only adds plausible competitors to an already well-matched query. The overall neutrality reflects the average of this targeted gain and a small loss. HyDE performs as intended in its failure mode and remains valuable, with the potential refinement of routing it by estimated query overlap to activate only on low-overlap queries.

### Dense retrieval ablation, the cost of the constraint

This ablation assesses the retrieval cost of using only lexical constraints. It uses a pretrained sentence transformer indexed in FAISS, scored against BM25 on identical questions, source chunks, and metrics, allowing for direct comparison of rows. The embedder resides solely within the ablation (`ablation/`) and not in the deployed system. Two models represent two realistic scenarios: a small, general model with a short context window and a more capable model that processes the entire chunk.

| Configuration | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --- | --- | --- | --- | --- |
| BM25 | 0.384 | 0.651 | 0.251 | 0.477 |
| `all-MiniLM-L6-v2` | 0.336 | 0.664 | 0.183 | 0.404 |
| `thenlper/gte-large` | 0.397 | 0.747 | 0.253 | 0.538 |

The small model fails to outperform BM25 in three out of four metrics, although the difference is minor. This is partly because the model is general instead of scientific, and its 256-token window truncates a 250-word chunk, meaning it actually processes less text than BM25. The more capable model, which reads the entire chunk, outperforms BM25 at the paper level, increasing recall from $0.651$ to $0.747$ and MRR from $0.477$ to $0.538$, while remaining tied at the chunk level. Its advantage is topical, as a dense embedder more accurately identifies the correct paper and is already effective at matching specific lexical passages, which leads to roughly ten points lower recall in paper retrieval at this initial stage.

| Configuration | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --- | --- | --- | --- | --- |
| lexical and rerank (served) | 0.555 | 0.808 | 0.418 | 0.720 |
| `gte-large` and rerank | 0.520 | 0.836 | 0.396 | 0.735 |

The results are split, with all gaps within the $0.08$ range, making the two pools effectively comparable. The reranked dense pool outperforms at the paper level, scoring $0.836$ compared to $0.808$, while at the chunk level, the reranked lexical pool leads with $0.555$ versus $0.520$. This reflects the previous topical-versus-exact distinction, now after reranking. The ten-point advantage of the embedder over raw BM25 in paper recall diminishes to about three points after reranking both pools, which is within the margin of error. Therefore, the reranker applied to a lexical pool nearly matches the embedder's performance, keeping the system lexical with minimal additional cost. Using a reranked dense first stage offers marginal improvement for paper retrieval and is a reasonable option if the constraints are relaxed; however, it is not a definitive improvement.

A weak embedder cannot be significantly improved by reranking alone. Reranking the MiniLM pool attains only $0.452$ chunk recall and $0.781$ paper recall, both lower than the lexical pipeline. This is because the reranker can only reorder the pool it receives and cannot identify correct results the first stage failed to produce. While it can refine a good candidate set, it cannot generate recall from a poor one.

### Tier 2 judged synthesis

Tier 2 covers all $n = 146$ answers, judged by `claude-opus-4-6`.

| Metric | Score | Scored over |
| --- | --- | --- |
| Faithfulness | 0.947 | 146 |
| Answer relevance | 0.777 | 143 |
| Context precision | 0.608 | 143 |

Out of 146 responses, three yielded unparseable relevance judgments and three produced unparseable precision judgments across four questions. These are excluded and counted separately rather than scored as zero, resulting in averages of relevance and precision over 143 assessments, while faithfulness is averaged over all 146. The failure rate for parsing is approximately 0.7 percent of judge assessments.

The synthesis data reveal a consistent pattern. Faithfulness remains high and nearly unchanged whether the correct paper is retrieved, with scores of 0.948 versus 0.946. This indicates the system bases its answers on grounding or chooses to decline rather than fabricating when retrieval fails. Answer relevance is slightly higher when the correct paper is retrieved, at 0.791 compared to 0.713, showing a modest but notable effect. Context precision is at 0.608, though this figure likely understates the system's accuracy, as many passages marked irrelevant by the judge still originate from the correct paper.

## Reproducibility

Reproducibility is categorized into three tiers because the deterministic core of a RAG pipeline can be exactly reproduced, whereas a model cannot. Clarifying which tier a result belongs to is more important than asserting the highest level of reproducibility.

The deterministic core is bitwise reproducible, producing identical bytes from a fixed corpus and configuration across chunking, cleaning, normalization, BM25 indexing, lexical retrieval, and the judge-free Tier 1 scoring. This is confirmed by rebuilding the index and recovering the same BM25 results to the digit. The model layer is only statistically reproducible, typically falling within a small noise range, rather than producing identical outputs. This is because the API is non-deterministic even at temperature zero, and models can change or be retired. A rerun may shift model-dependent figures by $0.01$ to $0.05$, while deterministic data remains consistent. As a result, the model layer serves as provenance, with committed result files like `results.jsonl` and `summary.json` for Tier 1, along with two Tier 2 files, documenting the prompts and configurations used. These files provide the reliable standard for any process relying on the language model.

The project therefore claims bitwise reproducibility for the deterministic core and provenance for the model layer, with the committed result files serving as the record. The one piece below that bar is corpus construction, which reads a live snapshot and live citations against a date-based cutoff. So, pinning the snapshot revision, freezing the citation enrichment, and fixing the cutoff date would achieve bitwise reproducibility.

## Limitations and future work

The evaluation has certain limitations. Its questions are generated by `claude-sonnet-4-6`, the same model used for HyDE, reranking, and synthesis. Although the Tier 2 judge is different and question-to-chunk overlap is openly measured, the benchmark remains synthetic and partly authored by the system's own model family. Human-curated questions would enhance its robustness. The faithfulness score, at $0.947$, reflects the proportion of answer claims supported by retrieved passages, not their scientific accuracy. Without domain-expert review, a confidently cited but incorrect passage could still be considered faithful. Additionally, the small set of 146 questions reduces stratification, especially since only fourteen fall into the high-overlap category.

Some limits are intentionally set. The system doesn't use dense retrieval, which results in about ten points of paper recall loss initially, but decreases to around three points after reranking both pools. Therefore, implementing a dense first stage could be an extension but not necessarily better. HyDE is applied to all queries, mainly benefiting paraphrased third questions but slightly impairing high-overlap questions; routing based on estimated overlap could maintain benefits while reducing costs. Retrieval and answering are performed in one step, without breaking down multi-step questions, calling tools, or self-critique, which are potential future improvements.

Other issues are minor. The corpus is a snapshot at a specific date, so rerunnings aren't reproducible unless a versioned corpus is used. The current presentation doesn't ensure per-paper diversity, so a paper might appear multiple times until deduplication is implemented. Some residual noise remains, such as acknowledgment text under malformed sections and about one percent of chunks as short fragments; these can be eliminated with a body-level acknowledgment filter and more rigorous chunking.

## References

1. Es, Shahul, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2023. "RAGAS, Automated Evaluation of Retrieval Augmented Generation." arXiv preprint 2309.15217.

2. Gao, Luyu, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. "Precise Zero-Shot Dense Retrieval without Relevance Labels." In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1, Long Papers), 1762 to 1777. ACL Anthology.

3. Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks." In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 9459 to 9474.

4. Robertson, Stephen, and Hugo Zaragoza. 2009. "The Probabilistic Relevance Framework, BM25 and Beyond." Foundations and Trends in Information Retrieval 3 (4), 333 to 389.

5. Sun, Weiwei, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. "Is ChatGPT Good at Search? Investigating Large Language Models as Re Ranking Agents." In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 14918 to 14937. ACL Anthology.

6. Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. "Judging LLM as a Judge with MT Bench and Chatbot Arena." In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, 46595 to 46623.
