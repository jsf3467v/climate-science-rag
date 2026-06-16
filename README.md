[![CI](https://github.com/jsf3467v/climate-science-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/climate-science-rag/actions/workflows/ci.yml)

---
title: Climate arXiv RAG
sdk: gradio
sdk_version: "5.9.1"
app_file: src/app.py
pinned: false
---

# Climate Science arXiv RAG

Grounded question answering with citations over approximately 3,000 climate science
arXiv papers. A question is expanded into a hypothetical answer passage, which is then
matched against a lexical index, reranked by a language model, and answered solely using
the retrieved passages. The design includes inline citations and an explicit refusal when
the corpus does not address the question.

The system runs as an interactive chat interface built with Gradio. A user asks a climate
science question in plain language and gets back an answer with inline citations, or an
explicit refusal when the corpus does not cover the question. The chat is the front door to
the whole pipeline below.

The main constraint is that only the LLM is pretrained, so the served system has no dense
embedders or cross encoders. Retrieval relies solely on lexical methods such as BM25, while
the language model is used for query expansion, reranking, synthesis, and evaluation. An
ablation (below) measures what this costs against a strong pretrained embedder, and finds
that the reranker recovers nearly all of the embedder advantage, so the constraint buys a
lightweight, CPU only, easily inspectable retrieval system at little measured cost.

See `DEPLOYME.md` for build, deploy, and local run instructions, including how to launch the
chat interface.

## Architecture

The pipeline is a sequence of single purpose stages. The expensive work, the downloads and the 
model calls, is cached or checkpointed, so a stop and rerun resumes instead of repeating it.

1. **Corpus** (`corpus.py`). Enumerates climate papers from a static arXiv metadata
   snapshot (no live API rate limits, license carrying), enriches with Semantic Scholar
   citations and fields, and filters by keyword and a citation and recency gate down to
   3,000 papers.

2. **Full text and chunking** (`fulltext.py`). Fetches LaTeX source and recovers section
   structure, falling back to the PDF when source is missing. Sections are split into
   250 word windows with 40 word overlap, each chunk carrying provenance. Approximately
   119k chunks survive cleaning.

3. **Cleaning and normalization** (`clean.py`, `normalize.py`). Drops reference list and
   data table chunks with a prose density gate first (genuine prose is kept regardless of
   its citations), then citation structure (DOIs, "et al", parenthetical years) and a
   numeric ratio test on the remainder. Normalization scrubs LaTeX to text placeholder
   artifacts, leaving real content untouched.

4. **Index and retrieval** (`indexer.py`, `retrieve.py`). One BM25 index over a shared
   tokenizer, with no embeddings.

5. **Query expansion** (`hyde.py`). HyDE has the model write a short, plausible answer
   passage. Retrieval runs on that passage (prepended with the query) so lexical matching
   reaches domain terms a paraphrased question never names.

6. **Reranking** (`rerank.py`). The model reorders the BM25 over HyDE candidate pool
   listwise, in one call.

7. **Synthesis** (`synthesize.py`). The model answers from the reranked passages only,
   cites them by number, and declines when the answer is absent.

8. **Evaluation** (`evaluation/`). Tier 1 is judge free, deterministic recall@k and MRR
   against each question source chunk. Tier 2 is an independent judge model scoring
   faithfulness, answer relevance, and context precision. The reranked path that deploys
   is the exact path the eval scores.

## Dependancies

* **Python**, CPU only. No GPU anywhere in the served stack.

* **rank_bm25** for lexical retrieval, **scikit-learn** for the shared tokenizer stopword
  set, and **joblib** to persist the index.

* **pylatexenc** and **PyMuPDF** for LaTeX and PDF full text extraction.

* **Hugging Face `datasets`** for the arXiv metadata snapshot and the **Semantic Scholar**
  API for citation metadata.

* **Anthropic Claude**. `claude-sonnet-4-6` for HyDE, reranking, synthesis, and writing the
  eval questions, and `claude-opus-4-6` as the independent Tier 2 judge (enforced to differ
  from the generation model so it never grades its own work).

* **Gradio** for the interactive chat app.

* **sentence-transformers** and **FAISS** appear only in the ablation (`ablation/`) to
  measure the cost of the lexical only constraint, and neither enters the served system.

## Results

Tier 1 retrieval, n=146 questions, cutoff k=10.

| Configuration | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --- | --- | --- | --- | --- |
| BM25 | 0.384 | 0.651 | 0.251 | 0.477 |
| with HyDE | 0.384 | 0.685 | 0.225 | 0.448 |
| with HyDE and rerank | 0.555 | 0.808 | 0.418 | 0.720 |
| rerank, no HyDE | 0.527 | 0.822 | 0.396 | 0.712 |

The served path is BM25 over HyDE followed by the listwise reranker, the row marked with
HyDE and rerank, and it is the exact path the evaluation scores. The final row reranks a
plain BM25 pool with HyDE removed, included to test whether query expansion earns its keep.

Chunk level recall is a strict single reference metric. Each question is scored against the
one chunk it was written from out of roughly 119k, so retrieving an equally good neighbor
counts as a miss. The paper level numbers, which credit any chunk from the correct paper,
are the fairer read of retrieval quality, and the paper level MRR shows that when the right
paper is found it usually sits at rank one or two.

The gain is concentrated in the reranker. Moving from BM25 to the reranked path raises chunk
recall from 0.384 to 0.555 and paper MRR from 0.477 to 0.720, a step well clear of the
roughly 0.08 confidence interval at this sample size. The reranker is the component carrying
the result.

### Does HyDE earn its keep

In aggregate, HyDE and the no HyDE rerank path are tied, 0.555 against 0.527 chunk recall and
0.808 against 0.822 paper recall, both inside the noise. An earlier reading of these
aggregate numbers concluded HyDE could be dropped. Stratifying the questions by the lexical
overlap between the question and its source chunk shows that conclusion is wrong, because the
aggregate tie is the average of two opposite effects. Low overlap questions (overlap below
0.30, the paraphrased third of the set, 54 of 146) are the failure mode HyDE was built for,
and high overlap questions (14 of 146) already share the corpus vocabulary and need no help.

Recall by overlap band follows, as chunk recall then paper recall, for the reranked paths.

| Configuration | low overlap (n=54) | mid overlap (n=78) | high overlap (n=14) |
| --- | --- | --- | --- |
| BM25 | 0.093 / 0.500 | 0.487 / 0.705 | 0.929 / 0.929 |
| with HyDE and rerank | 0.352 / 0.815 | 0.654 / 0.795 | 0.786 / 0.857 |
| rerank, no HyDE | 0.185 / 0.685 | 0.692 / 0.885 | 0.929 / 1.000 |

On the paraphrased band, HyDE nearly doubles reranked chunk recall, from 0.185 to 0.352, and improves 
paper recall by thirteen points, from 0.685 to 0.815. It performs slightly negatively on high-overlap
questions, where the expansion only adds plausible competitors to an already well-matched
query. The overall neutrality is the average of this specific gain and the small loss. 
Thus, HyDE proves effective precisely in the failure mode it was designed for, and it remains valuable. 
Routing HyDE based on estimated query overlap—applying it only to low-overlap queries—is a logical refinement.


### Dense retrieval ablation, the cost of the constraint

An ablation evaluates the cost of the lexical-only constraint during retrieval. A pretrained
sentence transformer is indexed in FAISS and scored against BM25 using the same questions, source chunks, 
and recall and MRR metrics, allowing for direct comparison of rows. The
embedder exists solely within the ablation (`ablation/`) and is not part of the deployed system. 
Two models represent two honest scenarios: a small, general model with a short context window and a more 
powerful model that processes the full chunk.


| Configuration | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --- | --- | --- | --- | --- |
| BM25 | 0.384 | 0.651 | 0.251 | 0.477 |
| `all-MiniLM-L6-v2` | 0.336 | 0.664 | 0.183 | 0.404 |
| `thenlper/gte-large` | 0.397 | 0.747 | 0.253 | 0.538 |

The small model does not outperform BM25, missing three out of four metrics. However, this is a narrow miss, 
as the model is more general rather than scientific, and its 256-token window truncates a 250-word chunk, 
meaning it processes less text than BM25. The more advanced model, which reads the entire chunk, outperforms BM25 
at the paper level, increasing recall from 0.651 to 0.747 and MRR from 0.477 to 0.538, while tying at the chunk level. 
The main advantage is topical relevance: a dense embedder is better at identifying the correct paper, and for exact 
passage matching based on lexical terms, it is already competitive. Therefore, at the initial retrieval stage, this 
constraint reduces paper recall by about ten points compared to a strong embedder.


The single-stage gap isn't the full story because the served pipeline also reranks. By keeping the reranker fixed and 
only changing the first stage, we can see what the embedder contributes once
reranking is involved.


| Configuration | recall@10 (chunk) | recall@10 (paper) | MRR@10 (chunk) | MRR@10 (paper) |
| --- | --- | --- | --- | --- |
| lexical and rerank (served) | 0.555 | 0.808 | 0.418 | 0.720 |
| `gte-large` and rerank | 0.520 | 0.836 | 0.396 | 0.735 |

The result shows a split, with all gaps falling within the 0.08 confidence band, indicating the two are
statistically comparable. The reranked dense pool performs slightly better at the paper level, with scores 
of 0.836 versus 0.808, while the lexical pool reranked shows a slight edge at the chunk level, scoring 0.555 
compared to 0.520. This reflects the same topical versus exact match distinction seen in the single-stage results, 
now after reranking. The ten-point paper recall advantage of the embedder over BM25 diminishes to about three 
points once both pools are reranked, with this difference falling within the noise. The reranker, which operates 
on a lexical pool, nearly recovers the embedder's strong advantage, explaining why the system remains lexical with minimal measured
cost. A reranked dense first stage is marginally better for paper retrieval and remains a viable option if constraints can be relaxed, 
but it does not constitute a decisive victory.


A weak embedder cannot be improved using this method. Reranking the MiniLM pool achieves only 0.452
chunk recall and 0.781 paper recall, which are below the performance of the lexical pipeline. 
This is because the reranker can only reorder the pool it receives and cannot recover the correct 
results that were not initially surfaced. While the reranker can enhance a good candidate set, it cannot generate recall from a weak set.


### Tier 2 judged synthesis

n=146, judge `claude-opus-4-6`.

| Metric | Score | Scored over |
| --- | --- | --- |
| Faithfulness | 0.947 | 146 |
| Answer relevance | 0.777 | 143 |
| Context precision | 0.608 | 143 |

Out of the 146 answers evaluated, three yielded unparseable relevance judgments and three yielded 
unparseable precision judgments across four different questions. These responses are excluded and 
tallied separately rather than assigned a score of 0.0. Consequently, the average relevance and 
precision scores are calculated over 143 responses, while faithfulness is averaged over all 146. 
The rate of parse failures is approximately 0.7% of judge assessments.

Three findings stand out in the synthesis data. Faithfulness remains consistently high and
nearly identical across retrieval outcomes, 0.948 when the correct paper is retrieved and 0.946
when it is not, indicating that the system tends to ground or decline rather than hallucinate,
even in cases of retrieval failure. The relevance of answers is slightly higher when the correct
paper is retrieved, with scores of 0.791 compared to 0.713, reflecting a modest but notable effect
rather than a strong correlation. Finally, the context precision score is 0.608, which serves as a
conservative estimate since many passages marked as irrelevant by the judge still originate from the correct
paper.


## Reproducibility

Reproducibility is outlined in three explicit tiers, as the deterministic core of
a RAG pipeline is reproducible, unlike a model. Clarifying which tier is
claimed is more important than asserting the highest one.


* **Bitwise**. Identical bytes are produced at every stage. The deterministic core—comprising 
chunking, cleaning, normalization, BM25 indexing, lexical retrieval, and the judge-free Tier 1 
scoring—is reproducible in a bitwise manner from a fixed corpus and configuration. This is 
verified by rebuilding the index and obtaining the exact same BM25 retrieval results. The 
model-dependent parts—such as HyDE, reranking, synthesis, and the Tier 2 judge—are only 
bitwise reproducible when using on-disk caches that store every model response, keyed by model, 
prompt, and inputs. The cache acts as a frozen snapshot. Using it ensures that reruns replay identically, 
making the question set and cached outputs equivalent to committed inputs rather than newly generated outputs.


* **Statistical**. A rerun falls within the expected noise range. This is a less reliable fallback for 
the model layer when caches are missing, as the API is not bitwise deterministic even at temperature 0, 
and models may change or be retired. Clearing the cache and rerunning shifted the model-dependent numbers 
by 0.01 to 0.05, staying within the stated interval, while the deterministic BM25 rows remained identical.


* **Provenance**. The specific conditions that led to a result are recorded and verifiable, even if a deprecated 
model prevents re-computation. For anything reliant on LLMs, this represents the realistic gold standard, and it's 
more honest than pretending the model is deterministic.


This project aims for bitwise reproducibility of the deterministic basis and uses
provenance and cache backed replay for model dependent results. The only component not yet
meeting this standard is corpus construction, which reads a live snapshot and live
citations with a date based cutoff (see the limitation below). Fixing the snapshot
revision, freezing the citation enrichment, and setting the cutoff date would bring it to
bitwise reproducibility.

## Limitations and future work

* **Synthetic, Single Model Family Evaluation:** The current evaluation questions are generated by `claude-sonnet-4-6`, which also performs HyDE, reranking, and synthesis. The Tier 2 judge uses a different model (`claude-opus-4-6`), and the overlap between questions and source chunks is measured so that the keyword edge passed to BM25 is transparent rather than hidden. However, since the benchmark is synthetic and partly authored by the system's own model family, it has certain validity limitations. Human-curated questions would improve the evaluation's robustness significantly.


* **Faithfulness Measures Groundedness, Not Truth:** The 0.947 faithfulness score indicates the proportion of answer claims supported by retrieved passages, rather than verifying their scientific correctness. No domain expert validation was performed, so a confidently cited but incorrect passage could still be scored as faithful. This score reflects hallucination resistance, not factual accuracy.


* **No Dense Retrieval, by Design:** A strong embedder outperforms BM25 by about ten points in paper recall at the single retrieval stage. When pools are reranked, this gap narrows to around three points and is within the noise range. The lexical pipeline performs better at the chunk level, so implementing a dense first-stage reranking would give marginal gains at the paper level and is a viable extension.


* **HyDE is a targeted lever applied unconditionally.** HyDE significantly improves reranked chunk recall on roughly one third of questions, almost doubling it, but has a mild negative effect on high overlap questions. Applying HyDE conditionally based on estimated query overlap could preserve the benefits while reducing unnecessary costs.


* **Evaluation Set:** Consisting of 146 questions with intervals of roughly 0.08, both tiers now cover the entire set. A larger dataset would improve stratification, especially since the high-overlap band contains only 14 questions.


* **Single-Shot, Not Agentic:** The system retrieves and answers in one go, without decomposing multi-step questions, using tools, or self-critique. These are potential enhancements for future development.


* **Presentation vs. Measurement:** The current display does not reflect per-paper diversity, so the same paper might appear multiple times. Simple deduplication would improve presentation without affecting scoring.


* **Corpus is a Point-in-Time Snapshot:** The corpus captures a live snapshot with a recency cutoff based on date, making reruns non-reproducible. Using a versioned corpus would lock the snapshot and cutoff for consistency.


* **Residual Data Noise:** Some acknowledgment texts under malformed sections may persist after cleaning, and about 1% of chunks are short fragments. Implementing a body-level acknowledgment filter and more restrictive chunking would address these issues.



## References

1. Es, Shahul, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2023. "RAGAS, Automated Evaluation of Retrieval Augmented Generation." arXiv preprint 2309.15217.


2. Gao, Luyu, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. "Precise Zero-Shot Dense Retrieval without Relevance Labels." In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1, Long Papers), 1762 to 1777. ACL Anthology.


3. Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks." In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 9459 to 9474.


4. Robertson, Stephen, and Hugo Zaragoza. 2009. "The Probabilistic Relevance Framework, BM25 and Beyond." Foundations and Trends in Information Retrieval 3 (4), 333 to 389.


5. Sun, Weiwei, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. "Is ChatGPT Good at Search? Investigating Large Language Models as Re Ranking Agents." In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 14918 to 14937. ACL Anthology.


6. Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. "Judging LLM as a Judge with MT Bench and Chatbot Arena." In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, 46595 to 46623.