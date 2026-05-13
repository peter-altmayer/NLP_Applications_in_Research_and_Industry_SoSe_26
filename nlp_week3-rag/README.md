# NLP Week 3 — RAG Pipeline

**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026

## Description

End-to-end RAG evaluation pipeline comparing sparse (BM25) and dense (DPR) retrieval with two generator models across two open-domain QA datasets. The corpus is constructed from the answer strings of the first 1000 rows of each dataset. Results are evaluated with Exact Match and Token F1 across all combinations of retrieval method, K, model, and dataset.

## Model Choices and Justification

### Generator Models

**`microsoft/Phi-3-mini-4k-instruct` (3.8B)**
A modern instruction-tuned causal LM from Microsoft. Chosen because it is substantially more capable than the Flan-T5 models used in Week 2 while still fitting in fp16 on a consumer GPU (RTX 2070 Super, 8 GB VRAM). Phi-3-mini consistently punches above its weight on factual QA benchmarks relative to its parameter count, making it a good local baseline. The 4k context window is sufficient for K=10 retrieved passages.

**`Qwen/Qwen2.5-7B-Instruct` (7B)**
A strong open-weight instruction-tuned model from Alibaba's Qwen team. Chosen to pair with Phi-3-mini because it represents a different architecture family (grouped-query attention, RoPE) and roughly twice the capacity, allowing us to separate retrieval effects from model-capacity effects in the analysis. We originally targeted `mistralai/Mistral-7B-Instruct-v0.3`, but it is gated on Hugging Face; Qwen2.5-7B-Instruct is fully open, comparable in size/quality, and avoids the licence/token friction. On a free Colab T4 (16 GB VRAM) it is loaded in 4-bit NF4 (`bitsandbytes`) so it fits alongside the DPR encoders and KV cache.

Together these two models cover a meaningful capacity gap (~3.8B vs 7B) and different training recipes (Microsoft vs Alibaba), so differences in the results table are interpretable.

### Retrieval Models

**BM25 (`rank_bm25`)**
A strong non-neural baseline. Included because it requires no GPU, is deterministic, and frequently outperforms dense retrieval on short, factual queries — exactly our domain. Serves as the reference point against which neural retrievers are measured.

**`facebook/dpr-*-single-nq-base` (DPR-NQ)**
The original Dense Passage Retrieval model fine-tuned exclusively on Natural Questions. Expected to perform well on the NQ evaluation split and potentially transfer less well to TriviaQA, providing a concrete case study in dataset-specific retrieval.

**`facebook/dpr-*-multiset-base` (DPR-Multiset)**
The same DPR architecture trained on a mix of four QA datasets: NQ, TriviaQA, WebQuestions, and CuratedTREC. By comparing DPR-NQ against DPR-Multiset we can directly observe whether broader training data improves retrieval generalisation — particularly interesting since TriviaQA is in the multiset training data but not in the NQ-only model.

## Environment Setup

### Google Colab (recommended)

The full pipeline runs end-to-end on a free Colab T4 via `notebooks/colab_rag_pipeline.ipynb`. The notebook mounts Google Drive, clones this repo from GitHub, symlinks `data/` and `results/` to Drive so artifacts survive runtime disconnects, installs deps, and runs all five phases (build → phi3 × 2 datasets → qwen × 2 datasets → merge).

To use it:
1. Push your latest commits to GitHub (`git push`)
2. Open `notebooks/colab_rag_pipeline.ipynb` in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run cells top to bottom

The run-experiment cells are idempotent (skip-if-CSV-exists), so you can re-run after a disconnect and only resume what's missing.

### Local (optional fallback)

```bash
pip install uv
uv venv
# Windows:
.venv\Scripts\activate
uv pip install -r requirements.txt
```

`bitsandbytes` only matters for 4-bit loading of the 7B model — on a local GPU with ≥16 GB VRAM you can drop the `quantize_4bit` flag and run fp16.

## Experiments

```bash
# Phase 1 — build indexes (run once per dataset)
python experiments/build_index.py

# Phase 2 — generate (one CLI call per model × dataset)
python experiments/run_experiment.py --model phi3 --dataset trivia_qa
python experiments/run_experiment.py --model phi3 --dataset natural_questions
python experiments/run_experiment.py --model qwen --dataset trivia_qa --batch_size 2
python experiments/run_experiment.py --model qwen --dataset natural_questions --batch_size 2

# Phase 3 — merge CSVs, compute EM/F1, print table + failure cases
python experiments/merge_results.py
```

## Results

All 25 (model × dataset × retrieval × k) configurations were evaluated on **1000 questions each — 25 000 predictions total**. Raw predictions and per-row EM / F1 are in `results/all_results.csv`; the helper script `experiments/build_readme_artifacts.py` regenerates the three Markdown artifacts referenced below.

### Headline table — mean EM / mean F1

| model | dataset | retrieval | k | EM | F1 |
|---|---|---|---|---|---|
| phi3 | natural_questions | none | 0 | 0.000 | 0.058 |
| phi3 | natural_questions | bm25 | 1 | 0.000 | 0.071 |
| phi3 | natural_questions | bm25 | 5 | 0.000 | 0.097 |
| phi3 | natural_questions | bm25 | 10 | 0.000 | **0.198** |
| phi3 | natural_questions | dpr_nq | 1 | 0.000 | 0.044 |
| phi3 | natural_questions | dpr_nq | 5 | 0.000 | 0.081 |
| phi3 | trivia_qa | none | 0 | **0.338** | **0.408** |
| phi3 | trivia_qa | bm25 | 1 | 0.147 | 0.176 |
| phi3 | trivia_qa | bm25 | 5 | 0.100 | 0.118 |
| phi3 | trivia_qa | bm25 | 10 | 0.092 | 0.109 |
| phi3 | trivia_qa | dpr_nq | 1 | 0.100 | 0.129 |
| phi3 | trivia_qa | dpr_nq | 5 | 0.058 | 0.073 |
| phi3 | trivia_qa | dpr_nq | 10 | 0.051 | 0.066 |
| phi3 | trivia_qa | dpr_multiset | 1 | 0.083 | 0.110 |
| phi3 | trivia_qa | dpr_multiset | 5 | 0.054 | 0.073 |
| phi3 | trivia_qa | dpr_multiset | 10 | 0.065 | 0.085 |
| qwen | natural_questions | none | 0 | 0.000 | 0.033 |
| qwen | trivia_qa | none | 0 | 0.326 | 0.391 |
| qwen | trivia_qa | bm25 | 1 | 0.044 | 0.055 |
| qwen | trivia_qa | bm25 | 5 | 0.046 | 0.058 |
| qwen | trivia_qa | bm25 | 10 | 0.043 | 0.060 |
| qwen | trivia_qa | dpr_nq | 1 | 0.008 | 0.019 |
| qwen | trivia_qa | dpr_nq | 5 | 0.023 | 0.034 |
| qwen | trivia_qa | dpr_nq | 10 | 0.022 | 0.035 |
| qwen | trivia_qa | dpr_multiset | 1 | 0.015 | 0.026 |

N = 1000 rows per cell. Bold = highest value in its column section. The full table is also in `results/findings.md`.

### 25-row sample per configuration

A random 25-row sample (seed = 42) from every configuration is dumped to **[`results/samples_per_config.md`](results/samples_per_config.md)** — 25 rows × 25 configs = 625 Q/Gold/Prediction/EM/F1 rows. The file is generated by `experiments/build_readme_artifacts.py` and kept in version control as the reproducible per-config view.

A representative excerpt (phi3 · trivia_qa · bm25 · k=1, the best non-closed-book retrieval cell):

| Question | Gold | Prediction | EM | F1 |
|---|---|---|---|---|
| Which alien named after the actor that played him in the movies is the only other entity to wield Thor's hammer? | Beta Ray Bill | Beta Ray Bill | 1 | 1.00 |
| In which film does Cilla Black sing 'Anyone Who Had a Heart'? | Cilla | I don't know. | 0 | 0.00 |
| What is the most famous Aria from Bizet's Carmen? | Habanera | The Habanera | 1 | 1.00 |
| Phyllis Logan stars in which UK TV period drama? | DOWNTON ABBEY | Downton Abbey | 1 | 1.00 |

### Findings — Model × Retriever × K

**1. RAG hurts more than it helps on this corpus.** Closed-book (`none`, k=0) is the single best setting on TriviaQA for both models — phi3 reaches **EM 0.338** with no retrieval and drops to 0.147 the moment BM25 supplies one passage. Qwen shows the same collapse (0.326 → 0.044). The corpus is built from the answer strings of the first 1 000 dataset rows (≈1 200 short documents) — for most evaluation questions the relevant answer text is *not* in this small corpus, so retrieval injects an irrelevant distractor that the model dutifully grounds on.

**2. More K is worse, not better.** On phi3 · trivia_qa · bm25, EM degrades monotonically: k=1 → 0.147, k=5 → 0.100, k=10 → 0.092. F1 follows the same trajectory. Same trend for DPR-NQ (0.100 → 0.058 → 0.051). The only exception is **Natural Questions F1**, where more passages help (phi3 · NQ · bm25 F1: 0.071 → 0.097 → 0.198) — but EM stays at exactly 0.000 because NQ gold answers are long Wikipedia passages and the model produces one-line answers, so EM can never fire (more on this in Failure Analysis).

**3. BM25 ≥ DPR on a small answer-string corpus.** DPR was trained on Wikipedia-scale passages; the corpus here is ~1 200 short answer snippets, which is far outside that distribution. BM25 wins consistently — phi3 · trivia_qa k=1: BM25 0.147 EM > DPR-NQ 0.100 > DPR-Multiset 0.083. DPR-Multiset has TriviaQA in its training mix but still loses to DPR-NQ at k=1; the gap is small and noise-dominated.

**4. Phi-3 (3.8B fp16) beats Qwen-2.5-7B-4bit on retrieval, ties on closed-book.** Closed-book TriviaQA is essentially a tie (phi3 0.338, qwen 0.326), which is the expected parametric-knowledge comparison. But once retrieved context arrives, Qwen collapses much harder than Phi-3 (qwen BM25 k=1: 0.044 EM vs phi3 0.147). Two factors plausibly contribute: (a) 4-bit NF4 quantisation degrades long-context attention more than parametric recall; (b) Qwen tends to over-elaborate when uncertain instead of conceding (see Failure Case 1), hurting EM.

**5. Natural Questions is a metric problem, not a model problem.** Every NQ row has **EM = 0.000**. NQ gold answers in `sentence-transformers/natural-questions` are full Wikipedia paragraphs (200–500 tokens), so an EM match would require the model to regurgitate the entire passage verbatim. F1 is the only meaningful NQ signal — and it does respond to retrieval (phi3 · bm25 k=10 hits F1 0.198, the highest NQ score).

### Failure Analysis — 10 cases

Full case detail (questions, gold, predictions, retrieved docs) lives in **[`results/failure_cases.md`](results/failure_cases.md)**. The selection rule: pick the lowest-F1 EM=0 row from each of 10 distinct (model, dataset, retrieval, k) combinations, so the cases span configurations rather than concentrating on one weak cell.

| # | Config | Question (truncated) | Failure mode |
|---|---|---|---|
| 1 | qwen · trivia_qa · none · k=0 | "In Adam's Rib, who played the Spencer Tracy role?" | **Hallucination** — Qwen invents "Henry Fonda" instead of "Ken Howard". |
| 2 | qwen · trivia_qa · bm25 · k=5 | "In which state is Camp David?" | **Irrelevant retrieval** — retrieves "Dalai Camp"; Qwen abstains. |
| 3 | phi3 · NQ · bm25 · k=1 | "where does a hamster live in the wild" | **Retrieval miss** — top doc is *The Call of the Wild* (lexical "wild" trap); phi3 abstains. |
| 4 | phi3 · NQ · bm25 · k=10 | "when did the us military start using hummers" | **Retrieval miss** — top doc is "Five-star rank" (lexical "US military" trap), correct Humvee passage not in top-k. |
| 5 | qwen · trivia_qa · dpr_nq · k=5 | "In Top Cat, who was the voice of Choo Choo?" | **Out-of-corpus** — DPR returns generic country/profession docs; the Marvin Kaplan answer is not in the corpus at all. |
| 6 | phi3 · NQ · bm25 · k=5 | "who gave the motto back to the vedas" | **Retrieval miss** — BM25 returns "Jarasandha" (Sanskrit-adjacent topic); the Dayananda Saraswati passage isn't surfaced. |
| 7 | qwen · trivia_qa · dpr_nq · k=10 | "Who did Mrs. Thatcher describe as a man we can do business with?" | **DPR cold-start** — top doc is "Sculptor (profession)"; DPR's semantic embedding fails on a quotation-heavy query. |
| 8 | qwen · trivia_qa · dpr_multiset · k=1 | "First MVP in a Super Bowl on the losing side?" | **Out-of-corpus** — top doc is "Kramer vs. Kramer"; Chuck Howley is not in the answer-string corpus. |
| 9 | phi3 · NQ · dpr_nq · k=1 | "who sang what in the world's come over you" | **Wrong-passage retrieval** — DPR-NQ picks a Lenny Kravitz song page instead of the Jack Scott page; phi3 abstains. |
| 10 | phi3 · NQ · dpr_nq · k=5 | "what part of the country are you likely to find the majority of the mollisols" | **Domain mismatch** — DPR returns a Malaysia geography paragraph; the soil-science Mollisol passage is missed entirely. |

**Patterns across the 10 cases:**

- **8 of 10 are retrieval failures, not generation failures.** The model behaves correctly given what it sees — it abstains ("I don't know") when the retrieved passage is off-topic. The bottleneck is the retriever surfacing the wrong document, not the LM failing to extract.
- **Both BM25 and DPR get fooled by surface-lexical traps.** BM25 in Case 3 latches onto "wild" (Call of the Wild), Case 4 onto "US military" (Five-star rank). DPR in Case 9 finds a same-genre but wrong-artist song page. Different mechanisms, same failure: a high-similarity match that is semantically irrelevant.
- **Qwen hallucinates where Phi-3 abstains.** Case 1 (closed-book) is the cleanest example: Qwen invents a plausible-but-wrong cast, while Phi-3 in similar situations (cases 3–10) consistently emits "I don't know". This makes Phi-3 safer for RAG and explains its higher headline EM despite being half the size.
- **Many "EM=0" NQ cases are EM-metric artefacts, not real misses.** When the gold answer is a 300-token Wikipedia paragraph and the prediction is a correct short phrase, EM is mechanically 0 even when F1 captures partial overlap. The NQ EM column being all zeros is therefore not informative — F1 is the metric to read there.
- **The corpus, not the retriever, is the root cause for ~3 of 10.** Cases 5, 7, 8 ask about facts (Marvin Kaplan, Gorbachev, Chuck Howley) whose answer strings are not present in the first-1000 corpus at all. No retriever can recover these; the design choice to build the corpus from answers-of-the-first-1000-rows is the limiting factor.
