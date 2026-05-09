# RAG Pipeline — Design Spec
**Date:** 2026-05-09
**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026

---

## Overview

A modular RAG evaluation pipeline that compares sparse vs. dense retrieval and two generator architectures across two QA datasets. The corpus is the answer strings from the first 1000 rows of each dataset. Execution is split into a local phase (Phi-3-mini) and a Colab phase (Mistral-7B), with results merged offline.

---

## Datasets

| Dataset | HF path | Split | Rows used |
|---|---|---|---|
| TriviaQA | `mandarjoshi/trivia_qa` | `rc.nocontext` | first 1000 |
| Natural Questions | `sentence-transformers/natural-questions` | `train` | first 1000 |

**Corpus construction:** for each dataset, extract the answer string(s) from each row and flatten into a list of documents. These answer strings are the only documents in the retrieval index — no external knowledge base.

---

## Models

### Generator models

| Key | HF model ID | Environment | Precision |
|---|---|---|---|
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` | Local RTX 2070 Super | fp16 |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | Google Colab (T4/A100) | fp16 |

### Retrieval models

| Key | Type | HF model IDs |
|---|---|---|
| `bm25` | Sparse (BM25) | `rank_bm25` library — no neural model |
| `dpr_nq` | Dense | `facebook/dpr-question_encoder-single-nq-base` + `facebook/dpr-ctx_encoder-single-nq-base` |
| `dpr_multiset` | Dense | `facebook/dpr-question_encoder-multiset-base` + `facebook/dpr-ctx_encoder-multiset-base` |

---

## Experiment Grid

| Dimension | Values |
|---|---|
| Dataset | trivia_qa, natural_questions |
| Retrieval method | bm25, dpr_nq, dpr_multiset |
| Generator | phi3, mistral |
| K | 0, 1, 5, 10 |

**K=0** means no context is provided — the generator answers from parametric memory only. Retrieval method is irrelevant for K=0; results are shared across retrieval columns.

Total unique generator calls: 4 (K=0 combos) × 1000 + 36 (K>0 combos) × 1000 = **40,000**

---

## Architecture

### Two-phase execution

**Phase 1 — Index** (`experiments/build_index.py`, run once)
- Load first 1000 rows of each dataset
- Extract corpus (answer strings)
- Save corpus to `data/processed/{dataset}_corpus.json`
- Build and save BM25 index to `data/processed/{dataset}_bm25.pkl`
- Build and save DPR embeddings to `data/processed/{dataset}_dpr_nq.npy` and `{dataset}_dpr_multiset.npy`

**Phase 2 — Experiment** (`experiments/run_experiment.py --model <key> --dataset <key>`)
- Load saved indexes (no re-embedding)
- Iterate over retrieval methods × K values
- For K=0: pass empty context list to generator
- For K>0: retrieve top-K docs, format as prompt context
- Write one CSV per (model, dataset, retrieval_method) to `results/`
- Skip existing CSVs (crash-safe resumption)

**Phase 3 — Merge** (`experiments/merge_results.py`)
- Concatenate all CSVs from `results/`
- Compute EM and token F1 per row
- Print pivot table: rows = retrieval × K, columns = model × dataset
- Save `results/all_results.csv`
- Print 10 worst failure cases with labels

---

## Module Design (`src/`)

### `src/dataset.py`
- `load_dataset(name: str, split: str, n: int = 1000) -> list[dict]`
  Returns records `{"question": str, "answers": list[str]}`. Normalises answer format differences between TriviaQA and NQ.
- `extract_corpus(records: list[dict]) -> list[str]`
  Flattens all answer strings into a single list (the retrieval corpus).

### `src/retriever.py`
- `build_bm25(corpus: list[str]) -> BM25Okapi`
- `save_bm25(index, path: Path)`
- `load_bm25(path: Path) -> BM25Okapi`
- `build_dpr_embeddings(corpus: list[str], ctx_encoder_name: str, device: str) -> np.ndarray`
- `save_embeddings(embeddings: np.ndarray, path: Path)`
- `load_embeddings(path: Path) -> np.ndarray`
- `retrieve_bm25(index, query: str, k: int) -> list[str]`
- `retrieve_dpr(embeddings: np.ndarray, corpus: list[str], q_encoder, tokenizer, query: str, k: int) -> list[str]`

All `retrieve_*` functions return `list[str]` of length k — uniform interface for the generator.

### `src/model.py`
- `load_generator(model_name: str) -> tuple[model, tokenizer]`
  Auto-detects CUDA, loads in fp16 if available.
- `build_prompt(question: str, context_docs: list[str]) -> str`
  System instruction: "Answer the question using only the provided context. If the context does not contain the answer, say 'I don't know'." Formats retrieved docs as a numbered list above the question.
- `generate_answers(model, tokenizer, records: list[dict], batch_size: int = 8) -> list[str]`
  Each record has `question` and `context_docs`. Returns one answer string per record.

### `src/evaluate.py`
- `normalize(text: str) -> str`
  Lowercase, strip articles and punctuation, collapse whitespace (standard SQuAD normalisation).
- `exact_match(pred: str, gold_list: list[str]) -> bool`
- `token_f1(pred: str, gold_list: list[str]) -> float`
- `build_results_table(df: pd.DataFrame) -> pd.DataFrame`
  Pivot: index = (retrieval_method, K), columns = (model, dataset), values = mean EM and F1.

---

## File Layout

```
nlp_week3-rag/
  data/
    raw/                          # unused (gitkeep)
    processed/
      trivia_qa_corpus.json
      trivia_qa_bm25.pkl
      trivia_qa_dpr_nq.npy
      trivia_qa_dpr_multiset.npy
      natural_questions_corpus.json
      natural_questions_bm25.pkl
      natural_questions_dpr_nq.npy
      natural_questions_dpr_multiset.npy
  experiments/
    build_index.py
    run_experiment.py
    merge_results.py
    run_baseline.py               # existing exploration script (unchanged)
  results/
    phi3_trivia_qa_bm25.csv
    phi3_trivia_qa_dpr_nq.csv
    phi3_trivia_qa_dpr_multiset.csv
    phi3_natural_questions_bm25.csv
    phi3_natural_questions_dpr_nq.csv
    phi3_natural_questions_dpr_multiset.csv
    mistral_*.csv                 # produced on Colab, copied here
    all_results.csv
  src/
    __init__.py
    dataset.py
    retriever.py
    model.py
    evaluate.py
  notebooks/
    exploration.ipynb
  requirements.txt
  README.md
```

---

## Result CSV Schema

Each CSV written by `run_experiment.py` has these columns:

| Column | Type | Notes |
|---|---|---|
| `dataset` | str | `trivia_qa` or `natural_questions` |
| `model` | str | `phi3` or `mistral` |
| `retrieval_method` | str | `none`, `bm25`, `dpr_nq`, `dpr_multiset` |
| `k` | int | 0, 1, 5, or 10 |
| `question` | str | Original question text |
| `gold_answers` | str | JSON-encoded list of gold answer strings |
| `retrieved_docs` | str | JSON-encoded list of retrieved doc strings (empty list for K=0) |
| `prediction` | str | Raw model output |

`merge_results.py` adds `em` (bool→int) and `f1` (float) columns during the merge pass.

---

## Evaluation Output

Results table structure (mean EM / mean F1):

```
                        phi3                    mistral
                  trivia_qa  natural_q     trivia_qa  natural_q
retrieval  K
none       0        xx / xx    xx / xx       xx / xx    xx / xx
bm25       1        xx / xx    xx / xx       xx / xx    xx / xx
           5        xx / xx    xx / xx       xx / xx    xx / xx
           10       xx / xx    xx / xx       xx / xx    xx / xx
dpr_nq     1        ...
...
```

---

## Failure Analysis

`merge_results.py` prints 10 failure cases: rows where EM=0 and F1 is lowest, sampled across retrieval methods and models. Each case shows: question, gold answers, retrieved docs, generated answer, EM, F1, and a label for likely failure source (retriever / generator / both).

---

## Dependencies

```
datasets
transformers
torch
rank_bm25
numpy
pandas
```

Pinned after setup: `uv pip freeze > requirements.txt`

---

## Colab Workflow

1. Upload `nlp_week3-rag/` or `git clone` the repo
2. `pip install uv && uv pip install -r requirements.txt`
3. `python experiments/build_index.py` (if not already done locally and uploaded)
4. `python experiments/run_experiment.py --model mistral --dataset trivia_qa`
5. `python experiments/run_experiment.py --model mistral --dataset natural_questions`
6. Download all `results/mistral_*.csv` back to local machine
7. Run `python experiments/merge_results.py` locally
