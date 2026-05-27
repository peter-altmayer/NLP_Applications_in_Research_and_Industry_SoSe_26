# Retrieval Benchmark Design — 2026-05-27

## Overview

An 8-method information-retrieval benchmark comparing BM25, TF-IDF, dense, hybrid, ColBERT, cross-encoder re-ranking, and HyDE across MS MARCO and SciFact. Deliverable: a reproducible Python project with cached artifacts, full metric tables, and written analysis answering 7 specific questions.

---

## Project Location

```
NLP Applications in Research and Industry/
└── nlp_week4-retrieval-benchmark/    ← new subdirectory in course repo
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── config.py
    ├── src/
    │   ├── data/
    │   ├── methods/
    │   ├── eval/
    │   ├── analysis/                 ← disagreement.py, qualitative.py
    │   └── cache.py
    ├── scripts/
    ├── results/
    ├── cache/                        ← gitignored
    ├── notebooks/
    └── docs/
```

---

## Datasets

### MS MARCO (`ms_marco`, v1.1, `validation` split)
- Sample exactly **300 queries**, `seed=42`.
- Build corpus from all passages in those queries' `passages.passage_text` lists.
- Add distractor passages (from other validation queries not in the 300) to reach ~5–10k total.
- `doc_id` format: `f"msmarco_{query_idx}_{passage_idx}"` — stable across runs.
- `qrels`: `{query_id: {doc_id: 1}}` where `is_selected == 1`.
- Primary metric: MRR@10 (binary relevance, 1 relevant passage per query).

### SciFact (`BeIR/scifact` + `BeIR/scifact-qrels`)
- Full 300 test queries + full 5k-doc corpus — no sampling.
- Corpus: `BeIR/scifact` `corpus` split; each doc = title + " " + text.
- qrels: load from `BeIR/scifact-qrels`; `score` field = graded relevance (0, 1, or 2).
- Primary metric: NDCG@10 (graded relevance).

Both loaders return: `{"queries": dict[str, str], "corpus": dict[str, str], "qrels": dict[str, dict[str, int]]}`.

---

## Cache Architecture (`src/cache.py`)

```
cache/
  embeddings/
    {model_id_sanitized}__{dataset}__{doc_ids_hash}.npy    # doc embeddings
    {model_id_sanitized}__{dataset}__queries__{hash}.npy   # query embeddings
  bm25/
    {dataset}__{doc_ids_hash}.pkl
  tfidf/
    {dataset}__{doc_ids_hash}.pkl
  colbert/
    {dataset}__index/                                       # ColBERT index dir
  hyde/
    {dataset}__hypotheticals.json                           # query → generated text
```

- `doc_ids_hash = hashlib.sha1(",".join(sorted(doc_ids)).encode()).hexdigest()[:12]`
- `load_or_compute(path, fn)`: if path exists, load and return; else call `fn()`, save, return.
- Supports `.npy` (numpy), `.pkl` (pickle), `.json`, and directory existence checks (ColBERT).
- Second run of the full benchmark must complete in 2–5 minutes.

---

## The 8 Retrieval Methods

Build order respects dependencies:

| # | Name | Implementation | Key notes |
|---|------|----------------|-----------|
| M1 | BM25 | `rank_bm25.BM25Okapi` | Lowercase + whitespace tokenization |
| M2 | TF-IDF | `sklearn.TfidfVectorizer` + cosine sim | Standard settings |
| M3 | Dense (general) | `sentence-transformers/all-MiniLM-L6-v2` | Baseline dense |
| M4 | Dense (domain) | `sentence-transformers/msmarco-distilbert-base-v3` | MS MARCO fine-tuned |
| M5 | Hybrid RRF | M1 + M3 via Reciprocal Rank Fusion, k=60 | Reuses cached M1 + M3 results |
| M6 | ColBERT | `ragatouille` library | Fallback: `pylate`; document if Colab needed |
| M7 | Cross-encoder re-rank | Top-100 from M3 → `cross-encoder/ms-marco-MiniLM-L-6-v2` | Truncate doc, never query; 512-token cap |
| M8 | HyDE | JGU KI-Chat API → M3 dense retrieval | ~10 queries/dataset only; separate script |

**HyDE config:**
- Base URL: `https://ki-chat.uni-mainz.de/api` (OpenAI-compatible)
- Model: `GPT OSS 120B`, `reasoning_effort="low"`, `max_tokens=150`
- API key: `os.environ["JGU_API_KEY"]` — never hardcoded or logged
- Rate limit: 1 req/sec sustained; serial calls + `time.sleep(1.1)` between requests
- Fail fast with clear error if `JGU_API_KEY` is unset

---

## Retriever Interface (`src/methods/base.py`)

```python
from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # Returns: [(doc_id, score), ...] sorted descending by score
        ...
```

All methods implement this ABC. `runner.py` calls `.index()` once, then `.search()` per query.

---

## Evaluation Metrics (`src/eval/metrics.py`)

Use `pytrec_eval` for all metrics. Implement once, reuse for every method × dataset combination.

| Metric | Details |
|--------|---------|
| P@k | k ∈ {1, 5, 10} |
| R@k | k ∈ {10, 50, 100} |
| MRR@10 | Primary for MS MARCO |
| MAP@100 | |
| NDCG@10 | Primary for SciFact; respects graded labels |
| Latency | Avg wall-clock ms/query via `time.perf_counter`; 3-query warmup; averaged ≥20 queries; single-query serial |

Unit test: a tiny 3-doc, 2-query known example validates all metrics before use.

---

## Runner (`src/eval/runner.py`)

- Iterates: for each dataset × for each method → call `.index()`, measure `.search()` latency, collect rankings, compute all metrics.
- HyDE skipped in `run_benchmark.py`; run separately via `run_hyde_subset.py`.
- Output: one `pandas.DataFrame` per dataset, 8 rows × ≥9 cols. Best value per column bolded in `.md` export.

---

## Output Files

```
results/
  msmarco_results.csv
  msmarco_results.md          # best-per-column bolded
  scifact_results.csv
  scifact_results.md
  per_query/                  # {method}_{dataset}_rankings.csv per method
  hyde_hypotheticals.json     # {query_id: {query, prompt, hypothetical, model, reasoning_effort}}
```

---

## Analysis (notebooks/analysis.ipynb + README)

7 questions, each with concrete query examples:

1. Overall winner — does ranking flip between metrics?
2. BM25 vs dense — 3–5 queries where BM25 wins; hypothesize why.
3. Top-vs-worst disagreement — 5 queries with maximum rank divergence.
4. HyDE win/fail — ~5 queries where it helped, ~5 where it hurt; comment on hypothetical quality.
5. M4 vs M3 — domain-specific benefit by dataset.
6. Hybrid (M5) — does RRF beat both components on every metric?
7. Re-rank cost-benefit (M7) — ms overhead vs NDCG/MRR gain; M3's stage-1 R@100 reported separately.

---

## Engineering Constraints

- **Determinism:** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` at top of every script.
- **PyTorch:** CPU-only install (`--index-url https://download.pytorch.org/whl/cpu`).
- **Batch sizes:** 32–64 for encoding; `num_workers=0`.
- **Tokenizer warnings:** suppressed with `transformers.logging.set_verbosity_error()`.
- **ColBERT escape hatch:** if ragatouille/pylate indexing exceeds ~90 min, index on Colab T4, download index, continue locally. Document in README if used.
- **`.gitignore`:** `cache/`, `results/raw/`, `.env`, `*.pkl`, `*.npy`, `*.pt`, `__pycache__/`.

---

## Implementation Order (Approach A — Infrastructure-first)

1. Project scaffold + `config.py` + `.gitignore` + `requirements.txt`
2. Data loaders (`load_msmarco.py`, `load_scifact.py`) + smoke tests
3. Cache layer (`cache.py`) + unit test for idempotency
4. Eval harness (`metrics.py` with known-answer unit test, `runner.py` skeleton)
5. M1 (BM25) → full end-to-end pipeline working
6. M2 (TF-IDF)
7. M3 (Dense/MiniLM) — establishes embedding cache pattern
8. M4 (Dense/msmarco)
9. M5 (Hybrid RRF) — reuses M1 + M3
10. M7 (Cross-encoder) — reuses M3 first-stage
11. M6 (ColBERT via ragatouille)
12. M8 (HyDE, separate script `run_hyde_subset.py`)
13. Results tables + analysis notebook
14. README with exact run commands + analysis answers
