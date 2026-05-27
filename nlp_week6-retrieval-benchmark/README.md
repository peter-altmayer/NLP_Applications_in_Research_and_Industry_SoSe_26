# Retrieval Benchmark

8-method information-retrieval benchmark across MS MARCO and SciFact.

| # | Method | Implementation |
|---|--------|----------------|
| M1 | BM25 | `rank_bm25.BM25Okapi` |
| M2 | TF-IDF | `sklearn.TfidfVectorizer` + cosine sim |
| M3 | Dense (general) | `sentence-transformers/all-MiniLM-L6-v2` |
| M4 | Dense (domain) | `sentence-transformers/msmarco-distilbert-base-v3` |
| M5 | Hybrid RRF | M1 + M3 via Reciprocal Rank Fusion (k=60) |
| M6 | ColBERT | `colbert-ir/colbertv2.0` via `ragatouille` |
| M7 | Cross-encoder re-rank | Top-100 from M3 → `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| M8 | HyDE | JGU KI-Chat API → M3 dense retrieval |

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install CPU-only PyTorch first (avoids spurious GPU detection)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Running the Benchmark

**Always run scripts from the `nlp_week6-retrieval-benchmark/` directory** (required for ColBERT index path resolution).

```bash
# Run M1–M7 on both datasets
# First run: ~60–120 min (model downloads + indexing)
# Cached re-run: ~2–5 min
python scripts/run_benchmark.py

# Run HyDE (M8) — ~10 queries per dataset — requires JGU API key
export JGU_API_KEY=<your_key>       # Linux/macOS
set JGU_API_KEY=<your_key>          # Windows cmd
$env:JGU_API_KEY="<your_key>"       # PowerShell
python scripts/run_hyde_subset.py
```

Results are written to `results/`:
- `results/{dataset}_results.csv` — full metric table
- `results/{dataset}_results.md` — same table, best per column **bolded**
- `results/per_query/` — per-query rankings (rank, score, is_relevant) per method

## Running Tests

```bash
pytest                    # 28 fast unit tests (no internet required)
pytest -m integration     # integration tests (require HF model downloads)
```

## Hardware Notes

Tested on AMD Ryzen 7 8845HS, CPU-only. Expected times (first run, no cache):

| Step | Time |
|------|------|
| MS MARCO loading | ~5 min |
| Embedding 10k passages (MiniLM) | ~10 min |
| Embedding 5k passages (msmarco-distilbert) | ~5 min |
| BM25 / TF-IDF indexing | <1 min |
| ColBERT indexing (~10k passages) | **30–60 min** |
| Cross-encoder (100 candidates × 300 queries) | ~15 min |

**ColBERT escape hatch:** If indexing exceeds ~90 min, run it on a free Colab T4 GPU, download the index, and place it at `cache/colbert/{dataset}__index/`. Continue locally from there.

## Cache

All heavy computation is cached under `cache/` (gitignored).

| Cache file | Contents |
|------------|----------|
| `cache/embeddings/*.npy` | Dense document embeddings |
| `cache/bm25/*.pkl` | BM25 index |
| `cache/tfidf/*.pkl` | TF-IDF vectorizer + matrix |
| `cache/colbert/{dataset}__index/` | ColBERT index directory |
| `cache/hyde/{dataset}__hypotheticals.json` | HyDE generated passages |

Second full run completes in **2–5 minutes**.

## API Key

`JGU_API_KEY` must be set as an environment variable to run HyDE (M8).  
Add a `.env` file (gitignored) or export in your shell. **Never commit the key.**

The HyDE script fails immediately with a clear error if `JGU_API_KEY` is unset.

## Project Structure

```
nlp_week6-retrieval-benchmark/
├── config.py                    # paths, seeds, model IDs, k-values
├── requirements.txt
├── README.md
├── pytest.ini
├── src/
│   ├── cache.py                 # load_or_compute + doc_ids_hash
│   ├── data/
│   │   ├── load_msmarco.py      # MS MARCO loader (300 queries + distractors)
│   │   └── load_scifact.py      # SciFact loader (300 queries, 5k docs, graded qrels)
│   ├── methods/
│   │   ├── base.py              # Retriever ABC
│   │   ├── bm25.py              # M1
│   │   ├── tfidf.py             # M2
│   │   ├── dense.py             # M3 + M4 (parameterized)
│   │   ├── hybrid_rrf.py        # M5
│   │   ├── colbert.py           # M6
│   │   ├── cross_encoder.py     # M7
│   │   └── hyde.py              # M8
│   ├── eval/
│   │   ├── metrics.py           # evaluate() → P@k, R@k, MRR@10, MAP@100, NDCG@10
│   │   └── runner.py            # run_retriever, measure_latency, save_per_query, bold_best_md
│   └── analysis/
│       ├── disagreement.py      # gold_rank, find_max_divergence_queries, find_bm25_beats_dense
│       └── qualitative.py       # format_query_comparison, dump_disagreement_examples
├── scripts/
│   ├── run_benchmark.py         # one-shot entry point (M1–M7)
│   └── run_hyde_subset.py       # HyDE-only, ~10 queries/dataset
├── tests/
│   ├── conftest.py              # shared fixtures (tiny_corpus, cache_dir, …)
│   ├── test_cache.py
│   ├── test_metrics.py          # known-answer unit tests
│   ├── test_methods.py
│   ├── test_analysis.py
│   └── test_data.py             # integration tests (marked, not run by default)
├── notebooks/
│   └── analysis.ipynb           # 7 analysis questions with concrete examples
├── results/                     # generated at runtime (not gitignored)
└── cache/                       # gitignored
```

## Results

*(populated after running the benchmark)*

### MS MARCO — validation split, 300 queries, ~5–10k passage corpus

| Method | P@1 | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency_ms |
|--------|-----|-----|------|------|------|-------|--------|---------|---------|------------|
| BM25 | | | | | | | | | | |
| TF-IDF | | | | | | | | | | |
| Dense-MiniLM | | | | | | | | | | |
| Dense-msmarco | | | | | | | | | | |
| Hybrid-RRF | | | | | | | | | | |
| CrossEncoder | | | | | | | | | | |
| ColBERT | | | | | | | | | | |
| HyDE (10q) | | | | | | | | | | |

### SciFact — test split, 300 queries, 5k-doc corpus

| Method | P@1 | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency_ms |
|--------|-----|-----|------|------|------|-------|--------|---------|---------|------------|
| BM25 | | | | | | | | | | |
| TF-IDF | | | | | | | | | | |
| Dense-MiniLM | | | | | | | | | | |
| Dense-msmarco | | | | | | | | | | |
| Hybrid-RRF | | | | | | | | | | |
| CrossEncoder | | | | | | | | | | |
| ColBERT | | | | | | | | | | |
| HyDE (10q) | | | | | | | | | | |

## Analysis

See `notebooks/analysis.ipynb` for full analysis with concrete query examples:

1. **Overall winner** — which method wins per metric; cases where rankings flip
2. **BM25 vs Dense** — 3–5 queries where BM25 beats MiniLM, with hypotheses (rare terms, exact match)
3. **Top-vs-worst disagreement** — 5 queries with maximum gold-rank spread across all methods
4. **HyDE win/fail** — hypothetical quality inspection for ~5 helped and ~5 hurt queries
5. **M4 vs M3** — domain-specific benefit by dataset (expected: M4 helps on MS MARCO)
6. **Hybrid M5** — does RRF beat both BM25 and Dense on every metric?
7. **Re-rank cost-benefit** — M3 Stage-1 R@100 vs CrossEncoder gains vs latency overhead
