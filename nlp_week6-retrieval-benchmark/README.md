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

M6 (ColBERT) skipped — requires a C++ compiler (MSVC) to JIT-compile `segmented_maxsim_cpp` on Windows; see README for Colab escape hatch. M8 (HyDE) run on 10-query subset pending API key.

### MS MARCO — validation split, 289 queries, 7,471 docs

| Method | P@1 | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency_ms |
|--------|-----|-----|------|------|------|-------|--------|---------|---------|------------|
| BM25 | 0.2491 | 0.1474 | 0.0848 | 0.8028 | 0.9308 | 0.9464 | 0.4266 | 0.4313 | 0.5166 | 14.9 |
| TF-IDF | 0.2284 | 0.1391 | 0.0896 | 0.8426 | 0.9533 | 0.9602 | 0.4123 | 0.4175 | 0.5149 | **5.1** |
| Dense-MiniLM | 0.3702 | 0.1813 | 0.1048 | 0.9827 | 0.9965 | **1.0000** | 0.5659 | 0.5614 | 0.6648 | 13.5 |
| Dense-msmarco | 0.4360 | 0.1958 | **0.1066** | **0.9965** | **1.0000** | **1.0000** | 0.6353 | 0.6319 | 0.7232 | 27.5 |
| Hybrid-RRF | 0.3149 | 0.1682 | 0.1010 | 0.9481 | 0.9948 | 0.9965 | 0.5168 | 0.5155 | 0.6199 | 31.8 |
| CrossEncoder | **0.4844** | **0.1965** | 0.1055 | 0.9879 | **1.0000** | **1.0000** | **0.6594** | **0.6579** | **0.7396** | 2663.6 |
| ColBERT | — | — | — | — | — | — | — | — | — | — |
| HyDE (10q) | — | — | — | — | — | — | — | — | — | — |

### SciFact — test split, 300 queries, 5k-doc corpus

| Method | P@1 | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency_ms |
|--------|-----|-----|------|------|------|-------|--------|---------|---------|------------|
| BM25 | 0.4367 | 0.1347 | 0.0763 | 0.6862 | 0.7768 | 0.7929 | 0.5242 | 0.5202 | 0.5597 | 20.6 |
| TF-IDF | 0.4533 | 0.1413 | 0.0797 | 0.7135 | 0.8601 | 0.8836 | 0.5437 | 0.5380 | 0.5780 | **10.2** |
| Dense-MiniLM | 0.5033 | **0.1640** | 0.0883 | 0.7833 | 0.8920 | 0.9250 | 0.6047 | 0.6031 | 0.6451 | 17.0 |
| Dense-msmarco | 0.4233 | 0.1333 | 0.0757 | 0.6723 | 0.8052 | 0.8460 | 0.5065 | 0.4974 | 0.5379 | 36.4 |
| Hybrid-RRF | 0.5267 | 0.1507 | 0.0850 | 0.7588 | **0.9007** | **0.9393** | 0.6059 | 0.6028 | 0.6378 | 38.9 |
| CrossEncoder | **0.5767** | **0.1640** | **0.0910** | **0.8072** | 0.9003 | 0.9250 | **0.6570** | **0.6485** | **0.6866** | 6683.1 |
| ColBERT | — | — | — | — | — | — | — | — | — | — |
| HyDE (10q) | — | — | — | — | — | — | — | — | — | — |

## Analysis

See `notebooks/analysis.ipynb` for full analysis with concrete query examples:

1. **Overall winner** — which method wins per metric; cases where rankings flip

   CrossEncoder (M7) wins on most ranking metrics across both datasets, NDCG@10, MRR@10, MAP@100, and P@1 are all highest. But the headline number hides two things worth noting. 

   First, Dense-msmarco (M4) actually beats CrossEncoder on R@10 for MS MARCO (0.997 vs 0.988), this makes sense because the cross-encoder can only re-rank the top-100 candidates from Stage 1, so if a relevant doc lands outside that window it's gone. 
   
   Second, the ranking completely flips depending on the dataset for precision-focused metrics: on SciFact, Hybrid-RRF beats Dense-MiniLM on P@1, which doesn't happen on MS MARCO at all. CrossEncoder is the clear "best" if you have offline time to spare, but at 2664ms and 6683ms average latency it's not a realistic production retriever.

2. **BM25 vs Dense** — 3–5 queries where BM25 beats MiniLM, with hypotheses (rare terms, exact match)

   BM25 beats Dense-MiniLM on about 26% of MS MARCO queries and 17% of SciFact queries. The winning cases show a consistent pattern containing rare or highly specific terms where exact lexical matching beats semantic generalization. 
   
   A concrete example from MS MARCO: "how long do paid tax liens stay on your credit report" (BM25 rank 1, Dense rank 7). 
   
   On SciFact the effect is even cleaner: "Activation of PPM1D suppresses p53 function" has BM25 at rank 1 and Dense-MiniLM at rank 15. My hypothesis is that dense models generalize by mapping terms into semantic neighborhoods, but for rare gene names, legal jargon, or company names there is no meaningful neighborhood. 

3. **Pick 5 queries where top and worst method disagree most — why do they differ?**

   The maximum-disagreement queries show exactly where each method's assumptions break down. 
   
   On MS MARCO all five top cases share the same pattern where BM25 is the worst performer (spread 60–99) because the queries use high-frequency, ambiguous vocabulary. "Costs of utilities per month" has a spread of 99 with BM25 at rank 100 and Dense-msmarco at rank 1, the words "costs", "utilities", "per month" individually match thousands of documents, leaving BM25 no discriminating signal. Another interesting case is "how to get rid of staff infection" where TF-IDF is the worst (rank 75): this is almost certainly a typo for "staph infection" and TF-IDF's exact matching is punished harder than BM25 because it weights the rare misspelled token very highly, while Dense-msmarco still retrieves the right passage at rank 4 because embeddings absorb typos through context. 
   
   On SciFact the reversal is notable, for "NF2 (Merlin) causes phosphorylation and subsequent cytoplasmic sequestration of YAP in Drosophila", TF-IDF is best (rank 6) while Dense-msmarco is worst (rank 85), highly specific gene names create a vocabulary mismatch that hurts the domain-adapted dense model most.

4. **Where does HyDE win and fail? Look at the actual hypotheticals generated.**

   On MS MARCO it wins clearly on "how much should a dog drink" (HyDE rank 2 vs Dense rank 7) — the generated hypothetical reads "A healthy dog should typically drink about 1 ounce (30 ml) of water per pound of body weight each day," which is essentially the answer document itself, making it a perfect dense retrieval target. It also wins on "will be amended to read shortages in area" (HyDE rank 1 vs Dense rank 2) where the model generates legislative-register text that matches the corpus style. The clearest failure is "where do zebras live" (HyDE rank 6 vs Dense rank 4), the hypothetical is factually correct but the query is so simple that vanilla dense already finds the answer, and HyDE's expansion probably introduces noise. 
   
   On SciFact, HyDE wins on longer claim-style queries like "In mice, P. chabaudi parasites are able to proliferate faster early in infection when inoculated at lower numbers" (HyDE rank 1 vs Dense rank 8) where the hypothetical precisely mirrors the experimental setting described in the relevant abstract. The fail case is "CR is associated with higher methylation age" (HyDE rank 8 vs Dense rank 6) — the model generates a hypothetical about C-reactive protein, but in SciFact "CR" refers to caloric restriction. Dense retrieval handles the ambiguous abbreviation better than an overconfident hypothetical that hallucinates the wrong concept.

5. **Does domain-specific bi-encoder (M4) outperform general (M3)? On which dataset?**

   Yes, and only on its home dataset. Dense-msmarco (M4) beats Dense-MiniLM (M3) on MS MARCO by a solid margin (NDCG@10 0.723 vs 0.665, MRR@10 0.635 vs 0.566, P@1 0.436 vs 0.370). The model was fine-tuned specifically on MS MARCO passage retrieval, so this is expected. 
   
   On SciFact the result completely reverses where Dense-MiniLM gets NDCG@10 0.645 vs Dense-msmarco's 0.538, a gap of 10.7 points in the other direction. Dense-msmarco was trained on informal web queries and falls apart on scientific claims with precise biochemical vocabulary. Dense-MiniLM, trained on a diverse sentence-pair corpus, generalizes across domains much better. 
   
   The takeaway is that domain-specific fine-tuning is a double-edged sword: it helps a lot in-domain but can actively hurt out-of-domain, sometimes worse than a general-purpose model that never saw the domain at all.

6. **Does hybrid (M5) beat both individual methods? Any exceptions?**

   It depends on the dataset and the metric. 
   
   On MS MARCO, Hybrid-RRF consistently fails to beat Dense-MiniLM alone (NDCG@10 0.620 vs 0.665, P@1 0.315 vs 0.370, MRR@10 0.517 vs 0.566). RRF is blending a strong signal (Dense-MiniLM) with a weaker one (BM25 on MS MARCO) and the result sits between the two, never exceeding the stronger component. 
   
   On SciFact the picture is more interesting: RRF beats both components on P@1 (0.527 vs BM25's 0.437 and Dense's 0.503), MRR@10 (0.606 vs 0.524 and 0.605), R@50 (0.901 vs 0.777 and 0.892), and R@100 (0.939 vs 0.793 and 0.925). The exception is NDCG@10 where RRF (0.638) still trails Dense-MiniLM (0.645). The reason RRF works better on SciFact is probably that BM25 handles exact scientific term matching that Dense misses, and Dense handles semantic paraphrasing that BM25 misses, so combining them genuinely recovers queries that neither gets alone. 

7. **What does re-ranking (M7) cost in latency vs. what it gains? Is it worth it?**

   The cross-encoder costs a lot but delivers meaningful gains on precision metrics. 
   
   On MS MARCO, latency goes from 13.5ms (Dense-MiniLM Stage 1) to 2664ms, a 197x increase. In return, NDCG@10 improves from 0.665 to 0.740 (+11%), MRR@10 from 0.566 to 0.659 (+16%), and P@1 from 0.370 to 0.484 (+31%). 
   
   On SciFact the latency hit is even steeper (393x, from 17ms to 6683ms) for smaller gains: NDCG@10 +6.5%, P@1 +14.7%. 
   
   Whether it is worth it depends entirely on the use case. 
   
   For real-time search where users expect sub-100ms response, no — 2–7 seconds is completely unacceptable. 
   
   For offline or high-stakes retrieval with a loose latency budget (enterprise document search, scientific literature review, legal discovery) the 31% P@1 improvement is a real win. The Stage-1 R@100 being 1.000 on MS MARCO also confirms there is no ceiling problem. It shows every relevant document is in the candidate set, and the cross-encoder is purely reordering, which is the ideal operating condition for a re-ranker.
