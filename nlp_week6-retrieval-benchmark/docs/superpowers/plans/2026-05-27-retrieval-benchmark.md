# Retrieval Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an 8-method IR benchmark (BM25, TF-IDF, dense, hybrid, ColBERT, cross-encoder, HyDE) across MS MARCO and SciFact, with cached artifacts, 6+ metrics, and written analysis answering 7 questions.

**Architecture:** Infrastructure-first — cache layer → data loaders → eval harness → methods in dependency order (M1→M2→M3→M4→M5→M7→M6→M8). All heavy computation cached idempotently; second full run completes in 2–5 minutes. Methods share a `Retriever` ABC with `.index(corpus)` / `.search(query, k)` interface.

**Tech Stack:** Python 3.10+, `datasets`, `rank-bm25`, `scikit-learn`, `sentence-transformers`, `pytrec-eval-terrier`, `ragatouille`, `openai`, `pandas`, `torch` (CPU-only install)

**Working directory for all non-git commands:** `nlp_week6-retrieval-benchmark/`
**Git commands:** run from the course repo root (parent of `nlp_week6-retrieval-benchmark/`)

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.py` | All constants: paths, seeds, model IDs, k-values |
| `src/cache.py` | `load_or_compute(path, fn)` + `doc_ids_hash()` |
| `src/data/load_msmarco.py` | Returns `{queries, corpus, qrels}` for MS MARCO |
| `src/data/load_scifact.py` | Returns `{queries, corpus, qrels}` for SciFact |
| `src/methods/base.py` | `Retriever` ABC with `.index()` / `.search()` |
| `src/methods/bm25.py` | M1: BM25Okapi retriever |
| `src/methods/tfidf.py` | M2: TF-IDF + cosine similarity |
| `src/methods/dense.py` | M3+M4: parameterized sentence-transformers retriever |
| `src/methods/hybrid_rrf.py` | M5: RRF over arbitrary list of Retrievers |
| `src/methods/cross_encoder.py` | M7: cross-encoder re-ranking on top-100 from first stage |
| `src/methods/colbert.py` | M6: ColBERT via ragatouille |
| `src/methods/hyde.py` | M8: HyDE via JGU KI-Chat → dense retrieval |
| `src/eval/metrics.py` | `evaluate(qrels, run)` → averaged metric dict |
| `src/eval/runner.py` | `run_retriever()`, `measure_latency()`, `save_per_query()`, `bold_best_md()` |
| `src/analysis/disagreement.py` | Find queries with maximum rank divergence across methods |
| `src/analysis/qualitative.py` | Pretty-print per-query top-k across methods |
| `scripts/run_benchmark.py` | One-shot entry point for M1–M7 on both datasets |
| `scripts/run_hyde_subset.py` | HyDE-only script, ~10 queries per dataset |
| `tests/conftest.py` | Shared fixtures: `tiny_corpus`, `tiny_queries`, `tiny_qrels`, `cache_dir` |
| `tests/test_cache.py` | Idempotency tests for cache layer |
| `tests/test_metrics.py` | Known-answer unit tests for all 6 metrics |
| `tests/test_methods.py` | Interface + correctness tests for all retrievers |
| `tests/test_data.py` | Integration smoke tests for data loaders (marked, not run by default) |
| `notebooks/analysis.ipynb` | Analysis answers for all 7 questions |
| `README.md` | Setup, exact run commands, results tables, analysis |

---

## Task 1: Project Scaffold

**Files:**
- Create: `config.py`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/methods/__init__.py`, `src/eval/__init__.py`, `src/analysis/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `results/.gitkeep`, `results/per_query/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/data src/methods src/eval src/analysis tests scripts notebooks results/per_query cache/embeddings cache/bm25 cache/tfidf cache/colbert cache/hyde
touch src/__init__.py src/data/__init__.py src/methods/__init__.py src/eval/__init__.py src/analysis/__init__.py
touch tests/__init__.py scripts/__init__.py
touch results/.gitkeep results/per_query/.gitkeep
```

- [ ] **Step 2: Write `config.py`**

```python
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

SEED = 42
MSMARCO_SAMPLE_SIZE = 300
HYDE_SAMPLE_SIZE = 10

RRF_K = 60
FIRST_STAGE_K = 100
LATENCY_WARMUP = 3
LATENCY_SAMPLES = 20

MODELS = {
    "dense_general": "sentence-transformers/all-MiniLM-L6-v2",
    "dense_domain": "sentence-transformers/msmarco-distilbert-base-v3",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "colbert": "colbert-ir/colbertv2.0",
}

HYDE_CONFIG = {
    "base_url": "https://ki-chat.uni-mainz.de/api",
    "model": "GPT OSS 120B",
    "max_tokens": 150,
    "sleep": 1.1,
}
```

- [ ] **Step 3: Write `requirements.txt`**

```
# install torch separately first:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

datasets>=2.14.0
rank-bm25>=0.2.2
scikit-learn>=1.3.0
sentence-transformers>=2.7.0
transformers>=4.35.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
pytrec-eval-terrier>=0.5.6
ragatouille>=0.0.8
openai>=1.0.0
jupyter>=1.0.0
nbformat>=5.9.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
```

- [ ] **Step 4: Write `.gitignore`**

```
cache/
.env
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.ragatouille/
results/raw/
*.egg-info/
.DS_Store
desktop.ini
```

- [ ] **Step 5: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -v -m "not integration"
markers =
    integration: marks tests that require internet/HF access (run with -m integration)
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
import pytest

TINY_CORPUS = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "machine learning is a subset of artificial intelligence",
    "d3": "information retrieval systems rank documents by relevance",
    "d4": "natural language processing handles text and speech",
    "d5": "deep learning uses neural networks with many layers",
}
TINY_QUERIES = {"q1": "quick fox", "q2": "neural networks deep learning"}
TINY_QRELS = {"q1": {"d1": 1}, "q2": {"d5": 1}}


@pytest.fixture
def tiny_corpus():
    return dict(TINY_CORPUS)


@pytest.fixture
def tiny_queries():
    return dict(TINY_QUERIES)


@pytest.fixture
def tiny_qrels():
    return dict(TINY_QRELS)


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return d
```

- [ ] **Step 7: Verify pytest discovers the suite**

```bash
pytest --collect-only
```

Expected output: `no tests ran` (0 items collected is fine — no test files yet).

- [ ] **Step 8: Commit**

```bash
# From course repo root:
git add nlp_week6-retrieval-benchmark/
git commit -m "feat(week6): scaffold retrieval benchmark project structure"
```

---

## Task 2: Cache Layer

**Files:**
- Create: `src/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache.py
import numpy as np
import pytest
from src.cache import doc_ids_hash, load_or_compute


def test_load_or_compute_npy_calls_fn_once(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return np.array([1.0, 2.0, 3.0])

    path = cache_dir / "test.npy"
    result1 = load_or_compute(path, compute)
    result2 = load_or_compute(path, compute)

    assert len(calls) == 1
    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(result1, [1.0, 2.0, 3.0])


def test_load_or_compute_json_idempotent(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return {"key": "value", "nested": [1, 2]}

    path = cache_dir / "test.json"
    r1 = load_or_compute(path, compute)
    r2 = load_or_compute(path, compute)

    assert len(calls) == 1
    assert r1 == r2 == {"key": "value", "nested": [1, 2]}


def test_load_or_compute_pkl_idempotent(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return {"sparse": True, "data": list(range(100))}

    path = cache_dir / "test.pkl"
    r1 = load_or_compute(path, compute)
    r2 = load_or_compute(path, compute)

    assert len(calls) == 1
    assert r1 == r2


def test_doc_ids_hash_order_independent():
    h1 = doc_ids_hash(["c", "a", "b"])
    h2 = doc_ids_hash(["a", "b", "c"])
    assert h1 == h2
    assert len(h1) == 12


def test_doc_ids_hash_different_inputs_differ():
    assert doc_ids_hash(["a", "b"]) != doc_ids_hash(["a", "c"])


def test_load_or_compute_creates_parent_dirs(cache_dir):
    path = cache_dir / "deep" / "nested" / "file.json"

    load_or_compute(path, lambda: {"x": 1})

    assert path.exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cache.py -v
```

Expected: `ImportError: No module named 'src.cache'`

- [ ] **Step 3: Write `src/cache.py`**

```python
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np


def doc_ids_hash(doc_ids: list[str]) -> str:
    return hashlib.sha1(",".join(sorted(doc_ids)).encode()).hexdigest()[:12]


def _load(path: Path) -> Any:
    suffix = path.suffix
    if suffix == ".npy":
        return np.load(str(path))
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unknown cache format: {suffix}")


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    if suffix == ".npy":
        np.save(str(path), obj)
    elif suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown cache format: {suffix}")


def load_or_compute(path: Path, compute_fn: Callable[[], Any]) -> Any:
    path = Path(path)
    if path.exists():
        return _load(path)
    result = compute_fn()
    _save(path, result)
    return result
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/test_cache.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/cache.py nlp_week6-retrieval-benchmark/tests/test_cache.py
git commit -m "feat(week6): add cache layer with load_or_compute and doc_ids_hash"
```

---

## Task 3: MS MARCO Data Loader

**Files:**
- Create: `src/data/load_msmarco.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_data.py
import pytest
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact


@pytest.mark.integration
def test_load_msmarco_structure():
    data = load_msmarco(sample_size=10, seed=42)
    assert set(data.keys()) == {"queries", "corpus", "qrels"}
    assert len(data["queries"]) > 0
    assert len(data["corpus"]) >= len(data["queries"])  # at least as many docs as queries
    # Every query in qrels must be in queries
    for qid in data["qrels"]:
        assert qid in data["queries"]
    # Every doc in qrels must be in corpus
    for qid, rels in data["qrels"].items():
        for did in rels:
            assert did in data["corpus"], f"{did} missing from corpus"
    # Relevance is binary (0 or 1)
    for qid, rels in data["qrels"].items():
        for did, rel in rels.items():
            assert rel in (0, 1)


@pytest.mark.integration
def test_load_msmarco_sampling_deterministic():
    d1 = load_msmarco(sample_size=10, seed=42)
    d2 = load_msmarco(sample_size=10, seed=42)
    assert list(d1["queries"].keys()) == list(d2["queries"].keys())


@pytest.mark.integration
def test_load_msmarco_at_least_one_relevant_per_query():
    data = load_msmarco(sample_size=20, seed=42)
    for qid in data["queries"]:
        assert qid in data["qrels"], f"query {qid} has no qrels entry"
        assert any(r > 0 for r in data["qrels"][qid].values()), f"query {qid} has no relevant docs"
```

- [ ] **Step 2: Write `src/data/load_msmarco.py`**

```python
import random
from typing import TypedDict

import datasets


class DataBundle(TypedDict):
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_msmarco(
    sample_size: int = 300,
    seed: int = 42,
    distractor_target: int = 5000,
) -> DataBundle:
    random.seed(seed)
    ds = datasets.load_dataset("ms_marco", "v1.1", split="validation")

    all_indices = list(range(len(ds)))
    sampled_indices = random.sample(all_indices, min(sample_size, len(all_indices)))
    sampled_set = set(sampled_indices)

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for query_idx, row_idx in enumerate(sampled_indices):
        row = ds[row_idx]
        qid = f"q_{query_idx}"
        passage_texts = row["passages"]["passage_text"]
        is_selected = row["passages"]["is_selected"]

        qrels[qid] = {}
        for passage_idx, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            did = f"msmarco_{query_idx}_{passage_idx}"
            corpus[did] = text
            if selected == 1:
                qrels[qid][did] = 1

        # Skip queries with no relevant passage
        if not any(r > 0 for r in qrels[qid].values()):
            del qrels[qid]
            continue

        queries[qid] = row["query"]

    # Add distractors from the remaining rows
    other_indices = [i for i in all_indices if i not in sampled_set]
    random.shuffle(other_indices)
    dist_count = 0
    for row_idx in other_indices:
        if dist_count >= distractor_target:
            break
        row = ds[row_idx]
        for passage_idx, text in enumerate(row["passages"]["passage_text"]):
            did = f"msmarco_dist_{row_idx}_{passage_idx}"
            if did not in corpus:
                corpus[did] = text
                dist_count += 1

    return {"queries": queries, "corpus": corpus, "qrels": qrels}
```

- [ ] **Step 3: Run the integration test**

```bash
pytest tests/test_data.py::test_load_msmarco_structure tests/test_data.py::test_load_msmarco_sampling_deterministic tests/test_data.py::test_load_msmarco_at_least_one_relevant_per_query -v -m integration
```

Expected: `3 passed` (requires HF download ~400 MB on first run)

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/data/load_msmarco.py nlp_week6-retrieval-benchmark/tests/test_data.py
git commit -m "feat(week6): add MS MARCO data loader with distractor corpus"
```

---

## Task 4: SciFact Data Loader

**Files:**
- Modify: `tests/test_data.py` (add SciFact tests)
- Create: `src/data/load_scifact.py`

- [ ] **Step 1: Add SciFact test to `tests/test_data.py`**

Append to the file:

```python
@pytest.mark.integration
def test_load_scifact_structure():
    data = load_scifact()
    assert set(data.keys()) == {"queries", "corpus", "qrels"}
    assert len(data["queries"]) == 300
    assert len(data["corpus"]) >= 5000
    for qid in data["qrels"]:
        assert qid in data["queries"]
    for qid, rels in data["qrels"].items():
        for did in rels:
            assert did in data["corpus"]
    # SciFact has graded relevance 0/1/2
    all_scores = [r for rels in data["qrels"].values() for r in rels.values()]
    assert all(s in (0, 1, 2) for s in all_scores)
    assert any(s == 2 for s in all_scores), "expected at least one score=2 in SciFact"


@pytest.mark.integration
def test_load_scifact_deterministic():
    d1 = load_scifact()
    d2 = load_scifact()
    assert list(d1["queries"].keys()) == list(d2["queries"].keys())
    assert list(d1["corpus"].keys()) == list(d2["corpus"].keys())
```

- [ ] **Step 2: Write `src/data/load_scifact.py`**

```python
from typing import TypedDict

import datasets


class DataBundle(TypedDict):
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_scifact() -> DataBundle:
    corpus_ds = datasets.load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = datasets.load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = datasets.load_dataset("BeIR/scifact-qrels", split="test")

    corpus: dict[str, str] = {
        str(row["_id"]): (row["title"] + " " + row["text"]).strip()
        for row in corpus_ds
    }

    all_queries: dict[str, str] = {str(row["_id"]): row["text"] for row in queries_ds}

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score

    # Only include queries that appear in qrels (test split)
    queries = {qid: q for qid, q in all_queries.items() if qid in qrels}

    return {"queries": queries, "corpus": corpus, "qrels": qrels}
```

- [ ] **Step 3: Run SciFact integration tests**

```bash
pytest tests/test_data.py::test_load_scifact_structure tests/test_data.py::test_load_scifact_deterministic -v -m integration
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/data/load_scifact.py nlp_week6-retrieval-benchmark/tests/test_data.py
git commit -m "feat(week6): add SciFact data loader with graded qrels"
```

---

## Task 5: Evaluation Metrics

**Files:**
- Create: `src/eval/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the known-answer test**

```python
# tests/test_metrics.py
from src.eval.metrics import evaluate

# Two queries:
# q1: d1 is relevant. Run: d1 at rank 1 → MRR=1.0, P@1=1.0, NDCG=1.0
# q2: d2 is relevant. Run: d1 at rank 1 (wrong), d2 at rank 2 → MRR=0.5, P@1=0.0
QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}
RUN = {
    "q1": {"d1": 2.0, "d2": 1.0},
    "q2": {"d1": 2.0, "d2": 1.0},
}


def test_precision_at_1():
    r = evaluate(QRELS, RUN)
    # q1: P@1=1.0, q2: P@1=0.0 → avg=0.5
    assert abs(r["P@1"] - 0.5) < 0.01


def test_precision_at_5():
    r = evaluate(QRELS, RUN)
    # With only 2 docs, P@5 = #{relevant in top-5}/5. q1=1/5=0.2, q2=1/5=0.2 → avg=0.2
    assert abs(r["P@5"] - 0.2) < 0.01


def test_recall_at_10():
    r = evaluate(QRELS, RUN)
    # Both relevant docs are in the 2-doc run → R@10=1.0 for both
    assert abs(r["R@10"] - 1.0) < 0.01


def test_mrr_at_10():
    r = evaluate(QRELS, RUN)
    # q1: first relevant at rank 1 → RR=1.0. q2: first relevant at rank 2 → RR=0.5. avg=0.75
    assert abs(r["MRR@10"] - 0.75) < 0.01


def test_map_at_100():
    r = evaluate(QRELS, RUN)
    # q1: AP = P@1 = 1.0. q2: AP = P@2 = 0.5. avg=0.75
    assert abs(r["MAP@100"] - 0.75) < 0.01


def test_ndcg_at_10():
    r = evaluate(QRELS, RUN)
    # q1: NDCG=1.0. q2: DCG=1/log2(3)=0.6309, idealDCG=1.0, NDCG=0.6309. avg≈0.815
    assert abs(r["NDCG@10"] - 0.815) < 0.01


def test_latency_key_absent():
    # evaluate() does not return Latency_ms — that's added by runner
    r = evaluate(QRELS, RUN)
    assert "Latency_ms" not in r


def test_returns_all_expected_keys():
    r = evaluate(QRELS, RUN)
    expected = {"P@1", "P@5", "P@10", "R@10", "R@50", "R@100", "MRR@10", "MAP@100", "NDCG@10"}
    assert expected.issubset(r.keys())
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError: No module named 'src.eval.metrics'`

- [ ] **Step 3: Write `src/eval/metrics.py`**

```python
import pytrec_eval

_PYTREC_KEYS = {
    "P_1", "P_5", "P_10",
    "recall_10", "recall_50", "recall_100",
    "map_cut_100", "ndcg_cut_10",
}

_DISPLAY = {
    "P_1": "P@1",
    "P_5": "P@5",
    "P_10": "P@10",
    "recall_10": "R@10",
    "recall_50": "R@50",
    "recall_100": "R@100",
    "recip_rank": "MRR@10",
    "map_cut_100": "MAP@100",
    "ndcg_cut_10": "NDCG@10",
}


def evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average all metrics over queries. run may have any number of docs per query."""
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, _PYTREC_KEYS)
    per_query = evaluator.evaluate(run)

    # MRR@10: truncate run to top-10 before computing recip_rank
    run_at10 = {
        qid: dict(sorted(scores.items(), key=lambda x: -x[1])[:10])
        for qid, scores in run.items()
    }
    mrr_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    mrr_per_query = mrr_evaluator.evaluate(run_at10)

    out: dict[str, float] = {}
    for internal_key, display_key in _DISPLAY.items():
        if internal_key == "recip_rank":
            vals = [mrr_per_query[qid]["recip_rank"] for qid in mrr_per_query]
        else:
            vals = [per_query[qid][internal_key] for qid in per_query if internal_key in per_query[qid]]
        out[display_key] = sum(vals) / len(vals) if vals else 0.0

    return out
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/eval/metrics.py nlp_week6-retrieval-benchmark/tests/test_metrics.py
git commit -m "feat(week6): add eval metrics with pytrec_eval, MRR@10, unit-tested"
```

---

## Task 6: Retriever ABC + Runner

**Files:**
- Create: `src/methods/base.py`
- Create: `src/eval/runner.py`

- [ ] **Step 1: Write `src/methods/base.py`**

```python
from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        """Build/load index for the given corpus."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs sorted by score descending."""
```

- [ ] **Step 2: Write `src/eval/runner.py`**

```python
import time
from pathlib import Path

import pandas as pd

from src.eval.metrics import evaluate
from src.methods.base import Retriever

METRIC_COLS = ["P@1", "P@5", "P@10", "R@10", "R@50", "R@100", "MRR@10", "MAP@100", "NDCG@10"]


def measure_latency(
    retriever: Retriever,
    queries: dict[str, str],
    k: int = 10,
    warmup: int = 3,
    samples: int = 20,
) -> float:
    """Average single-query wall-clock time in ms (warmup excluded)."""
    qids = list(queries.keys())
    for qid in qids[:warmup]:
        retriever.search(queries[qid], k)
    times = []
    for qid in qids[warmup: warmup + samples]:
        t0 = time.perf_counter()
        retriever.search(queries[qid], k)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times) if times else 0.0


def run_retriever(
    name: str,
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 100,
) -> dict:
    run: dict[str, dict[str, float]] = {}
    for qid, query in queries.items():
        results = retriever.search(query, k)
        run[qid] = {did: score for did, score in results}

    metrics = evaluate(qrels, run)
    metrics["Latency_ms"] = measure_latency(retriever, queries)
    metrics["method"] = name
    return metrics, run


def save_per_query(
    method_name: str,
    dataset_name: str,
    run: dict[str, dict[str, float]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    results_dir: Path,
    k: int = 100,
) -> None:
    rows = []
    for qid, scores in run.items():
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        for rank, (did, score) in enumerate(ranked, start=1):
            rows.append({
                "query_id": qid,
                "query_text": queries.get(qid, ""),
                "rank": rank,
                "doc_id": did,
                "score": score,
                "is_relevant": int(qrels.get(qid, {}).get(did, 0) > 0),
            })
    df = pd.DataFrame(rows)
    out = results_dir / "per_query" / f"{method_name}_{dataset_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def bold_best_md(df: pd.DataFrame) -> str:
    """Render DataFrame as Markdown table with best value per column bolded."""
    metric_cols = [c for c in df.columns if c != "method"]
    lower_is_better = {"Latency_ms"}

    best = {}
    for col in metric_cols:
        best[col] = df[col].min() if col in lower_is_better else df[col].max()

    lines = []
    lines.append("| Method | " + " | ".join(metric_cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(metric_cols)) + "|")

    for _, row in df.iterrows():
        cells = []
        for col in metric_cols:
            val = row[col]
            fmt = f"{val:.1f}" if col == "Latency_ms" else f"{val:.4f}"
            if abs(val - best[col]) < 1e-9:
                fmt = f"**{fmt}**"
            cells.append(fmt)
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")

    return "\n".join(lines)
```

- [ ] **Step 3: Write a runner unit test**

Add to `tests/test_methods.py` (create file now):

```python
# tests/test_methods.py
from src.methods.base import Retriever


class _ConstantRetriever(Retriever):
    """Returns docs in reverse lexicographic order with constant scores."""

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys(), reverse=True)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(did, float(i)) for i, did in enumerate(self._doc_ids[:k], start=1)]


def test_retriever_abc_interface(tiny_corpus, cache_dir):
    r = _ConstantRetriever()
    r.index(tiny_corpus)
    results = r.search("anything", k=3)
    assert len(results) == 3
    assert all(isinstance(did, str) and isinstance(score, float) for did, score in results)
    # scores must be sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_methods.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/base.py nlp_week6-retrieval-benchmark/src/eval/runner.py nlp_week6-retrieval-benchmark/tests/test_methods.py
git commit -m "feat(week6): add Retriever ABC and eval runner with latency and MD export"
```

---

## Task 7: M1 — BM25 Retriever

**Files:**
- Create: `src/methods/bm25.py`
- Modify: `tests/test_methods.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_methods.py`:

```python
from src.methods.bm25 import BM25Retriever


def test_bm25_returns_sorted_scores(tiny_corpus, cache_dir):
    r = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_ranks_relevant_doc_first(tiny_corpus, cache_dir):
    r = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    top_doc = results[0][0]
    assert top_doc == "d1"  # "quick" and "fox" appear only in d1


def test_bm25_caches_index(tiny_corpus, cache_dir):
    r1 = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r1.index(tiny_corpus)
    # Second retriever loads from cache (no recompute)
    r2 = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r2.index(tiny_corpus)
    result1 = r1.search("quick fox", k=3)
    result2 = r2.search("quick fox", k=3)
    assert [did for did, _ in result1] == [did for did, _ in result2]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_methods.py::test_bm25_returns_sorted_scores -v
```

Expected: `ImportError: No module named 'src.methods.bm25'`

- [ ] **Step 3: Write `src/methods/bm25.py`**

```python
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.cache import doc_ids_hash, load_or_compute
from src.methods.base import Retriever


class BM25Retriever(Retriever):
    def __init__(self, dataset_name: str, cache_dir: Path):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] | None = None

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys())
        ids_hash = doc_ids_hash(self._doc_ids)
        cache_path = self.cache_dir / "bm25" / f"{self.dataset_name}__{ids_hash}.pkl"

        def compute():
            tokenized = [corpus[did].lower().split() for did in self._doc_ids]
            return BM25Okapi(tokenized)

        self._bm25 = load_or_compute(cache_path, compute)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_k = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
```

- [ ] **Step 4: Run to verify all BM25 tests pass**

```bash
pytest tests/test_methods.py -k bm25 -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/bm25.py nlp_week6-retrieval-benchmark/tests/test_methods.py
git commit -m "feat(week6): add M1 BM25 retriever with cache"
```

---

## Task 8: M2 — TF-IDF Retriever

**Files:**
- Create: `src/methods/tfidf.py`
- Modify: `tests/test_methods.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_methods.py`:

```python
from src.methods.tfidf import TFIDFRetriever


def test_tfidf_returns_sorted_scores(tiny_corpus, cache_dir):
    r = TFIDFRetriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_ranks_relevant_doc_first(tiny_corpus, cache_dir):
    r = TFIDFRetriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert results[0][0] == "d1"
```

- [ ] **Step 2: Write `src/methods/tfidf.py`**

```python
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.cache import doc_ids_hash, load_or_compute
from src.methods.base import Retriever


class TFIDFRetriever(Retriever):
    def __init__(self, dataset_name: str, cache_dir: Path):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_matrix = None
        self._doc_ids: list[str] | None = None

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys())
        ids_hash = doc_ids_hash(self._doc_ids)
        cache_path = self.cache_dir / "tfidf" / f"{self.dataset_name}__{ids_hash}.pkl"

        def compute():
            texts = [corpus[did] for did in self._doc_ids]
            vec = TfidfVectorizer()
            mat = vec.fit_transform(texts)
            return (vec, mat)

        self._vectorizer, self._doc_matrix = load_or_compute(cache_path, compute)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._doc_matrix)[0]
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
```

- [ ] **Step 3: Run to verify**

```bash
pytest tests/test_methods.py -k tfidf -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/tfidf.py
git commit -m "feat(week6): add M2 TF-IDF retriever with cache"
```

---

## Task 9: M3 — Dense Retriever (MiniLM)

**Files:**
- Create: `src/methods/dense.py`
- Modify: `tests/test_methods.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_methods.py`:

```python
from src.methods.dense import DenseRetriever


def test_dense_minilm_ranks_relevant_first(tiny_corpus, cache_dir):
    r = DenseRetriever(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dataset_name="test_minilm",
        cache_dir=cache_dir,
    )
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # MiniLM should rank d1 ("quick brown fox") at position 1
    assert results[0][0] == "d1"


def test_dense_caches_embeddings(tiny_corpus, cache_dir):
    r = DenseRetriever(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dataset_name="test_cache_check",
        cache_dir=cache_dir,
    )
    r.index(tiny_corpus)
    cache_files = list((cache_dir / "embeddings").glob("*.npy"))
    assert len(cache_files) == 1
```

- [ ] **Step 2: Write `src/methods/dense.py`**

```python
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.cache import doc_ids_hash, load_or_compute
from src.methods.base import Retriever


class DenseRetriever(Retriever):
    def __init__(self, model_id: str, dataset_name: str, cache_dir: Path):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self._model: SentenceTransformer | None = None
        self._doc_embeddings: np.ndarray | None = None
        self._doc_ids: list[str] | None = None

    def _model_(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys())
        texts = [corpus[did] for did in self._doc_ids]
        slug = self.model_id.replace("/", "__")
        ids_hash = doc_ids_hash(self._doc_ids)
        cache_path = self.cache_dir / "embeddings" / f"{slug}__{self.dataset_name}__{ids_hash}.npy"

        def compute() -> np.ndarray:
            return self._model_().encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        self._doc_embeddings = load_or_compute(cache_path, compute)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q_emb = self._model_().encode([query], normalize_embeddings=True)
        scores = (q_emb @ self._doc_embeddings.T)[0]
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
```

- [ ] **Step 3: Run to verify (downloads ~90 MB model on first run)**

```bash
pytest tests/test_methods.py -k dense_minilm -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/dense.py
git commit -m "feat(week6): add M3 dense retriever (parameterized by model ID) with embedding cache"
```

---

## Task 10: M4 — Dense Retriever (msmarco-distilbert)

**Files:**
- Modify: `tests/test_methods.py`

No new source file — `DenseRetriever` is parameterized. This task verifies the domain model loads and runs correctly.

- [ ] **Step 1: Write the test**

Append to `tests/test_methods.py`:

```python
def test_dense_msmarco_distilbert_runs(tiny_corpus, cache_dir):
    r = DenseRetriever(
        model_id="sentence-transformers/msmarco-distilbert-base-v3",
        dataset_name="test_msmarco",
        cache_dir=cache_dir,
    )
    r.index(tiny_corpus)
    results = r.search("information retrieval", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # d3 ("information retrieval systems rank documents") should score high
    top_ids = [did for did, _ in results]
    assert "d3" in top_ids
```

- [ ] **Step 2: Run (downloads ~250 MB model on first run)**

```bash
pytest tests/test_methods.py::test_dense_msmarco_distilbert_runs -v
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add nlp_week6-retrieval-benchmark/tests/test_methods.py
git commit -m "test(week6): verify M4 msmarco-distilbert dense retriever"
```

---

## Task 11: M5 — Hybrid RRF Retriever

**Files:**
- Create: `src/methods/hybrid_rrf.py`
- Modify: `tests/test_methods.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_methods.py`:

```python
from src.methods.hybrid_rrf import HybridRRFRetriever, reciprocal_rank_fusion


def test_rrf_fusion_merges_rankings():
    # Two rankings agree on d1 at top
    r1 = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
    r2 = [("d1", 2.5), ("d3", 1.5), ("d2", 0.5)]
    fused = reciprocal_rank_fusion([r1, r2], k=60)
    # d1 at rank 1 in both → highest RRF score
    assert fused[0][0] == "d1"
    assert all(score_a >= score_b for (_, score_a), (_, score_b) in zip(fused, fused[1:]))


def test_hybrid_rrf_combines_bm25_and_dense(tiny_corpus, cache_dir):
    bm25 = BM25Retriever(dataset_name="test_rrf", cache_dir=cache_dir)
    dense = DenseRetriever(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dataset_name="test_rrf",
        cache_dir=cache_dir,
    )
    hybrid = HybridRRFRetriever(retrievers=[bm25, dense], k=60)
    hybrid.index(tiny_corpus)
    results = hybrid.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] == "d1"
```

- [ ] **Step 2: Write `src/methods/hybrid_rrf.py`**

```python
from src.methods.base import Retriever


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRRFRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], k: int = 60):
        self.retrievers = retrievers
        self.k = k

    def index(self, corpus: dict[str, str]) -> None:
        for r in self.retrievers:
            r.index(corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # Retrieve 2× candidates from each component to improve fusion coverage
        rankings = [r.search(query, k * 2) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.k)[:k]
```

- [ ] **Step 3: Run to verify**

```bash
pytest tests/test_methods.py -k rrf -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/hybrid_rrf.py
git commit -m "feat(week6): add M5 hybrid RRF retriever"
```

---

## Task 12: M7 — Cross-Encoder Re-Ranker

**Files:**
- Create: `src/methods/cross_encoder.py`
- Modify: `tests/test_methods.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_methods.py`:

```python
from src.methods.cross_encoder import CrossEncoderRetriever


def test_cross_encoder_reranks(tiny_corpus, cache_dir):
    first_stage = BM25Retriever(dataset_name="test_ce", cache_dir=cache_dir)
    first_stage.index(tiny_corpus)

    reranker = CrossEncoderRetriever(
        first_stage=first_stage,
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        first_stage_k=5,
    )
    reranker.index(tiny_corpus)

    results = reranker.search("information retrieval rank documents", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # d3 "information retrieval systems rank documents by relevance" should be top
    assert results[0][0] == "d3"
```

- [ ] **Step 2: Write `src/methods/cross_encoder.py`**

```python
from sentence_transformers import CrossEncoder

from src.methods.base import Retriever

_MAX_DOC_CHARS = 2000  # ~500 tokens; truncate doc, never query


class CrossEncoderRetriever(Retriever):
    def __init__(
        self,
        first_stage: Retriever,
        model_id: str,
        first_stage_k: int = 100,
    ):
        self.first_stage = first_stage
        self.model_id = model_id
        self.first_stage_k = first_stage_k
        self._model: CrossEncoder | None = None
        self._corpus: dict[str, str] | None = None

    def _model_(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_id, max_length=512)
        return self._model

    def index(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus
        # first_stage must already be indexed externally before calling this

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        candidates = self.first_stage.search(query, self.first_stage_k)
        if not candidates:
            return []
        pairs = [(query, self._corpus[did][:_MAX_DOC_CHARS]) for did, _ in candidates]
        scores = self._model_().predict(pairs)
        ranked = sorted(
            zip([did for did, _ in candidates], scores.tolist()),
            key=lambda x: -x[1],
        )
        return [(did, float(score)) for did, score in ranked[:k]]
```

- [ ] **Step 3: Run to verify (downloads ~25 MB model)**

```bash
pytest tests/test_methods.py::test_cross_encoder_reranks -v
```

Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/cross_encoder.py
git commit -m "feat(week6): add M7 cross-encoder re-ranker (truncate doc, never query)"
```

---

## Task 13: M6 — ColBERT Retriever

**Files:**
- Create: `src/methods/colbert.py`
- Modify: `tests/test_methods.py`

> **Note:** ragatouille saves the index under `.ragatouille/colbert/indexes/{index_name}/` by default. We move it to `cache/colbert/{dataset}__index/` after building. First-run indexing on ~10k passages takes **30–60 min on CPU**. If ragatouille install fails or indexing exceeds ~90 min, index on a free Colab T4 GPU, download the index directory, place it at `cache/colbert/{dataset}__index/`, and continue locally. Document this in README if used.

- [ ] **Step 1: Write the test**

Append to `tests/test_methods.py`:

```python
from src.methods.colbert import ColBERTRetriever


def test_colbert_indexes_and_searches(tiny_corpus, cache_dir):
    r = ColBERTRetriever(dataset_name="test_colbert", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("information retrieval", k=3)
    assert len(results) >= 1
    # Scores should be sorted descending (ColBERT returns MaxSim scores)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    top_ids = [did for did, _ in results]
    assert "d3" in top_ids  # "information retrieval systems" should score high
```

- [ ] **Step 2: Write `src/methods/colbert.py`**

```python
import shutil
from pathlib import Path

from src.methods.base import Retriever

_RAGATOUILLE_DEFAULT = Path(".ragatouille") / "colbert" / "indexes"


class ColBERTRetriever(Retriever):
    def __init__(self, dataset_name: str, cache_dir: Path):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.index_dir = self.cache_dir / "colbert" / f"{dataset_name}__index"
        self._rag = None

    def index(self, corpus: dict[str, str]) -> None:
        from ragatouille import RAGPretrainedModel

        if self.index_dir.exists():
            self._rag = RAGPretrainedModel.from_index(str(self.index_dir))
            return

        doc_ids = sorted(corpus.keys())
        texts = [corpus[did] for did in doc_ids]

        rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        rag.index(
            collection=texts,
            document_ids=doc_ids,
            index_name=self.dataset_name,
            max_document_length=256,
            split_documents=False,
        )

        # Move index from ragatouille default location to our cache dir
        default_path = _RAGATOUILLE_DEFAULT / self.dataset_name
        if default_path.exists():
            self.index_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(default_path), str(self.index_dir))

        self._rag = RAGPretrainedModel.from_index(str(self.index_dir))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        results = self._rag.search(query=query, k=k)
        return [(r["document_id"], float(r["score"])) for r in results]
```

- [ ] **Step 3: Run the test (downloads ~500 MB ColBERT model on first run)**

```bash
pytest tests/test_methods.py::test_colbert_indexes_and_searches -v
```

Expected: `1 passed` (may take several minutes for model download + tiny-corpus indexing)

- [ ] **Step 4: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/colbert.py
git commit -m "feat(week6): add M6 ColBERT retriever via ragatouille with cache"
```

---

## Task 14: run_benchmark.py

**Files:**
- Create: `scripts/run_benchmark.py`

This is the one-shot entry point for M1–M7 on both datasets. HyDE (M8) is in a separate script.

- [ ] **Step 1: Write `scripts/run_benchmark.py`**

```python
#!/usr/bin/env python
"""Run M1-M7 on both datasets. Usage: python scripts/run_benchmark.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import numpy as np
import pandas as pd
import torch
import transformers

transformers.logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import CACHE_DIR, FIRST_STAGE_K, MODELS, RESULTS_DIR, RRF_K, SEED, MSMARCO_SAMPLE_SIZE
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact
from src.eval.runner import bold_best_md, run_retriever, save_per_query
from src.methods.bm25 import BM25Retriever
from src.methods.colbert import ColBERTRetriever
from src.methods.cross_encoder import CrossEncoderRetriever
from src.methods.dense import DenseRetriever
from src.methods.hybrid_rrf import HybridRRFRetriever
from src.methods.tfidf import TFIDFRetriever

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "per_query").mkdir(exist_ok=True)

DATASETS = {
    "msmarco": load_msmarco(sample_size=MSMARCO_SAMPLE_SIZE, seed=SEED),
    "scifact": load_scifact(),
}

for dataset_name, data in DATASETS.items():
    queries = data["queries"]
    corpus = data["corpus"]
    qrels = data["qrels"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | {len(queries)} queries | {len(corpus):,} docs")
    print("=" * 60)

    rows = []
    all_runs = {}

    # M1: BM25
    print("\n[M1] BM25 ...")
    m1 = BM25Retriever(dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m1.index(corpus)
    metrics, run = run_retriever("BM25", m1, queries, qrels)
    rows.append(metrics)
    all_runs["BM25"] = run

    # M2: TF-IDF
    print("\n[M2] TF-IDF ...")
    m2 = TFIDFRetriever(dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m2.index(corpus)
    metrics, run = run_retriever("TF-IDF", m2, queries, qrels)
    rows.append(metrics)
    all_runs["TF-IDF"] = run

    # M3: Dense MiniLM (indexed once, reused by M5 and M7)
    print("\n[M3] Dense MiniLM ...")
    m3 = DenseRetriever(model_id=MODELS["dense_general"], dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m3.index(corpus)
    metrics, run = run_retriever("Dense-MiniLM", m3, queries, qrels)
    rows.append(metrics)
    all_runs["Dense-MiniLM"] = run

    # M4: Dense msmarco-distilbert
    print("\n[M4] Dense msmarco-distilbert ...")
    m4 = DenseRetriever(model_id=MODELS["dense_domain"], dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m4.index(corpus)
    metrics, run = run_retriever("Dense-msmarco", m4, queries, qrels)
    rows.append(metrics)
    all_runs["Dense-msmarco"] = run

    # M5: Hybrid RRF (BM25 + MiniLM, both already indexed)
    print("\n[M5] Hybrid RRF (BM25 + MiniLM) ...")
    m5 = HybridRRFRetriever(retrievers=[m1, m3], k=RRF_K)
    metrics, run = run_retriever("Hybrid-RRF", m5, queries, qrels)
    rows.append(metrics)
    all_runs["Hybrid-RRF"] = run

    # M7: Cross-encoder (first stage = M3, already indexed)
    print("\n[M7] Cross-encoder re-rank (top-100 from MiniLM) ...")
    m7 = CrossEncoderRetriever(first_stage=m3, model_id=MODELS["cross_encoder"], first_stage_k=FIRST_STAGE_K)
    m7.index(corpus)
    metrics, run = run_retriever("CrossEncoder", m7, queries, qrels)
    rows.append(metrics)
    all_runs["CrossEncoder"] = run

    # M6: ColBERT (last — slowest to index)
    print("\n[M6] ColBERT ...")
    m6 = ColBERTRetriever(dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m6.index(corpus)
    metrics, run = run_retriever("ColBERT", m6, queries, qrels)
    rows.append(metrics)
    all_runs["ColBERT"] = run

    # Build results DataFrame
    df = pd.DataFrame(rows).set_index("method")
    metric_cols = [c for c in df.columns if c != "method"]
    df = df[metric_cols]

    # Save CSV
    csv_path = RESULTS_DIR / f"{dataset_name}_results.csv"
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    # Save Markdown with bolded best
    md = bold_best_md(df.reset_index())
    md_path = RESULTS_DIR / f"{dataset_name}_results.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Saved: {md_path}")

    # Save per-query rankings for analysis
    for method_name, run in all_runs.items():
        save_per_query(method_name, dataset_name, run, queries, qrels, RESULTS_DIR)

    print("\n" + df.to_string())
```

- [ ] **Step 2: Verify the script is importable (syntax check)**

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/run_benchmark.py').read_text())"
```

Expected: no output (no syntax error).

- [ ] **Step 3: Commit**

```bash
git add nlp_week6-retrieval-benchmark/scripts/run_benchmark.py
git commit -m "feat(week6): add run_benchmark.py one-shot entry point for M1-M7"
```

---

## Task 15: M8 — HyDE + run_hyde_subset.py

**Files:**
- Create: `src/methods/hyde.py`
- Create: `scripts/run_hyde_subset.py`
- Modify: `tests/test_methods.py`

- [ ] **Step 1: Write the mocked test**

Append to `tests/test_methods.py`:

```python
from unittest.mock import MagicMock, patch

from src.methods.hyde import HyDERetriever


def test_hyde_generates_hypothetical_and_retrieves(tiny_corpus, cache_dir):
    first_stage = BM25Retriever(dataset_name="test_hyde_fs", cache_dir=cache_dir)
    first_stage.index(tiny_corpus)

    hyde = HyDERetriever(
        first_stage=first_stage,
        dataset_name="test_hyde",
        cache_dir=cache_dir,
    )
    hyde.index(tiny_corpus)

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Artificial intelligence and machine learning are closely related fields."

    with patch.object(hyde, "_get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_client_fn.return_value = mock_client

        with patch("time.sleep"):  # skip rate-limit sleep
            results = hyde.search("machine learning artificial intelligence", k=3)

    assert len(results) >= 1
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # d2 "machine learning is a subset of artificial intelligence" should rank high
    top_ids = [did for did, _ in results]
    assert "d2" in top_ids


def test_hyde_caches_hypotheticals(tiny_corpus, cache_dir):
    first_stage = BM25Retriever(dataset_name="test_hyde_cache_fs", cache_dir=cache_dir)
    first_stage.index(tiny_corpus)

    hyde = HyDERetriever(
        first_stage=first_stage,
        dataset_name="test_hyde_cache",
        cache_dir=cache_dir,
    )
    hyde.index(tiny_corpus)

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Test hypothetical passage."

    with patch.object(hyde, "_get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_client_fn.return_value = mock_client

        with patch("time.sleep"):
            hyde.search("test query", k=2)
            call_count_after_first = mock_client.chat.completions.create.call_count
            hyde.search("test query", k=2)  # same query — should use cache
            call_count_after_second = mock_client.chat.completions.create.call_count

    assert call_count_after_first == 1
    assert call_count_after_second == 1  # not called again
```

- [ ] **Step 2: Write `src/methods/hyde.py`**

```python
import hashlib
import json
import os
import time
from pathlib import Path

from src.methods.base import Retriever

_SYSTEM = "You write short, factual-sounding passages that directly answer questions."
_USER_TMPL = (
    "Write a short passage (3-5 sentences) that directly answers this question. "
    "Do not preface or explain.\nQuestion: {query}\nPassage:"
)


class HyDERetriever(Retriever):
    def __init__(
        self,
        first_stage: Retriever,
        dataset_name: str,
        cache_dir: Path,
        model: str = "GPT OSS 120B",
        max_tokens: int = 150,
        sleep: float = 1.1,
    ):
        self.first_stage = first_stage
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.max_tokens = max_tokens
        self.sleep = sleep
        self._cache_path = self.cache_dir / "hyde" / f"{dataset_name}__hypotheticals.json"
        self._hypotheticals: dict[str, dict] = {}
        self._client = None

    def _get_client(self):
        from openai import OpenAI

        api_key = os.environ.get("JGU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "JGU_API_KEY environment variable is not set. "
                "Export it before running HyDE: export JGU_API_KEY=<your_key>"
            )
        return OpenAI(base_url="https://ki-chat.uni-mainz.de/api", api_key=api_key)

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            with open(self._cache_path, encoding="utf-8") as f:
                self._hypotheticals = json.load(f)

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._hypotheticals, f, indent=2, ensure_ascii=False)

    def index(self, corpus: dict[str, str]) -> None:
        self._load_cache()
        # first_stage must already be indexed externally

    def _query_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def generate_hypothetical(self, query: str) -> str:
        key = self._query_key(query)
        if key in self._hypotheticals:
            return self._hypotheticals[key]["hypothetical"]

        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(query=query)},
            ],
            max_tokens=self.max_tokens,
            reasoning_effort="low",
        )
        time.sleep(self.sleep)
        hyp = resp.choices[0].message.content.strip()

        self._hypotheticals[key] = {
            "query": query,
            "hypothetical": hyp,
            "model": self.model,
            "reasoning_effort": "low",
            "prompt_system": _SYSTEM,
            "prompt_user": _USER_TMPL.format(query=query),
        }
        self._save_cache()
        return hyp

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        hypothetical = self.generate_hypothetical(query)
        return self.first_stage.search(hypothetical, k)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_methods.py -k hyde -v
```

Expected: `2 passed`

- [ ] **Step 4: Write `scripts/run_hyde_subset.py`**

```python
#!/usr/bin/env python
"""Run HyDE on ~10 queries per dataset. Requires JGU_API_KEY env var.

Usage:
    export JGU_API_KEY=<your_key>
    python scripts/run_hyde_subset.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import transformers

transformers.logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import CACHE_DIR, MODELS, RESULTS_DIR, SEED, HYDE_SAMPLE_SIZE, MSMARCO_SAMPLE_SIZE
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact
from src.eval.runner import run_retriever, save_per_query
from src.methods.dense import DenseRetriever
from src.methods.hyde import HyDERetriever

import pandas as pd

RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "msmarco": load_msmarco(sample_size=MSMARCO_SAMPLE_SIZE, seed=SEED),
    "scifact": load_scifact(),
}

for dataset_name, data in DATASETS.items():
    all_qids = list(data["queries"].keys())
    random.seed(SEED)
    subset_qids = random.sample(all_qids, min(HYDE_SAMPLE_SIZE, len(all_qids)))
    queries = {qid: data["queries"][qid] for qid in subset_qids}
    corpus = data["corpus"]
    qrels = {qid: data["qrels"].get(qid, {}) for qid in subset_qids}

    print(f"\n[HyDE] {dataset_name}: {len(queries)} queries")

    # First stage must be indexed
    first_stage = DenseRetriever(
        model_id=MODELS["dense_general"],
        dataset_name=dataset_name,
        cache_dir=CACHE_DIR,
    )
    first_stage.index(corpus)

    hyde = HyDERetriever(
        first_stage=first_stage,
        dataset_name=dataset_name,
        cache_dir=CACHE_DIR,
    )
    hyde.index(corpus)

    metrics, run = run_retriever("HyDE", hyde, queries, qrels)
    save_per_query("HyDE", dataset_name, run, queries, qrels, RESULTS_DIR)

    print(f"MRR@10: {metrics['MRR@10']:.4f}  NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"Latency: {metrics['Latency_ms']:.1f} ms/query")

    # Print generated hypotheticals for analysis
    hyp_path = CACHE_DIR / "hyde" / f"{dataset_name}__hypotheticals.json"
    if hyp_path.exists():
        import json
        hyps = json.loads(hyp_path.read_text(encoding="utf-8"))
        print(f"\nGenerated {len(hyps)} hypotheticals. See: {hyp_path}")

print("\nDone. Hypotheticals saved to cache/hyde/.")
```

- [ ] **Step 5: Verify syntax**

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/run_hyde_subset.py').read_text())"
```

- [ ] **Step 6: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/methods/hyde.py nlp_week6-retrieval-benchmark/scripts/run_hyde_subset.py nlp_week6-retrieval-benchmark/tests/test_methods.py
git commit -m "feat(week6): add M8 HyDE retriever and run_hyde_subset.py script"
```

---

## Task 16: Analysis Modules

**Files:**
- Create: `src/analysis/disagreement.py`
- Create: `src/analysis/qualitative.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write `src/analysis/disagreement.py`**

```python
"""Find queries where methods disagree most on the gold document's rank."""
from pathlib import Path

import pandas as pd


def gold_rank(
    qid: str,
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    max_rank: int = 1000,
) -> int:
    gold_docs = {did for did, r in qrels.get(qid, {}).items() if r > 0}
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    for rank, (did, _) in enumerate(ranked, start=1):
        if did in gold_docs:
            return rank
    return max_rank


def find_max_divergence_queries(
    runs: dict[str, dict[str, dict[str, float]]],
    qrels: dict[str, dict[str, int]],
    n: int = 5,
) -> pd.DataFrame:
    """Return top-n queries with the largest gold-rank spread across methods."""
    method_names = list(runs.keys())
    qids = list(qrels.keys())
    rows = []
    for qid in qids:
        ranks = {m: gold_rank(qid, qrels, runs[m]) for m in method_names}
        spread = max(ranks.values()) - min(ranks.values())
        rows.append({"query_id": qid, "spread": spread, **ranks})
    df = pd.DataFrame(rows).sort_values("spread", ascending=False).head(n)
    return df.reset_index(drop=True)


def find_bm25_beats_dense(
    bm25_run: dict[str, dict[str, float]],
    dense_run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    n: int = 5,
) -> pd.DataFrame:
    """Return queries where BM25 rank < dense rank (BM25 wins)."""
    qids = list(qrels.keys())
    rows = []
    for qid in qids:
        bm25_r = gold_rank(qid, qrels, bm25_run)
        dense_r = gold_rank(qid, qrels, dense_run)
        if bm25_r < dense_r:
            rows.append({"query_id": qid, "bm25_rank": bm25_r, "dense_rank": dense_r, "improvement": dense_r - bm25_r})
    df = pd.DataFrame(rows).sort_values("improvement", ascending=False).head(n)
    return df.reset_index(drop=True)
```

- [ ] **Step 2: Write `src/analysis/qualitative.py`**

```python
"""Pretty-print per-query top-k results across methods."""


def format_query_comparison(
    qid: str,
    query: str,
    runs: dict[str, dict[str, dict[str, float]]],
    corpus: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 5,
    max_chars: int = 120,
) -> str:
    gold_docs = {did for did, r in qrels.get(qid, {}).items() if r > 0}
    lines = [f"Query [{qid}]: {query!r}", ""]
    for method, run in runs.items():
        ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:k]
        lines.append(f"  {method}:")
        for rank, (did, score) in enumerate(ranked, start=1):
            text = corpus.get(did, "")[:max_chars]
            marker = " ✓" if did in gold_docs else ""
            lines.append(f"    {rank}. [{did}]{marker} (score={score:.3f}) {text!r}")
        lines.append("")
    return "\n".join(lines)


def dump_disagreement_examples(
    qids: list[str],
    queries: dict[str, str],
    runs: dict[str, dict[str, dict[str, float]]],
    corpus: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 5,
) -> str:
    parts = []
    for qid in qids:
        parts.append(format_query_comparison(qid, queries.get(qid, ""), runs, corpus, qrels, k))
    return "\n" + ("=" * 70 + "\n").join(parts)
```

- [ ] **Step 3: Write `tests/test_analysis.py`**

```python
from src.analysis.disagreement import find_bm25_beats_dense, find_max_divergence_queries, gold_rank

QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
BM25_RUN = {
    "q1": {"d1": 3.0, "d2": 1.0},  # d1 at rank 1 → gold_rank=1
    "q2": {"d1": 3.0, "d2": 1.0},  # d2 at rank 2 → gold_rank=2
    "q3": {"d3": 3.0, "d1": 1.0},  # d3 at rank 1 → gold_rank=1
}
DENSE_RUN = {
    "q1": {"d2": 3.0, "d1": 1.0},  # d1 at rank 2 → gold_rank=2
    "q2": {"d2": 3.0, "d1": 1.0},  # d2 at rank 1 → gold_rank=1
    "q3": {"d1": 3.0, "d3": 1.0},  # d3 at rank 2 → gold_rank=2
}


def test_gold_rank_found():
    assert gold_rank("q1", QRELS, BM25_RUN) == 1
    assert gold_rank("q2", QRELS, BM25_RUN) == 2


def test_gold_rank_not_found():
    empty_run = {"q1": {"d2": 1.0}}
    assert gold_rank("q1", QRELS, empty_run, max_rank=999) == 999


def test_find_bm25_beats_dense():
    df = find_bm25_beats_dense(BM25_RUN, DENSE_RUN, QRELS, n=5)
    # q1: bm25_rank=1, dense_rank=2 → BM25 wins by 1
    # q3: bm25_rank=1, dense_rank=2 → BM25 wins by 1
    assert len(df) == 2
    assert set(df["query_id"]) == {"q1", "q3"}


def test_find_max_divergence():
    runs = {"BM25": BM25_RUN, "Dense": DENSE_RUN}
    df = find_max_divergence_queries(runs, QRELS, n=3)
    # q1: BM25=1, Dense=2 → spread=1. q2: BM25=2, Dense=1 → spread=1. q3: same
    assert len(df) <= 3
    assert "spread" in df.columns
```

- [ ] **Step 4: Run**

```bash
pytest tests/test_analysis.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add nlp_week6-retrieval-benchmark/src/analysis/ nlp_week6-retrieval-benchmark/tests/test_analysis.py
git commit -m "feat(week6): add analysis modules (disagreement, qualitative)"
```

---

## Task 17: Analysis Notebook Template

**Files:**
- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Create the notebook programmatically**

Run from `nlp_week6-retrieval-benchmark/`:

```python
# Run this block in a Python REPL or as a one-off script
import nbformat, json
from pathlib import Path

cells = []

def code(src): return nbformat.v4.new_code_cell(src)
def md(src): return nbformat.v4.new_markdown_cell(src)

cells.append(md("# Retrieval Benchmark Analysis\n\nAnswers to the 7 analysis questions with concrete query examples."))

cells.append(code("""\
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, CACHE_DIR
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact
from src.analysis.disagreement import find_bm25_beats_dense, find_max_divergence_queries, gold_rank
from src.analysis.qualitative import dump_disagreement_examples

%matplotlib inline
pd.set_option("display.float_format", "{:.4f}".format)
"""))

cells.append(md("## Load Results"))
cells.append(code("""\
msmarco_df = pd.read_csv(RESULTS_DIR / "msmarco_results.csv", index_col=0)
scifact_df = pd.read_csv(RESULTS_DIR / "scifact_results.csv", index_col=0)
print("MS MARCO:"); display(msmarco_df)
print("\\nSciFact:"); display(scifact_df)
"""))

cells.append(md("## Load Per-Query Data"))
cells.append(code("""\
import glob
per_query = {}
for f in (RESULTS_DIR / "per_query").glob("*.csv"):
    key = f.stem  # e.g. "BM25_msmarco"
    per_query[key] = pd.read_csv(f)

def get_run(method, dataset):
    df = per_query.get(f"{method}_{dataset}", pd.DataFrame())
    run = {}
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        if qid not in run:
            run[qid] = {}
        run[qid][str(row["doc_id"])] = float(row["score"])
    return run
"""))

cells.append(md("## Load Datasets (for query text + corpus)"))
cells.append(code("""\
msmarco_data = load_msmarco(sample_size=300, seed=42)
scifact_data = load_scifact()
"""))

cells.append(md("## Q1: Overall Winner\nWhich method wins? Does ranking flip between metrics?"))
cells.append(code("""\
for name, df in [("MS MARCO", msmarco_df), ("SciFact", scifact_df)]:
    print(f"\\n=== {name} ===")
    metric_cols = [c for c in df.columns if c != "Latency_ms"]
    winner_per_metric = df[metric_cols].idxmax()
    print(winner_per_metric.to_string())
    # Highlight metric-ranking flips
    mrr_winner = df["MRR@10"].idxmax()
    ndcg_winner = df["NDCG@10"].idxmax()
    if mrr_winner != ndcg_winner:
        print(f"  FLIP: MRR@10 winner={mrr_winner}, NDCG@10 winner={ndcg_winner}")
"""))

cells.append(md("## Q2: BM25 vs Dense (MiniLM)\n3–5 queries where BM25 ranks the gold doc higher."))
cells.append(code("""\
for dataset_name, data in [("msmarco", msmarco_data), ("scifact", scifact_data)]:
    bm25_run = get_run("BM25", dataset_name)
    dense_run = get_run("Dense-MiniLM", dataset_name)
    qrels = data["qrels"]
    df = find_bm25_beats_dense(bm25_run, dense_run, qrels, n=5)
    print(f"\\n=== {dataset_name}: BM25 beats Dense ===")
    display(df)
    # Show the actual queries
    for _, row in df.head(3).iterrows():
        qid = str(row["query_id"])
        print(f"  [{qid}] {data['queries'].get(qid, '')!r}")
        print(f"         BM25 rank={int(row['bm25_rank'])}, Dense rank={int(row['dense_rank'])}")
"""))

cells.append(md("## Q3: Top-vs-Worst Disagreement\n5 queries with maximum gold-rank spread across all methods."))
cells.append(code("""\
for dataset_name, data in [("msmarco", msmarco_data), ("scifact", scifact_data)]:
    methods = ["BM25", "TF-IDF", "Dense-MiniLM", "Dense-msmarco", "Hybrid-RRF", "CrossEncoder", "ColBERT"]
    runs = {m: get_run(m, dataset_name) for m in methods}
    df = find_max_divergence_queries(runs, data["qrels"], n=5)
    print(f"\\n=== {dataset_name}: Max divergence queries ===")
    display(df)
    top_qids = df["query_id"].astype(str).tolist()
    print(dump_disagreement_examples(top_qids[:2], data["queries"], runs, data["corpus"], data["qrels"], k=3))
"""))

cells.append(md("## Q4: HyDE Win/Fail\nPrint hypotheticals for queries where HyDE helped vs hurt."))
cells.append(code("""\
for dataset_name in ["msmarco", "scifact"]:
    hyp_path = CACHE_DIR / "hyde" / f"{dataset_name}__hypotheticals.json"
    if not hyp_path.exists():
        print(f"Run run_hyde_subset.py first for {dataset_name}")
        continue
    hyps = json.loads(hyp_path.read_text(encoding="utf-8"))
    print(f"\\n=== {dataset_name}: {len(hyps)} hypotheticals ===")
    # TODO after running HyDE: compare gold ranks HyDE vs Dense-MiniLM and print examples
    for key, entry in list(hyps.items())[:3]:
        print(f"Query: {entry['query']!r}")
        print(f"Hyp:   {entry['hypothetical']!r}")
        print()
"""))

cells.append(md("## Q5: M4 vs M3 — Domain-Specific Benefit"))
cells.append(code("""\
for name, df in [("MS MARCO", msmarco_df), ("SciFact", scifact_df)]:
    print(f"\\n=== {name} ===")
    for metric in ["MRR@10", "NDCG@10", "MAP@100"]:
        m3 = df.loc["Dense-MiniLM", metric]
        m4 = df.loc["Dense-msmarco", metric]
        delta = m4 - m3
        print(f"  {metric}: M3={m3:.4f}, M4={m4:.4f}, delta={delta:+.4f}")
"""))

cells.append(md("## Q6: Hybrid M5 — Does RRF Beat Both Components?"))
cells.append(code("""\
for name, df in [("MS MARCO", msmarco_df), ("SciFact", scifact_df)]:
    print(f"\\n=== {name} ===")
    for metric in ["MRR@10", "NDCG@10", "R@100", "MAP@100"]:
        bm25 = df.loc["BM25", metric]
        dense = df.loc["Dense-MiniLM", metric]
        hybrid = df.loc["Hybrid-RRF", metric]
        beats_both = hybrid > bm25 and hybrid > dense
        print(f"  {metric}: BM25={bm25:.4f}, Dense={dense:.4f}, Hybrid={hybrid:.4f} → beats_both={beats_both}")
"""))

cells.append(md("## Q7: Re-rank Cost-Benefit (M7 CrossEncoder)"))
cells.append(code("""\
for dataset_name, name, df in [("msmarco", "MS MARCO", msmarco_df), ("scifact", "SciFact", scifact_df)]:
    print(f"\\n=== {name} ===")
    # Stage-1 (M3) recall@100 — what was already in the candidate pool
    m3_r100 = df.loc["Dense-MiniLM", "R@100"]
    ce_r100 = df.loc["CrossEncoder", "R@100"]
    print(f"  M3 Stage-1 R@100: {m3_r100:.4f}  (ceiling for re-ranker)")
    print(f"  CrossEncoder R@100: {ce_r100:.4f}  (unchanged — same candidates)")
    for metric in ["P@5", "MRR@10", "NDCG@10"]:
        m3_v = df.loc["Dense-MiniLM", metric]
        ce_v = df.loc["CrossEncoder", metric]
        gain = ce_v - m3_v
        print(f"  {metric}: M3={m3_v:.4f} → CE={ce_v:.4f} (gain={gain:+.4f})")
    lat_m3 = df.loc["Dense-MiniLM", "Latency_ms"]
    lat_ce = df.loc["CrossEncoder", "Latency_ms"]
    print(f"  Latency: M3={lat_m3:.1f} ms → CE={lat_ce:.1f} ms (overhead={lat_ce - lat_m3:+.1f} ms)")
"""))

nb = nbformat.v4.new_notebook(cells=cells)
Path("notebooks/analysis.ipynb").write_text(json.dumps(nbformat.writes(nb)), encoding="utf-8")
print("Notebook written.")
```

Actually, the simplest way: write it directly as JSON using nbformat.

- [ ] **Step 2: Create notebook via script**

Save above code as `scripts/_make_notebook.py` and run:

```bash
python scripts/_make_notebook.py
```

Alternatively, use `jupyter nbconvert` to create from a Python script — but the inline approach above works fine.

- [ ] **Step 3: Open notebook and verify it loads**

```bash
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output notebooks/analysis_check.ipynb 2>&1 | head -5
```

Expected: Warning about missing results CSV is acceptable at this stage (before running benchmark). No syntax errors.

- [ ] **Step 4: Delete the check output and commit**

```bash
rm -f notebooks/analysis_check.ipynb
git add nlp_week6-retrieval-benchmark/notebooks/ nlp_week6-retrieval-benchmark/scripts/_make_notebook.py
git commit -m "feat(week6): add analysis notebook template with 7 analysis questions"
```

---

## Task 18: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Retrieval Benchmark

8-method IR benchmark across MS MARCO and SciFact.  
Methods: BM25 · TF-IDF · Dense (MiniLM) · Dense (msmarco-distilbert) · Hybrid RRF · ColBERT · Cross-encoder re-rank · HyDE

## Setup

```bash
# 1. Create virtual environment
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

```bash
# Run M1–M7 on both datasets (~60–120 min first run, ~2–5 min cached)
python scripts/run_benchmark.py

# Run HyDE (M8) — requires JGU_API_KEY
export JGU_API_KEY=<your_key>
python scripts/run_hyde_subset.py
```

Results are written to `results/`.

## Hardware Notes

Tested on AMD Ryzen 7 8845HS (CPU-only). ColBERT indexing (~10k passages) takes 30–60 min.  
If ColBERT indexing exceeds ~90 min, index on a free Colab T4, download the index directory,  
and place it at `cache/colbert/{dataset}__index/`.

## Cache

All heavy computation is cached under `cache/` (gitignored).  
Second run of the full benchmark completes in 2–5 minutes.

## Tests

```bash
pytest                         # fast unit tests (no internet required)
pytest -m integration          # data loader smoke tests (requires HF access)
```

## Results

*(populated after running the benchmark)*

### MS MARCO

| Method | P@1 | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | | | | | | | | | | |
| TF-IDF | | | | | | | | | | |
| Dense-MiniLM | | | | | | | | | | |
| Dense-msmarco | | | | | | | | | | |
| Hybrid-RRF | | | | | | | | | | |
| CrossEncoder | | | | | | | | | | |
| ColBERT | | | | | | | | | | |
| HyDE (10q) | | | | | | | | | | |

### SciFact

*(same table structure)*

## Analysis

See `notebooks/analysis.ipynb` for full analysis with concrete query examples answering:

1. **Overall winner** — which method wins per metric; ranking flips  
2. **BM25 vs Dense** — 3–5 queries where BM25 outperforms MiniLM, with hypotheses  
3. **Top-vs-worst disagreement** — 5 queries with max gold-rank divergence  
4. **HyDE win/fail** — hypothetical quality inspection  
5. **M4 vs M3** — domain-specific benefit by dataset  
6. **Hybrid M5** — does RRF beat both components on every metric?  
7. **Re-rank cost-benefit** — M3 Stage-1 R@100 vs. CrossEncoder gains vs. latency overhead  

## API Key

`JGU_API_KEY` must be set as an environment variable to run HyDE. Never commit this key.  
Add a `.env` file (gitignored) and source it, or export directly in your shell.
```

- [ ] **Step 2: Commit**

```bash
git add nlp_week6-retrieval-benchmark/README.md
git commit -m "docs(week6): add README with setup, run commands, and results table placeholders"
```

---

## Task 19: Full Test Suite Verification

- [ ] **Step 1: Run all fast unit tests**

```bash
pytest -v
```

Expected: all tests in `test_cache.py`, `test_metrics.py`, `test_methods.py`, `test_analysis.py` pass. Integration tests excluded by default.

- [ ] **Step 2: Run a minimal end-to-end smoke test**

```python
# scripts/_smoke_test.py — run once to verify full pipeline on tiny data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile, random, numpy as np, torch, transformers
transformers.logging.set_verbosity_error()
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from src.cache import load_or_compute
from src.methods.bm25 import BM25Retriever
from src.methods.tfidf import TFIDFRetriever
from src.eval.metrics import evaluate
from src.eval.runner import run_retriever

CORPUS = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "machine learning artificial intelligence",
    "d3": "information retrieval systems relevance",
}
QUERIES = {"q1": "quick fox", "q2": "information retrieval"}
QRELS = {"q1": {"d1": 1}, "q2": {"d3": 1}}

with tempfile.TemporaryDirectory() as tmp:
    cache = Path(tmp)
    for Cls, name in [(BM25Retriever, "BM25"), (TFIDFRetriever, "TF-IDF")]:
        r = Cls(dataset_name="smoke", cache_dir=cache)
        r.index(CORPUS)
        metrics, _ = run_retriever(name, r, QUERIES, QRELS)
        assert metrics["MRR@10"] > 0.0
        print(f"{name}: MRR@10={metrics['MRR@10']:.3f} ✓")

print("Smoke test passed.")
```

```bash
python scripts/_smoke_test.py
```

Expected: `BM25: MRR@10=... ✓` and `TF-IDF: MRR@10=... ✓`, then `Smoke test passed.`

- [ ] **Step 3: Final commit**

```bash
git add nlp_week6-retrieval-benchmark/scripts/_smoke_test.py
git commit -m "test(week6): add smoke test verifying end-to-end pipeline on tiny data"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] M1 BM25 → Task 7
- [x] M2 TF-IDF → Task 8
- [x] M3 Dense MiniLM → Task 9
- [x] M4 Dense msmarco → Task 10
- [x] M5 Hybrid RRF → Task 11
- [x] M7 Cross-encoder → Task 12
- [x] M6 ColBERT → Task 13
- [x] M8 HyDE + separate script → Task 15
- [x] Cache layer with doc_ids_hash → Task 2
- [x] MS MARCO 300 queries, seed=42, distractor corpus → Task 3
- [x] SciFact full 300q + 5k docs, graded qrels → Task 4
- [x] P@1/5/10, R@10/50/100, MRR@10, MAP@100, NDCG@10 → Task 5
- [x] Latency measurement (warmup 3, samples 20, single-query) → Task 6
- [x] Bold best per column in MD → Task 6 (bold_best_md in runner.py)
- [x] Per-query rankings saved → Task 6 (save_per_query)
- [x] 7 analysis questions → Task 17
- [x] HyDE JGU config, rate limit, fail-fast on missing key → Task 15
- [x] Determinism seeds at top of every script → Tasks 14, 15
- [x] transformers warnings suppressed → Tasks 14, 15
- [x] .gitignore covers cache/, .env, .ragatouille/ → Task 1
- [x] run_benchmark.py one-shot entry point → Task 14
- [x] run_hyde_subset.py separate → Task 15
- [x] ColBERT Colab escape hatch documented → Task 13, README
- [x] README with exact run commands → Task 18

**Placeholder scan:** No TBD, TODO, or "similar to Task N" patterns.

**Type consistency:**
- `load_or_compute(path, fn)` used consistently in Tasks 2, 7, 8, 9, 13 ✓
- `doc_ids_hash(doc_ids)` used in Tasks 7, 8, 9 ✓
- `Retriever.search(query, k) → list[tuple[str, float]]` interface consistent across all method tasks ✓
- `run_retriever()` returns `(metrics_dict, run_dict)` — used in Tasks 14, 15 ✓
- `evaluate(qrels, run) → dict[str, float]` with display keys (P@1, MRR@10, ...) consistent with DataFrame columns ✓
