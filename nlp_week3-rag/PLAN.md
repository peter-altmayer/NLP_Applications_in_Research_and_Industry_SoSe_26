# RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a modular RAG evaluation pipeline comparing BM25 and two DPR retrievers with Phi-3-mini (local) and Mistral-7B (Colab) across TriviaQA and Natural Questions.

**Architecture:** Two-phase — `build_index.py` pre-computes and saves all retrieval indexes once; `run_experiment.py --model <key> --dataset <key>` loads saved indexes and generates answers for all retrieval × K combinations; `merge_results.py` joins all CSVs and prints the results table. Each `src/` module has one responsibility and a uniform interface.

**Tech Stack:** Python 3.10+, `datasets`, `transformers ≥ 4.40`, `torch`, `rank_bm25`, `numpy`, `pandas`, `accelerate`, `pytest`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/evaluate.py` | rewrite | EM/F1 metrics, results pivot table |
| `src/dataset.py` | rewrite | dataset loading, corpus extraction |
| `src/retriever.py` | rewrite | BM25 + DPR build/save/load/retrieve |
| `src/model.py` | rewrite | generator loading, prompt building, batched inference |
| `experiments/build_index.py` | rewrite | orchestrate indexing for all datasets |
| `experiments/run_experiment.py` | create | CLI: run one (model, dataset) pair |
| `experiments/merge_results.py` | create | merge CSVs, compute metrics, print table + failure cases |
| `tests/test_evaluate.py` | create | unit tests for metrics |
| `tests/test_dataset.py` | create | unit tests for corpus extraction |
| `tests/test_model.py` | create | unit tests for prompt building |
| `tests/test_retriever.py` | create | unit tests for BM25 + DPR smoke tests |
| `requirements.txt` | rewrite | pinned dependencies |

---

### Task 1: Environment Setup

**Files:**
- Modify: `nlp_week3-rag/requirements.txt`
- Modify: `nlp_week3-rag/.gitignore`

- [ ] **Step 1: Write requirements.txt**

```
datasets>=2.19
transformers>=4.40
torch>=2.2
rank_bm25>=0.2.2
numpy>=1.26
pandas>=2.2
accelerate>=0.30
pytest>=8.0
```

- [ ] **Step 2: Create and activate venv**

```bash
cd nlp_week3-rag
pip install uv
uv venv
# Windows:
.venv\Scripts\activate
uv pip install -r requirements.txt
```

- [ ] **Step 3: Verify torch + CUDA**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: version string followed by `True`.

- [ ] **Step 4: Update .gitignore to ignore result CSVs**

Ensure `nlp_week3-rag/.gitignore` contains:

```
.venv/
data/raw/
data/processed/
__pycache__/
*.py[cod]
.env
results/*.csv
```

The `results/*.csv` line prevents accidental commits of large CSV files. Commit `all_results.csv` manually when done.

- [ ] **Step 5: Create tests directory**

```bash
mkdir nlp_week3-rag/tests
touch nlp_week3-rag/tests/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add nlp_week3-rag/requirements.txt nlp_week3-rag/.gitignore nlp_week3-rag/tests/
git commit -m "feat(week3): add requirements, update gitignore, create tests dir"
```

---

### Task 2: src/evaluate.py — Metrics

**Files:**
- Rewrite: `nlp_week3-rag/src/evaluate.py`
- Create: `nlp_week3-rag/tests/test_evaluate.py`

- [ ] **Step 1: Write tests/test_evaluate.py**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import normalize, exact_match, token_f1


def test_normalize_lowercases():
    assert normalize("Albert Einstein") == "albert einstein"


def test_normalize_strips_articles():
    assert normalize("the cat") == "cat"
    assert normalize("a dog") == "dog"
    assert normalize("an apple") == "apple"


def test_normalize_strips_punctuation():
    assert normalize("Hello, World!") == "hello world"


def test_exact_match_hit():
    assert exact_match("Paris", ["Paris", "paris"]) is True


def test_exact_match_miss():
    assert exact_match("London", ["Paris"]) is False


def test_exact_match_normalizes_aliases():
    assert exact_match("Mt. Everest", ["Mount Everest", "Mt Everest"]) is True


def test_token_f1_perfect():
    assert token_f1("the quick brown fox", ["the quick brown fox"]) == 1.0


def test_token_f1_partial():
    f1 = token_f1("quick fox", ["the quick brown fox"])
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap():
    assert token_f1("banana", ["apple orange"]) == 0.0


def test_token_f1_takes_best_alias():
    f1_multi = token_f1("New York", ["New York City", "NYC"])
    f1_single = token_f1("New York", ["NYC"])
    assert f1_multi >= f1_single
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd nlp_week3-rag
pytest tests/test_evaluate.py -v
```

Expected: all fail with `ImportError`.

- [ ] **Step 3: Write src/evaluate.py**

```python
import re
import string

import pandas as pd


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def exact_match(pred: str, gold_list: list[str]) -> bool:
    pred_norm = normalize(pred)
    return any(normalize(g) == pred_norm for g in gold_list)


def token_f1(pred: str, gold_list: list[str]) -> float:
    pred_tokens = normalize(pred).split()
    best_f1 = 0.0
    for gold in gold_list:
        gold_tokens = normalize(gold).split()
        if not pred_tokens and not gold_tokens:
            f1 = 1.0
        elif not pred_tokens or not gold_tokens:
            f1 = 0.0
        else:
            common = {
                t: min(pred_tokens.count(t), gold_tokens.count(t))
                for t in set(pred_tokens) & set(gold_tokens)
            }
            num_common = sum(common.values())
            if num_common == 0:
                f1 = 0.0
            else:
                precision = num_common / len(pred_tokens)
                recall = num_common / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["retrieval_method", "k", "model", "dataset"])
        .agg(em=("em", "mean"), f1=("f1", "mean"))
        .round(3)
    )
    agg["em/f1"] = agg["em"].map("{:.3f}".format) + " / " + agg["f1"].map("{:.3f}".format)
    return agg["em/f1"].unstack(["model", "dataset"])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add nlp_week3-rag/src/evaluate.py nlp_week3-rag/tests/test_evaluate.py
git commit -m "feat(week3): implement evaluate.py with EM/F1 and results table"
```

---

### Task 3: src/dataset.py — Data Loading

**Files:**
- Rewrite: `nlp_week3-rag/src/dataset.py`
- Create: `nlp_week3-rag/tests/test_dataset.py`

- [ ] **Step 1: Inspect dataset schemas before writing loader**

```bash
python -c "
from datasets import load_dataset
tqa = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='train')
print('TriviaQA keys:', list(tqa[0].keys()))
print('TriviaQA answer field:', tqa[0]['answer'])

nq = load_dataset('sentence-transformers/natural-questions', split='train')
print('NQ keys:', list(nq[0].keys()))
print('NQ first row:', dict(list(nq[0].items())[:4]))
"
```

Expected TriviaQA answer: `{'value': '...', 'aliases': [...], 'normalized_value': '...', 'normalized_aliases': [...]}`

Expected NQ: `{'query': '...', 'positive': ['...'], ...}` — note the exact key names and whether `positive` is a string or list. Adjust `_load_natural_questions` below if they differ.

- [ ] **Step 2: Write tests/test_dataset.py**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import extract_corpus


def test_extract_corpus_flattens_all_answers():
    records = [
        {"question": "Q1", "answers": ["A1", "A1b"]},
        {"question": "Q2", "answers": ["A2"]},
    ]
    corpus = extract_corpus(records)
    assert "A1" in corpus
    assert "A1b" in corpus
    assert "A2" in corpus
    assert len(corpus) == 3


def test_extract_corpus_deduplicates():
    records = [
        {"question": "Q1", "answers": ["Paris"]},
        {"question": "Q2", "answers": ["Paris"]},
    ]
    corpus = extract_corpus(records)
    assert corpus.count("Paris") == 1


def test_extract_corpus_preserves_order():
    records = [
        {"question": "Q1", "answers": ["B", "A"]},
        {"question": "Q2", "answers": ["C"]},
    ]
    corpus = extract_corpus(records)
    assert corpus == ["B", "A", "C"]
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest tests/test_dataset.py -v
```

Expected: all fail with `ImportError`.

- [ ] **Step 4: Write src/dataset.py**

Adjust `_load_natural_questions` field names if Step 1 showed different keys than `query` / `positive`.

```python
from datasets import load_dataset as hf_load


def load_dataset_records(hf_path: str, split: str, n: int = 1000) -> list[dict]:
    ds = hf_load(hf_path, split=split)
    subset = ds.select(range(min(n, len(ds))))
    if "trivia_qa" in hf_path:
        return _load_trivia_qa(subset)
    if "natural-questions" in hf_path:
        return _load_natural_questions(subset)
    raise ValueError(f"Unknown dataset path: {hf_path}")


def _load_trivia_qa(subset) -> list[dict]:
    records = []
    for row in subset:
        answers = [row["answer"]["value"]] + list(row["answer"].get("aliases", []))
        records.append({
            "question": row["question"],
            "answers": list(dict.fromkeys(answers)),  # dedupe, preserve order
        })
    return records


def _load_natural_questions(subset) -> list[dict]:
    # Schema from Step 1 inspection: {"query": str, "positive": str | list[str], ...}
    records = []
    for row in subset:
        positive = row["positive"]
        answers = positive if isinstance(positive, list) else [positive]
        records.append({
            "question": row["query"],
            "answers": list(dict.fromkeys(answers)),
        })
    return records


def extract_corpus(records: list[dict]) -> list[str]:
    seen: set[str] = set()
    corpus: list[str] = []
    for rec in records:
        for ans in rec["answers"]:
            if ans not in seen:
                seen.add(ans)
                corpus.append(ans)
    return corpus
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_dataset.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Smoke-test real dataset loading**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.dataset import load_dataset_records, extract_corpus
recs = load_dataset_records('mandarjoshi/trivia_qa', 'rc.nocontext', n=5)
print('TriviaQA sample:', recs[0])
corp = extract_corpus(recs)
print(f'{len(corp)} corpus docs from 5 records')

recs2 = load_dataset_records('sentence-transformers/natural-questions', 'train', n=5)
print('NQ sample:', recs2[0])
"
```

- [ ] **Step 7: Commit**

```bash
git add nlp_week3-rag/src/dataset.py nlp_week3-rag/tests/test_dataset.py
git commit -m "feat(week3): implement dataset.py with TriviaQA and NQ loaders"
```

---

### Task 4: src/retriever.py — BM25

**Files:**
- Rewrite: `nlp_week3-rag/src/retriever.py`
- Create: `nlp_week3-rag/tests/test_retriever.py`

- [ ] **Step 1: Write tests/test_retriever.py (BM25 tests only)**

```python
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import build_bm25, load_bm25, retrieve_bm25, save_bm25

CORPUS = [
    "The capital of France is Paris.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital city of Japan.",
    "The Eiffel Tower is located in Paris, France.",
]


def test_retrieve_bm25_returns_k_docs():
    index = build_bm25(CORPUS)
    results = retrieve_bm25(index, CORPUS, "capital of France", k=2)
    assert len(results) == 2


def test_retrieve_bm25_relevance():
    index = build_bm25(CORPUS)
    results = retrieve_bm25(index, CORPUS, "Eiffel Tower Paris", k=1)
    assert "Eiffel Tower" in results[0]


def test_retrieve_bm25_returns_strings():
    index = build_bm25(CORPUS)
    results = retrieve_bm25(index, CORPUS, "Tokyo Japan", k=2)
    assert all(isinstance(r, str) for r in results)


def test_bm25_save_load_roundtrip():
    index = build_bm25(CORPUS)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = Path(f.name)
    save_bm25(index, path)
    loaded = load_bm25(path)
    original = index.get_scores("Paris".split())
    reloaded = loaded.get_scores("Paris".split())
    assert list(original) == list(reloaded)
    path.unlink()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_retriever.py -v
```

Expected: all fail with `ImportError`.

- [ ] **Step 3: Write src/retriever.py (BM25 section)**

```python
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(corpus: list[str]) -> BM25Okapi:
    return BM25Okapi([doc.lower().split() for doc in corpus])


def save_bm25(index: BM25Okapi, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_bm25(path: Path) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_bm25(index: BM25Okapi, corpus: list[str], query: str, k: int) -> list[str]:
    scores = index.get_scores(query.lower().split())
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i] for i in top_k]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Expected: all 4 BM25 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add nlp_week3-rag/src/retriever.py nlp_week3-rag/tests/test_retriever.py
git commit -m "feat(week3): implement BM25 retriever"
```

---

### Task 5: src/retriever.py — DPR

**Files:**
- Modify: `nlp_week3-rag/src/retriever.py` (add DPR functions)
- Modify: `nlp_week3-rag/tests/test_retriever.py` (add DPR smoke tests)

- [ ] **Step 1: Add DPR smoke tests to tests/test_retriever.py**

Append to the existing file:

```python
def test_dpr_embeddings_shape():
    pytest.importorskip("transformers")
    from src.retriever import build_dpr_embeddings
    corpus = ["hello world", "foo bar baz"]
    embeddings = build_dpr_embeddings(
        corpus,
        ctx_encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
        device="cpu",
    )
    assert embeddings.shape == (2, 768)


def test_retrieve_dpr_returns_k_docs():
    pytest.importorskip("transformers")
    import pytest
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    from src.retriever import build_dpr_embeddings, retrieve_dpr

    corpus = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital city of Japan.",
    ]
    embeddings = build_dpr_embeddings(
        corpus, "facebook/dpr-ctx_encoder-single-nq-base", device="cpu"
    )
    q_enc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    results = retrieve_dpr(embeddings, corpus, q_enc, q_tok, "capital France", k=2, device="cpu")
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
```

Also add `import pytest` at the top of the test file.

- [ ] **Step 2: Append DPR functions to src/retriever.py**

```python
import torch


# ── DPR ───────────────────────────────────────────────────────────────────────

def build_dpr_embeddings(
    corpus: list[str],
    ctx_encoder_name: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder_name)
    model = DPRContextEncoder.from_pretrained(ctx_encoder_name).to(device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            out = model(**enc)
            all_embeddings.append(out.pooler_output.cpu().numpy())
    return np.vstack(all_embeddings)


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    np.save(path, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def retrieve_dpr(
    embeddings: np.ndarray,
    corpus: list[str],
    q_encoder,
    q_tokenizer,
    query: str,
    k: int,
    device: str = "cuda",
) -> list[str]:
    q_encoder.eval()
    with torch.no_grad():
        enc = q_tokenizer(
            query, return_tensors="pt", truncation=True, max_length=64
        ).to(device)
        q_vec = q_encoder(**enc).pooler_output.cpu().numpy()  # (1, 768)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-8
    scores = (embeddings @ q_vec.T).squeeze() / norms
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i] for i in top_k]
```

- [ ] **Step 3: Confirm BM25 tests still pass**

```bash
pytest tests/test_retriever.py -k "not dpr" -v
```

Expected: all 4 BM25 tests PASS.

- [ ] **Step 4: Run DPR shape smoke test (downloads ~400 MB first run)**

```bash
pytest tests/test_retriever.py::test_dpr_embeddings_shape -v
```

Expected: PASS with shape `(2, 768)`.

- [ ] **Step 5: Commit**

```bash
git add nlp_week3-rag/src/retriever.py nlp_week3-rag/tests/test_retriever.py
git commit -m "feat(week3): add DPR embeddings and retrieval to retriever.py"
```

---

### Task 6: src/model.py — Generator

**Files:**
- Rewrite: `nlp_week3-rag/src/model.py`
- Create: `nlp_week3-rag/tests/test_model.py`

- [ ] **Step 1: Write tests/test_model.py**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_prompt


def test_build_prompt_no_context_structure():
    messages = build_prompt("Who invented the telephone?", [])
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_prompt_no_context_content():
    messages = build_prompt("Who invented the telephone?", [])
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Who invented the telephone?" in user_msg
    assert "Context" not in user_msg


def test_build_prompt_with_context_includes_docs():
    docs = ["Alexander Graham Bell invented the telephone.", "Bell was born in 1847."]
    messages = build_prompt("Who invented the telephone?", docs)
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Alexander Graham Bell" in user_msg
    assert "Bell was born in 1847" in user_msg


def test_build_prompt_with_context_numbered():
    docs = ["doc one", "doc two"]
    messages = build_prompt("Q?", docs)
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "1." in user_msg
    assert "2." in user_msg


def test_build_prompt_returns_list_of_role_content_dicts():
    messages = build_prompt("Q?", ["doc"])
    assert isinstance(messages, list)
    for m in messages:
        assert set(m.keys()) == {"role", "content"}
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_model.py -v
```

Expected: all fail with `ImportError`.

- [ ] **Step 3: Write src/model.py**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_generator(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt(question: str, context_docs: list[str]) -> list[dict]:
    if context_docs:
        context_text = "\n".join(f"{i + 1}. {doc}" for i, doc in enumerate(context_docs))
        system = (
            "Answer the question using only the provided context. "
            "Be concise — a few words or one short phrase. "
            "If the context does not contain the answer, say 'I don't know'."
        )
        user = f"Context:\n{context_text}\n\nQuestion: {question}"
    else:
        system = "Answer the question concisely — a few words or one short phrase."
        user = f"Question: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answers(
    model,
    tokenizer,
    records: list[dict],
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> list[str]:
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    predictions = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_prompt(r["question"], r["context_docs"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in batch
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            pred = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            predictions.append(pred)

    return predictions
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: all 5 tests PASS (no model downloaded — only `build_prompt` is tested).

- [ ] **Step 5: Smoke-test model loading (downloads ~7 GB first time)**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.model import load_generator, generate_answers
model, tok = load_generator('microsoft/Phi-3-mini-4k-instruct')
recs = [{'question': 'What is the capital of France?', 'context_docs': ['Paris is the capital of France.']}]
preds = generate_answers(model, tok, recs, batch_size=1)
print('Prediction:', preds[0])
"
```

Expected output: `"Paris"` or `"The capital of France is Paris."` or similar.

**VRAM note:** If you hit OOM during experiments (DPR encoders + Phi-3 together), load DPR question encoders on CPU — add `device="cpu"` when loading them in `run_experiment.py`. Encoding 1000 queries on CPU takes ~3 minutes, which is acceptable.

- [ ] **Step 6: Commit**

```bash
git add nlp_week3-rag/src/model.py nlp_week3-rag/tests/test_model.py
git commit -m "feat(week3): implement model.py with generator loading and batched inference"
```

---

### Task 7: experiments/build_index.py

**Files:**
- Rewrite: `nlp_week3-rag/experiments/build_index.py`

- [ ] **Step 1: Write experiments/build_index.py**

```python
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import extract_corpus, load_dataset_records
from src.retriever import build_bm25, build_dpr_embeddings, save_bm25, save_embeddings

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc.nocontext"),
    "natural_questions": ("sentence-transformers/natural-questions", "train"),
}

DPR_CTX_ENCODERS = {
    "dpr_nq": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr_multiset": "facebook/dpr-ctx_encoder-multiset-base",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds_key, (hf_path, split) in DATASETS.items():
        print(f"\n=== {ds_key} ===")
        records = load_dataset_records(hf_path, split, n=1000)
        corpus = extract_corpus(records)
        print(f"  {len(records)} records | {len(corpus)} unique corpus docs")

        corpus_path = PROCESSED / f"{ds_key}_corpus.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump({"records": records, "corpus": corpus}, f, ensure_ascii=False)
        print(f"  Corpus saved → {corpus_path.name}")

        bm25 = build_bm25(corpus)
        save_bm25(bm25, PROCESSED / f"{ds_key}_bm25.pkl")
        print("  BM25 index saved")

        for enc_key, enc_name in DPR_CTX_ENCODERS.items():
            out_path = PROCESSED / f"{ds_key}_{enc_key}.npy"
            if out_path.exists():
                print(f"  {enc_key}: already exists — skipping")
                continue
            print(f"  Building {enc_key} embeddings ({enc_name})...")
            embeddings = build_dpr_embeddings(corpus, enc_name, device=device)
            save_embeddings(embeddings, out_path)
            print(f"  {enc_key}: saved shape {embeddings.shape} → {out_path.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run build_index.py**

```bash
cd nlp_week3-rag
python experiments/build_index.py
```

Expected output:
```
Device: cuda
=== trivia_qa ===
  1000 records | ~1200 unique corpus docs
  Corpus saved → trivia_qa_corpus.json
  BM25 index saved
  Building dpr_nq embeddings...
  dpr_nq: saved shape (1200, 768) → trivia_qa_dpr_nq.npy
  ...
=== natural_questions ===
  ...
```

Total time: ~5–10 minutes on GPU.

- [ ] **Step 3: Verify all 8 files were created**

```bash
ls nlp_week3-rag/data/processed/
```

Expected: `trivia_qa_corpus.json`, `trivia_qa_bm25.pkl`, `trivia_qa_dpr_nq.npy`, `trivia_qa_dpr_multiset.npy`, and the same 4 for `natural_questions`.

- [ ] **Step 4: Commit**

```bash
git add nlp_week3-rag/experiments/build_index.py
git commit -m "feat(week3): implement build_index.py — corpus and index building"
```

---

### Task 8: experiments/run_experiment.py

**Files:**
- Create: `nlp_week3-rag/experiments/run_experiment.py`

- [ ] **Step 1: Write experiments/run_experiment.py**

```python
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import generate_answers, load_generator
from src.retriever import load_bm25, load_embeddings, retrieve_bm25, retrieve_dpr

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

GENERATOR_MODELS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

DPR_Q_ENCODERS = {
    "dpr_nq": "facebook/dpr-question_encoder-single-nq-base",
    "dpr_multiset": "facebook/dpr-question_encoder-multiset-base",
}

K_NONZERO = [1, 5, 10]


def _save(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {len(rows)} rows → {path.name}")


def _make_row(args, retrieval_method, k, rec, pred):
    return {
        "dataset": args.dataset,
        "model": args.model,
        "retrieval_method": retrieval_method,
        "k": k,
        "question": rec["question"],
        "gold_answers": json.dumps(rec["gold_answers"]),
        "retrieved_docs": json.dumps(rec["context_docs"]),
        "prediction": pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(GENERATOR_MODELS))
    parser.add_argument("--dataset", required=True, choices=["trivia_qa", "natural_questions"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset}")

    with open(PROCESSED / f"{args.dataset}_corpus.json", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    corpus = data["corpus"]

    bm25 = load_bm25(PROCESSED / f"{args.dataset}_bm25.pkl")
    embeddings = {k: load_embeddings(PROCESSED / f"{args.dataset}_{k}.npy") for k in DPR_Q_ENCODERS}

    # Load DPR question encoders on CPU to preserve VRAM for the generator
    dpr_encoders = {}
    for enc_key, enc_name in DPR_Q_ENCODERS.items():
        print(f"Loading {enc_key} question encoder (cpu)...")
        tok = DPRQuestionEncoderTokenizer.from_pretrained(enc_name)
        enc = DPRQuestionEncoder.from_pretrained(enc_name)
        enc.eval()
        dpr_encoders[enc_key] = (enc, tok)

    print(f"Loading generator: {GENERATOR_MODELS[args.model]}")
    gen_model, gen_tokenizer = load_generator(GENERATOR_MODELS[args.model])

    # ── K=0: no context, run once ──────────────────────────────────────────
    k0_path = RESULTS / f"{args.model}_{args.dataset}_none_k0.csv"
    if k0_path.exists():
        print("K=0: skipping (already exists)")
    else:
        print("\nK=0 (no context)...")
        gen_records = [
            {"question": r["question"], "gold_answers": r["answers"], "context_docs": []}
            for r in records
        ]
        preds = generate_answers(gen_model, gen_tokenizer, gen_records, batch_size=args.batch_size)
        _save([_make_row(args, "none", 0, r, p) for r, p in zip(gen_records, preds)], k0_path)

    # ── K > 0 ──────────────────────────────────────────────────────────────
    for retrieval_key in ["bm25", "dpr_nq", "dpr_multiset"]:
        for k in K_NONZERO:
            out_path = RESULTS / f"{args.model}_{args.dataset}_{retrieval_key}_k{k}.csv"
            if out_path.exists():
                print(f"Skipping {out_path.name}")
                continue

            print(f"\n{retrieval_key} k={k}...")
            gen_records = []
            for rec in records:
                if retrieval_key == "bm25":
                    docs = retrieve_bm25(bm25, corpus, rec["question"], k)
                else:
                    enc, tok = dpr_encoders[retrieval_key]
                    docs = retrieve_dpr(embeddings[retrieval_key], corpus, enc, tok, rec["question"], k, device="cpu")
                gen_records.append({
                    "question": rec["question"],
                    "gold_answers": rec["answers"],
                    "context_docs": docs,
                })

            preds = generate_answers(gen_model, gen_tokenizer, gen_records, batch_size=args.batch_size)
            _save([_make_row(args, retrieval_key, k, r, p) for r, p in zip(gen_records, preds)], out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with one run (watch GPU with nvidia-smi)**

In a separate terminal: `nvidia-smi -l 2`

Then run:
```bash
python experiments/run_experiment.py --model phi3 --dataset trivia_qa --batch_size 4
```

Confirm CSVs appear in `results/`. The first run produces 10 files (`none_k0`, `bm25_k1/5/10`, `dpr_nq_k1/5/10`, `dpr_multiset_k1/5/10`).

Expected time: ~2–3 hours for 1000 questions across all retrieval × K combinations.

- [ ] **Step 3: Commit**

```bash
git add nlp_week3-rag/experiments/run_experiment.py
git commit -m "feat(week3): implement run_experiment.py CLI"
```

---

### Task 9: experiments/merge_results.py

**Files:**
- Create: `nlp_week3-rag/experiments/merge_results.py`

- [ ] **Step 1: Write experiments/merge_results.py**

```python
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import build_results_table, exact_match, token_f1

RESULTS = Path(__file__).parent.parent / "results"


def main():
    csvs = sorted(p for p in RESULTS.glob("*.csv") if p.name != "all_results.csv")
    if not csvs:
        print("No CSVs found in results/ — run run_experiment.py first")
        return

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csvs)} files")

    df["em"] = df.apply(
        lambda r: int(exact_match(str(r["prediction"]), json.loads(r["gold_answers"]))), axis=1
    )
    df["f1"] = df.apply(
        lambda r: token_f1(str(r["prediction"]), json.loads(r["gold_answers"])), axis=1
    )

    df.to_csv(RESULTS / "all_results.csv", index=False)
    print("Saved all_results.csv\n")

    table = build_results_table(df)
    print("=== Results (mean EM / mean F1) ===")
    print(table.to_string())

    print("\n=== Top 10 Failure Cases (EM=0, lowest F1) ===")
    failures = df[df["em"] == 0].nsmallest(10, "f1")
    for _, row in failures.iterrows():
        gold = json.loads(row["gold_answers"])
        retrieved = json.loads(row["retrieved_docs"])
        print(
            f"\n[{row['dataset']} | {row['model']} | {row['retrieval_method']} | K={row['k']}]"
        )
        print(f"  Q:    {row['question']}")
        print(f"  Gold: {gold[:3]}")
        print(f"  Pred: {row['prediction']}")
        if retrieved:
            print(f"  Ret:  {retrieved[0][:120]}")
        print(f"  EM={row['em']}  F1={row['f1']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run after at least one experiment CSV exists**

```bash
python experiments/merge_results.py
```

Expected: prints results table and failure cases. If only local results are available so far, the Mistral columns will be missing — that is fine.

- [ ] **Step 3: Commit**

```bash
git add nlp_week3-rag/experiments/merge_results.py
git commit -m "feat(week3): implement merge_results.py with table and failure analysis"
```

---

### Task 10: Full Experiment Run + README

**Files:**
- Modify: `nlp_week3-rag/README.md`

- [ ] **Step 1: Run all local Phi-3 experiments**

```bash
python experiments/run_experiment.py --model phi3 --dataset trivia_qa --batch_size 8
python experiments/run_experiment.py --model phi3 --dataset natural_questions --batch_size 8
```

Estimated time: ~2–3 hours total on RTX 2070 Super. Safe to run overnight.

- [ ] **Step 2: Run Colab Mistral-7B experiments**

On Google Colab:
1. Clone or upload `nlp_week3-rag/`
2. `!pip install uv && !uv pip install -r requirements.txt`
3. Upload `data/processed/` from local (or re-run `build_index.py` on Colab)
4. `!python experiments/run_experiment.py --model mistral --dataset trivia_qa --batch_size 4`
5. `!python experiments/run_experiment.py --model mistral --dataset natural_questions --batch_size 4`
6. Download all `results/mistral_*.csv` into local `nlp_week3-rag/results/`

- [ ] **Step 3: Merge all results**

```bash
python experiments/merge_results.py
```

- [ ] **Step 4: Update README.md with Experiments, Results, and Failure Analysis sections**

Replace the three TODO sections in README.md with:

**Experiments section:**
```
## Experiments

Build indexes (run once):
    python experiments/build_index.py

Run local generator (Phi-3-mini):
    python experiments/run_experiment.py --model phi3 --dataset trivia_qa
    python experiments/run_experiment.py --model phi3 --dataset natural_questions

Run Colab generator (Mistral-7B) — same commands with --model mistral

Merge and evaluate:
    python experiments/merge_results.py
```

**Results and Failure Analysis sections:** paste the output of `merge_results.py` directly.

- [ ] **Step 5: Final commit**

```bash
git add nlp_week3-rag/README.md
git add -f nlp_week3-rag/results/all_results.csv
git commit -m "feat(week3): full experiment results, results table, failure analysis"
```

---

## Running All Unit Tests

After Tasks 1–6 are complete:

```bash
cd nlp_week3-rag
pytest tests/ -v -k "not dpr"
```

Expected: all tests for `evaluate`, `dataset`, and `model` PASS. Skip DPR tests during development to avoid downloading weights every run — run them once explicitly.
