# NLP Week 2 — Medical QA Pipeline Design

**Date:** 2026-04-28
**Author:** Peter Altmayer
**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026

---

## Overview

Build a reproducible Medical QA repository (`nlp-week2-qa/`) that satisfies the Week 2 assignment requirements:

- Task 1: Extractive QA pipeline (2 models, 25 PubMedQA examples)
- Task 2: Generative QA pipeline (2 models, 25 PubMedQA examples)
- One decoding experiment (temperature sweep on Flan-T5-large)
- Per-model result tables + merged tables committed to `results/`
- Full reproduction steps in README

---

## Repo Structure

```
nlp-week2-qa/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                      # pubmed_qa download (never pushed)
│   └── processed/
│       └── qa_pairs.csv          # 25 filtered examples (never pushed)
├── src/
│   ├── __init__.py
│   ├── dataset.py                # download, flatten, save qa_pairs.csv
│   └── model.py                  # pipeline wrappers for extractive + generative
├── experiments/
│   ├── run_extractive.py         # roberta + electra → 3 CSVs
│   ├── run_generative.py         # flan-t5-base + flan-t5-large → 3 CSVs
│   └── run_decoding.py           # flan-t5-large, temp 0.3/0.7/1.0 → 1 CSV
├── results/
│   ├── extractive_roberta.csv
│   ├── extractive_electra.csv
│   ├── extractive_all.csv
│   ├── generative_flan_t5_base.csv
│   ├── generative_flan_t5_large.csv
│   ├── generative_all.csv
│   └── decoding_results.csv
└── notebooks/
    └── exploration.ipynb
```

---

## Dataset

**Source:** `pubmed_qa`, config `pqa_labeled`, split `train`
- 1,000 expert-annotated biomedical QA pairs
- Loaded via `load_dataset("pubmed_qa", "pqa_labeled", split="train")`

**Processing (`src/dataset.py`):**
1. Take first 25 examples
2. Flatten `example["context"]["contexts"]` (list of paragraph strings) into one string joined by a single space
3. Save to `data/processed/qa_pairs.csv` with columns: `id`, `question`, `context`, `reference_answer` (mapped from `long_answer`)
4. Save raw download to `data/raw/pubmed_qa_raw.json`

Experiment scripts check if `data/processed/qa_pairs.csv` exists before calling `dataset.py` — no re-download on repeated runs.

---

## Models

### Task 1 — Extractive QA

| Model | HuggingFace ID | Size | Domain |
|---|---|---|---|
| RoBERTa | `deepset/roberta-base-squad2` | ~500 MB | General (SQuAD2) |
| ELECTRA | `deepset/electra-base-squad2` | ~110 MB | General (SQuAD2) |

Both loaded via `pipeline("question-answering", model=...)`.

### Task 2 — Generative QA

| Model | HuggingFace ID | Size | Notes |
|---|---|---|---|
| Flan-T5-base | `google/flan-t5-base` | ~250 MB | Instruction-tuned T5 |
| Flan-T5-large | `google/flan-t5-large` | ~770 MB | Same architecture, larger |

Both loaded via `pipeline("text2text-generation", model=...)`.

**Prompt template:**
```
Answer the question based on the context below.
Context: {context}
Question: {question}
Answer:
```

---

## Inference (`src/model.py`)

Two functions, both return a list of dicts:

```python
def run_extractive(model_name: str, examples: list[dict]) -> list[dict]:
    # returns [{question, context, predicted_answer, score}, ...]

def run_generative(model_name: str, examples: list[dict], **gen_kwargs) -> list[dict]:
    # returns [{question, context, predicted_answer}, ...]
```

`gen_kwargs` is passed through to the pipeline so the decoding experiment can override `temperature`, `do_sample`, etc. without touching `model.py`.

---

## Experiment Scripts

### `experiments/run_extractive.py`
1. Ensure `data/processed/qa_pairs.csv` exists
2. Run `run_extractive` for each model
3. Write `results/extractive_roberta.csv`, `results/extractive_electra.csv`
4. Merge with a `model` column → `results/extractive_all.csv`

### `experiments/run_generative.py`
1. Ensure processed CSV exists
2. Run `run_generative` for each model (default decoding: greedy via `num_beams=1`)
3. Write `results/generative_flan_t5_base.csv`, `results/generative_flan_t5_large.csv`
4. Merge → `results/generative_all.csv`

### `experiments/run_decoding.py`
1. Load processed CSV
2. Run `flan-t5-large` on first 10 examples at `temperature=0.3`, `0.7`, `1.0` (with `do_sample=True`)
3. Write `results/decoding_results.csv` with columns: `question`, `context`, `temp_0.3`, `temp_0.7`, `temp_1.0`

---

## Results & Output

All CSVs committed to `results/`. No evaluation metrics — raw outputs only.

**Result table columns:**
- Per-model: `question | context (truncated to 120 chars) | predicted_answer`
- Merged: adds `model` column
- Decoding: `question | context | temp_0.3 | temp_0.7 | temp_1.0`

README includes:
- All result tables (truncated context for readability)
- Decoding experiment table + analysis
- Failure analysis: 3+ cases with mechanism hypothesis (tokenization, span boundary ambiguity, out-of-domain vocabulary, etc.)

---

## Environment

- **Manager:** `uv`
- **Dependencies file:** `requirements.txt` (pinned)
- **Setup:**
  ```bash
  pip install uv
  uv venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  uv pip install -r requirements.txt
  ```
- **Dependencies:** `transformers`, `datasets`, `torch`, `pandas`

---

## Hardware Assumption

GPU with ≤ 6 GB VRAM. All models fit comfortably at fp32. If VRAM is tight during generative runs, add `device_map="auto"` and `torch_dtype=torch.float16` to the pipeline call.

---

## Constraints & Non-Goals

- No model training or fine-tuning — inference only
- No automated evaluation metrics (EM/F1 removed per design decision)
- No notebook-only code — notebooks are scratch space only
- `data/` never pushed to GitHub (covered by `.gitignore`)
