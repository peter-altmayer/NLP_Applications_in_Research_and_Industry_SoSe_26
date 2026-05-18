# NLP Medical QA — RAG Pipeline over Clinical Text

**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026  
**Author:** Peter Altmayer  
**Seed:** 42 | **Eval subset:** 200 samples per dataset | **k (retrieval):** 5

---

## Project Overview

A RAG-based QA pipeline over biomedical text evaluated across four model variants
(V1–V4) on three datasets. The study isolates the contribution of pre-training (V1→V2),
domain fine-tuning (V2→V3), and using a domain specialist (V3→V4).

See [`docs/12_step_framework.md`](docs/12_step_framework.md) for full design analysis.

---

## Models

| Variant | Description | HuggingFace ID | Params |
|---|---|---|---|
| **V1** | 2-layer BERT, random init, trained on SQuAD-10k | *(built locally)* | ~13M |
| **V2** | Pretrained general QA, no biomedical FT | `deepset/roberta-base-squad2` | 125M |
| **V3** | `bert-base-uncased` FT on BioASQ yesno + PubMedQA | *(trained locally)* | 110M |
| **V4** | PubMedBERT + SQuAD specialist (eval-only) | `qiaojin/PubMedBERT-base-uncased-abstract-squads` | 110M |

**Fixed retriever:** `sentence-transformers/all-MiniLM-L6-v2` (dense) + BM25  
**Faithfulness model:** `cross-encoder/nli-MiniLM2-L6-H768`

---

## Datasets

| Dataset | Type | Source | Samples (eval) |
|---|---|---|---|
| **BioASQ 13b** | Factoid + Yes/No | Local JSON (`data/raw/BioASQ-training13b/`) | 200 |
| **PubMedQA** | Yes/No/Maybe | HuggingFace `pubmed_qa/pqa_labeled` | 200 |
| **SQuAD** | Extractive span | HuggingFace `squad` | 200 |

---

## Repository Structure

```
nlp_medical_qa/
├── src/
│   ├── dataset.py      # BioASQ, PubMedQA, SQuAD loaders + corpus builder
│   ├── retriever.py    # BM25Retriever, DenseRetriever, compute_retrieval_metrics
│   ├── model.py        # V1–V4 wrappers, training loop
│   ├── evaluate.py     # EM, token-F1, ROUGE-L, BERTScore, faithfulness (NLI)
│   └── privacy.py      # Regex PHI masker
├── experiments/
│   ├── build_index.py      # Build BM25 + FAISS indexes
│   ├── run_experiment.py   # Main pipeline CLI
│   └── merge_results.py    # Aggregate CSVs → summary table + failure analysis
├── data/
│   ├── raw/BioASQ-training13b/training13b.json   # ← put BioASQ here
│   └── processed/          # indexes + checkpoints (git-ignored)
├── results/                # per-run CSVs (git-ignored)
└── docs/12_step_framework.md
```

---

## Setup

```bash
pip install uv
uv venv && .venv\Scripts\activate       # Windows
# or: source .venv/bin/activate         # Linux/macOS / Colab
uv pip install -r requirements.txt
```

> **Colab note:** replace `faiss-cpu` with `faiss-gpu` in requirements.txt for GPU acceleration.

---

## Reproduction Steps

Run all commands from the **project root** (`nlp_medical_qa/`).

### Phase 0 — Build indexes (once)

```bash
python experiments/build_index.py --datasets bioasq pubmedqa squad --retrieval bm25 dense
```

### Phase 1 — Train V1 and V3 (once)

```bash
python experiments/run_experiment.py --model v1 --fine-tune
python experiments/run_experiment.py --model v3 --fine-tune
```

V1 trains for ~20 min on a T4 GPU; V3 for ~30 min.

### Phase 2 — Evaluate all variants

```bash
# Quick single run
python experiments/run_experiment.py --model v2 --dataset bioasq --retrieval dense --k 5

# Run all 4 models × 3 datasets × 2 retrievers automatically
python experiments/run_experiment.py --all --n 200 --k 5
```

Add `--skip-faithfulness` to skip the NLI step and save ~40% time.

### Phase 3 — Merge and print results

```bash
python experiments/merge_results.py
```

Prints a formatted table (EM, F1, ROUGE-L, BERTScore, Faithfulness, P@k, R@k)
and a failure analysis section; also writes `results/summary.csv`.

---

## Settings (fixed for reproducibility)

| Setting | Value |
|---|---|
| Random seed | 42 |
| Eval samples per dataset | 200 |
| Retrieved passages (k) | 5 |
| Max context length | 384 tokens |
| V1 architecture | L=2, H=256, A=4, intermediate=1024 |
| V1 training data | SQuAD train, 10 000 examples, 3 epochs, lr=3e-4 |
| V3 fine-tune data | BioASQ yesno + PubMedQA train, 4 epochs, lr=2e-5 |
| Dense embedder | `all-MiniLM-L6-v2` |
| Faithfulness model | `cross-encoder/nli-MiniLM2-L6-H768` |

---

## Results

> *To be filled in after experiments are run.*

## Key Findings

> *To be filled in after analysis.*

## Failure Analysis

> *To be filled in after analysis.*

---

## Licenses

| Component | License |
|---|---|
| BioASQ data | Research use only (bioasq.org) |
| PubMedQA | MIT |
| SQuAD | CC BY-SA 4.0 |
| `deepset/roberta-base-squad2` | CC BY 4.0 |
| `qiaojin/PubMedBERT-base-uncased-abstract-squads` | MIT |
| `all-MiniLM-L6-v2` | Apache 2.0 |
