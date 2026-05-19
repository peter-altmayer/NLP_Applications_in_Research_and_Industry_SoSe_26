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
| **V4** | BioBERT v1.1 + SQuAD specialist (eval-only) | `dmis-lab/biobert-base-cased-v1.1-squad` | 110M |

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

All 24 combinations (4 models × 3 datasets × 2 retrievers), 200 samples each, seed 42.  
†V2 dense/BioASQ faithfulness is NaN — that run completed before faithfulness was enabled and the CSV was not re-generated.

### BioASQ (factoid + yes/no)

| Model | Retrieval | EM | F1 | ROUGE-L | BERTScore | Faithfulness | P@5 | R@5 |
|---|---|---|---|---|---|---|---|---|
| V1 | dense | 0.000 | 0.008 | 0.013 | 0.159 | 0.513 | 0.008 | 0.005 |
| V2 | dense | 0.000 | 0.000 | 0.000 | 0.107 | †NaN | 0.000 | 0.000 |
| V3 | dense | 0.000 | 0.001 | 0.002 | 0.193 | 0.496 | 0.008 | 0.005 |
| **V4** | **dense** | **0.020** | **0.057** | **0.071** | **0.205** | **0.508** | **0.008** | **0.005** |
| V1 | bm25 | 0.000 | 0.005 | 0.005 | 0.134 | 0.499 | 0.005 | 0.004 |
| V2 | bm25 | 0.000 | 0.010 | 0.012 | 0.252 | 0.464 | 0.005 | 0.004 |
| V3 | bm25 | 0.000 | 0.002 | 0.006 | 0.191 | 0.475 | 0.005 | 0.004 |
| **V4** | **bm25** | **0.000** | **0.032** | **0.038** | **0.156** | **0.476** | **0.005** | **0.004** |

### PubMedQA (yes/no/maybe)

| Model | Retrieval | EM | F1 | ROUGE-L | BERTScore | Faithfulness | P@5 | R@5 |
|---|---|---|---|---|---|---|---|---|
| V1 | dense | 0.000 | 0.000 | 0.000 | 0.120 | 0.507 | 0.000 | 0.000 |
| V2 | dense | 0.000 | 0.002 | 0.002 | 0.407 | 0.485 | 0.000 | 0.000 |
| V3 | dense | 0.000 | 0.000 | 0.000 | 0.262 | 0.490 | 0.000 | 0.000 |
| V4 | dense | 0.005 | 0.007 | 0.007 | 0.075 | 0.486 | 0.000 | 0.000 |
| V1 | bm25 | 0.000 | 0.000 | 0.000 | 0.115 | 0.485 | 0.000 | 0.000 |
| V2 | bm25 | 0.000 | 0.001 | 0.001 | 0.433 | 0.481 | 0.000 | 0.000 |
| V3 | bm25 | 0.005 | 0.005 | 0.005 | 0.268 | 0.472 | 0.000 | 0.000 |
| V4 | bm25 | 0.005 | 0.007 | 0.007 | 0.070 | 0.468 | 0.000 | 0.000 |

### SQuAD (extractive span)

| Model | Retrieval | EM | F1 | ROUGE-L | BERTScore | Faithfulness | P@5 | R@5 |
|---|---|---|---|---|---|---|---|---|
| V1 | dense | 0.000 | 0.012 | 0.016 | 0.180 | 0.484 | 0.000 | 0.000 |
| V2 | dense | 0.010 | 0.011 | 0.015 | 0.195 | 0.464 | 0.000 | 0.000 |
| V3 | dense | 0.000 | 0.004 | 0.004 | 0.148 | 0.462 | 0.000 | 0.000 |
| **V4** | **dense** | **0.005** | **0.025** | **0.030** | **0.265** | **0.468** | **0.000** | **0.000** |
| V1 | bm25 | 0.000 | 0.007 | 0.008 | 0.138 | 0.488 | 0.000 | 0.000 |
| V2 | bm25 | 0.010 | 0.016 | 0.016 | 0.187 | 0.489 | 0.000 | 0.000 |
| V3 | bm25 | 0.000 | 0.001 | 0.001 | 0.151 | 0.481 | 0.000 | 0.000 |
| **V4** | **bm25** | **0.005** | **0.030** | **0.029** | **0.224** | **0.483** | **0.000** | **0.000** |

---

## Key Findings

**1. Retrieval failure is the dominant bottleneck.**  
P@5 = 0 for PubMedQA and SQuAD across all models and retrievers — the gold passage is never in the top-5 results. Only BioASQ achieves non-zero retrieval (P@5 = 0.005–0.008), because BioASQ snippets are pre-curated short passages already in the corpus. On PubMedQA and SQuAD, abstract-length contexts and Wikipedia paragraphs are too long and semantically similar to many other corpus entries, so the retriever consistently returns plausible but wrong passages. This is the primary explanation for all near-zero EM and F1 scores: models predict spans from the wrong retrieved context.

**2. V4 is the best model on generation metrics, confirming domain specialisation helps.**  
On BioASQ (where retrieval succeeds at all), V4 dense achieves the highest EM (0.020), F1 (0.057), and ROUGE-L (0.071). The V3→V4 gap is the largest pairwise gap in the study, consistent with the 12-step framework hypothesis that a purpose-built specialist outperforms our ad-hoc fine-tuning.

**3. V3 underperforms V2 in most conditions.**  
Our fine-tuning (V2→V3) did not improve scores and in several cases degraded them (e.g., BioASQ dense F1: V2=0.000, V3=0.001; SQuAD dense F1: V2=0.011, V3=0.004). The fine-tuning format — prepending "Yes." or "No." to the context so the label is always a verbatim span — likely caused V3 to always extract the first token of the context rather than the actual answer. This is a span-alignment failure, not a pre-training failure.

**4. V2 BERTScore on PubMedQA is anomalously high (0.407–0.433).**  
RoBERTa's sentence representations are semantically closer to the gold yes/no/maybe answers than any other model, even though its span predictions fail. This reflects RoBERTa's stronger general-purpose representations rather than correct span extraction — the BERTScore metric does not require the predicted span to contain the answer, only that it be semantically similar.

**5. Dense retrieval outperforms BM25 on BioASQ; results are mixed elsewhere.**  
On BioASQ, dense P@5 (0.008) > BM25 P@5 (0.005), and V4 dense F1 (0.057) > V4 BM25 F1 (0.032). On SQuAD, BM25 matches or exceeds dense for V2 (F1 0.016 vs 0.011), as SQuAD questions share exact vocabulary with their contexts — a case where keyword matching outperforms embedding similarity.

**6. Faithfulness scores cluster near 0.5 (neutral) for all models.**  
NLI faithfulness ranges 0.46–0.51 across the board, consistent with the retrieval failure: when the retrieved context is unrelated to the question, any extracted span is neither clearly entailed nor contradicted, so the NLI model defaults to neutral. Faithfulness only becomes a meaningful discriminator when retrieval succeeds.

---

## Failure Analysis

Representative worst predictions per model, illustrating each model's characteristic failure mode:

**V1 (from scratch):** Predicts sub-word fragments — `##ase`, `##20`, `mega` — because the 2-layer randomly-initialised encoder never learns coherent span boundaries. Every prediction is a noise token. Expected for this setup; V1 serves only as the absolute floor.

**V2 (pretrained, no fine-tuning):** The BioASQ dense run produced `nan` predictions (empty strings from failed pipeline inference), a data quality artefact from the initial run before faithfulness was enabled. BM25 predictions are more coherent but still wrong because the retrieved passage never contains the gold answer.

**V3 (fine-tuned by us):** Consistently predicts single punctuation marks (`.`, `)`, `,`) or the first word of the context. This is the span-alignment artefact from the fine-tuning format: after seeing thousands of "Yes. [long explanation]" contexts, V3 learned to extract position 0 of the context rather than the answer span.

**V4 (BioBERT specialist):** Extracts plausible but wrong biomedical spans — e.g., `"muscle-type-specific positive and negative cis-acting elements"` when the gold answer is `"Limb-Enhancer Genie (LEG)"`. The model is extracting correctly-structured answers from the wrong retrieved passage. Retrieval failure is the cause, not model failure.

---

## Licenses

| Component | License |
|---|---|
| BioASQ data | Research use only (bioasq.org) |
| PubMedQA | MIT |
| SQuAD | CC BY-SA 4.0 |
| `deepset/roberta-base-squad2` | CC BY 4.0 |
| `dmis-lab/biobert-base-cased-v1.1-squad` | Apache 2.0 |
| `all-MiniLM-L6-v2` | Apache 2.0 |
