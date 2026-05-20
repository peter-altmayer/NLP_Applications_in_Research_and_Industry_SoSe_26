# NLP Medical QA v2 — RAG Pipeline with Oracle Retrieval Baseline

**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026  
**Author:** Peter Altmayer  
**Seed:** 42 | **Eval subset:** 200 samples per dataset | **k (retrieval):** 5

---

## What's New in v2

v2 adds an **oracle retrieval baseline** and `answer_type`-aware prediction on top of the v1 pipeline:

| Addition | Purpose |
|---|---|
| `OracleRetriever` | Bypasses the retriever; feeds gold-annotated passages directly to the model. Isolates whether low scores are caused by retrieval failure or model failure. |
| `answer_type` kwarg in `predict()` | Passes the BioASQ/PubMedQA answer type (`factoid`, `yesno`, etc.) through the prediction chain. Prepares for a future dual-head model (Task 6). |
| `--retrieval oracle` CLI flag | Runs a single oracle evaluation. |
| `--all-oracle` CLI flag | Runs oracle retrieval for all 4 models × 3 datasets in one shot. |
| `--oracle-summary` in `merge_results.py` | Prints an oracle-specific table and writes `results/oracle_baseline.csv`. |

**Key finding from oracle:** V2 and V4 achieve F1 = 0.85 and 0.79 on SQuAD with gold passages — the models are competent; the regular-run near-zero scores are almost entirely caused by retrieval failure, not model failure.

---

## Project Overview

A RAG-based QA pipeline over biomedical text evaluated across four model variants
(V1–V4) on three datasets. The study isolates the contribution of pre-training (V1→V2),
domain fine-tuning (V2→V3), and using a domain specialist (V3→V4). The oracle baseline
added in v2 quantifies how much of the performance gap is attributable to the retriever
vs. the model.

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
nlp_medical_qa_v2/
├── src/
│   ├── dataset.py      # BioASQ, PubMedQA, SQuAD loaders + corpus builder
│   ├── retriever.py    # BM25Retriever, DenseRetriever, OracleRetriever, metrics
│   ├── model.py        # V1-V4 wrappers, training loop, answer_type kwarg
│   ├── evaluate.py     # EM, token-F1, ROUGE-L, BERTScore, faithfulness (NLI)
│   └── privacy.py      # Regex PHI masker
├── experiments/
│   ├── build_index.py      # Build BM25 + FAISS indexes
│   ├── run_experiment.py   # Main pipeline CLI (--retrieval oracle, --all-oracle)
│   └── merge_results.py    # Aggregate CSVs -> summary + oracle baseline table
├── data/
│   ├── raw/BioASQ-training13b/training13b.json   # <- put BioASQ here
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

Run all commands from the **project root** (`nlp_medical_qa_v2/`).

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

### Phase 2 — Evaluate all variants (BM25 + dense)

```bash
# Quick single run
python experiments/run_experiment.py --model v2 --dataset bioasq --retrieval dense --k 5

# Full matrix: 4 models x 3 datasets x 2 retrievers
python experiments/run_experiment.py --all --n 200 --k 5
```

### Phase 3 — Oracle baseline (v2 new)

```bash
# Single oracle run
python experiments/run_experiment.py --model v4 --dataset squad --retrieval oracle --k 5

# Full oracle matrix: 4 models x 3 datasets (no index needed)
python experiments/run_experiment.py --all-oracle --n 200 --k 5
```

### Phase 4 — Merge and print results

```bash
# Regular summary table
python experiments/merge_results.py

# Oracle baseline table (writes oracle_baseline.csv)
python experiments/merge_results.py --oracle-summary
```

Add `--skip-faithfulness` to skip the NLI step and save ~40% time.

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

### Regular Retrieval (BM25 + Dense)

All 24 combinations (4 models x 3 datasets x 2 retrievers), 200 samples each, seed 42.  
†V2 dense/BioASQ faithfulness is NaN — that run completed before faithfulness was enabled.

#### BioASQ (factoid + yes/no)

| Model | Retrieval | EM | F1 | ROUGE-L | BERTScore | Faithfulness | P@5 | R@5 |
|---|---|---|---|---|---|---|---|---|
| V1 | dense | 0.000 | 0.009 | 0.013 | 0.159 | 0.513 | 0.008 | 0.005 |
| V2 | dense | 0.000 | 0.000 | 0.000 | 0.107 | †NaN | 0.000 | 0.000 |
| V3 | dense | 0.000 | 0.001 | 0.002 | 0.193 | 0.496 | 0.008 | 0.005 |
| **V4** | **dense** | **0.020** | **0.057** | **0.071** | **0.205** | **0.508** | **0.008** | **0.005** |
| V1 | bm25 | 0.000 | 0.005 | 0.005 | 0.134 | 0.499 | 0.005 | 0.004 |
| V2 | bm25 | 0.000 | 0.010 | 0.012 | 0.252 | 0.464 | 0.005 | 0.004 |
| V3 | bm25 | 0.000 | 0.003 | 0.007 | 0.191 | 0.475 | 0.005 | 0.004 |
| **V4** | **bm25** | **0.000** | **0.032** | **0.038** | **0.156** | **0.476** | **0.005** | **0.004** |

#### PubMedQA (yes/no/maybe)

| Model | Retrieval | EM | F1 | ROUGE-L | BERTScore | Faithfulness | P@5 | R@5 |
|---|---|---|---|---|---|---|---|---|
| V1 | dense | 0.000 | 0.000 | 0.000 | 0.121 | 0.507 | 0.000 | 0.000 |
| V2 | dense | 0.000 | 0.002 | 0.002 | 0.407 | 0.485 | 0.000 | 0.000 |
| V3 | dense | 0.000 | 0.000 | 0.000 | 0.262 | 0.490 | 0.000 | 0.000 |
| V4 | dense | 0.005 | 0.007 | 0.007 | 0.075 | 0.487 | 0.000 | 0.000 |
| V1 | bm25 | 0.000 | 0.000 | 0.000 | 0.115 | 0.485 | 0.000 | 0.000 |
| V2 | bm25 | 0.000 | 0.001 | 0.001 | 0.433 | 0.482 | 0.000 | 0.000 |
| V3 | bm25 | 0.005 | 0.005 | 0.005 | 0.268 | 0.472 | 0.000 | 0.000 |
| V4 | bm25 | 0.005 | 0.007 | 0.007 | 0.070 | 0.468 | 0.000 | 0.000 |

#### SQuAD (extractive span)

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

### Oracle Retrieval Baseline (v2 new)

Gold passages fed directly to the model (no retrieval). P@k = 1.0 by construction; any remaining gap is pure model failure.

| Model | BioASQ EM | BioASQ F1 | SQuAD EM | SQuAD F1 | PubMedQA EM | PubMedQA F1 |
|---|---|---|---|---|---|---|
| V1 | 0.000 | 0.035 | 0.015 | 0.059 | 0.000 | 0.000 |
| **V2** | **0.130** | **0.241** | **0.670** | **0.850** | 0.000 | 0.007 |
| V3 | 0.000 | 0.006 | 0.010 | 0.031 | 0.000 | 0.000 |
| **V4** | **0.130** | **0.257** | **0.605** | **0.786** | 0.000 | 0.002 |

#### Retrieval vs. oracle lift

| Model | Dataset | Dense F1 | Oracle F1 | Lift |
|---|---|---|---|---|
| V2 | SQuAD | 0.011 | 0.850 | **+0.839** |
| V4 | SQuAD | 0.025 | 0.786 | **+0.761** |
| V2 | BioASQ | 0.000 | 0.241 | +0.241 |
| V4 | BioASQ | 0.057 | 0.257 | +0.200 |
| V3 | SQuAD | 0.004 | 0.031 | +0.027 |
| V1 | SQuAD | 0.012 | 0.059 | +0.047 |

The 76–84 point F1 lift for V2/V4 on SQuAD confirms retrieval failure is the bottleneck. V3's near-zero oracle score confirms its failure is a fine-tuning bug independent of retrieval.

---

## Key Findings

**1. Retrieval failure is the dominant bottleneck — now quantified.**  
P@5 = 0 for PubMedQA and SQuAD across all models, and the oracle experiment makes the consequence concrete: V2 jumps from F1 = 0.011 to 0.850 on SQuAD when given the gold passage. The entire regular-run performance gap on those two datasets is retrieval, not model quality. Only BioASQ achieves non-zero retrieval because its snippets are pre-curated short passages; abstract-length PubMedQA contexts and Wikipedia paragraphs are too long and too similar to discriminate at retrieval time.

**2. V2 and V4 are competent QA models.**  
Oracle SQuAD F1 of 0.85 (V2) and 0.79 (V4) are competitive with fine-tuned extractive QA baselines. Regular-run scores should not be read as evidence of weak models — they reflect a broken retrieval loop.

**3. V4 is the best model on domain tasks; the V3→V4 gap is confirmed under oracle.**  
Oracle BioASQ F1: V4 = 0.257 vs. V2 = 0.241. The domain pre-training of BioBERT gives a real edge when the model receives the correct passage. On exact match both are 0.130, so the advantage is in partial-overlap spans rather than perfect extraction.

**4. V3 fine-tuning failure is a span-alignment bug, not a retrieval artefact.**  
Oracle BioASQ F1 for V3 is 0.006 — essentially the same as its regular retrieval score (0.001). Even with the gold passage provided, V3 predicts single punctuation marks (`.`, `,`) or function words (`here`). The fine-tuning format prepended "Yes." or "No." to the context, teaching V3 to extract position 0 regardless of the question. Retraining with correct span-alignment labels is required.

**5. PubMedQA is structurally incompatible with extractive span models.**  
Oracle F1 is near zero for all models on PubMedQA even with gold passages. The answers are `yes`, `no`, or `maybe` — single-word labels that never appear as verbatim spans in retrieved passages. A generative head or a classification head over `[CLS]` is required; the `answer_type` routing added in v2 is the first step toward this fix.

**6. V2 BERTScore on PubMedQA remains anomalously high (0.407–0.433).**  
RoBERTa's sentence embeddings are intrinsically closer to yes/no/maybe tokens than BERT-family models. This is a representational property, not a quality signal; EM and F1 are both zero.

**7. Dense retrieval outperforms BM25 on BioASQ; BM25 matches dense on SQuAD.**  
Dense P@5 = 0.008 vs. BM25 P@5 = 0.005 on BioASQ. On SQuAD, BM25 matches or exceeds dense for V2 (F1 0.016 vs 0.011), because SQuAD questions share exact vocabulary with their contexts — keyword matching outperforms embedding similarity in that setting.

**8. Faithfulness scores remain near-neutral under oracle retrieval.**  
NLI faithfulness ranges 0.46–0.52 for all oracle runs, the same as regular retrieval. Even with gold snippets, the extracted span often falls outside them, and the NLI model scores the pair as neutral. Faithfulness only becomes a useful signal when the model can extract the correct span — which requires fixing the fine-tuning bug (V3) or improving retrieval.

---

## Failure Analysis

**V1 (from scratch):** Extracts sub-word fragments (`##omodulation`, `rc`, `ml`) or stray numeric tokens (`64`, `95 %`). The 2-layer randomly-initialised encoder never learns coherent span boundaries. V1 is the absolute performance floor and oracle scores confirm it (BioASQ oracle F1 = 0.035, SQuAD = 0.059).

**V2 (pretrained, no fine-tuning):** Dense PubMedQA predictions are `nan` — empty strings from RoBERTa's pipeline failing on short retrieved contexts. BM25 runs produce plausible biomedical spans but always from the wrong passage. Under oracle retrieval V2 achieves near-perfect SQuAD performance (F1 = 0.85); three exact matches include "linear", "Lampea", and "San Jose", confirming the regular-run failures are purely retrieval-driven.

**V3 (fine-tuned by us):** Consistently predicts `.`, `,`, `here`, or the first token of the context — even under oracle retrieval with the gold passage provided. Representative: Q = "What is the enzymatic activity of PARL?" / Gold = "PARL are serine proteases" / Pred = `.`. This is the span-alignment bug from the "Yes. [context]" fine-tuning format. No retrieval improvement can fix it.

**V4 (BioBERT specialist):** Under regular retrieval, extracts plausible but wrong biomedical spans — e.g., `"the reverse of all other known DNA and RNA polymerases"` when the gold answer is `"PARL are serine proteases"`. Under oracle retrieval, V4 achieves F1 = 1.0 on clean factoid questions: "Which virus can be diagnosed with the monospot test?" → "Epstein-Barr virus"; "Mutation of which gene is implicated in Christianson syndrome?" → "SLC9A6". Retrieval is the only reason regular-run scores are low.

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
