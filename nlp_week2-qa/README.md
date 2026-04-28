# NLP Week 2 — Medical QA Pipelines

This project builds two question-answering pipelines — one extractive, one generative — on the PubMedQA benchmark dataset. Medical records contain answers buried in unstructured clinical text; a clinician querying a patient's history or a researcher scanning literature should not have to read hundreds of pages. We compare two pretrained models per task to understand how general-domain models behave on biomedical text, where domain-specific vocabulary, abbreviations, and reasoning patterns routinely differ from general web corpora.

---

## Repository Structure

```
nlp_week2-qa/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # pubmed_qa download — never pushed
│   └── processed/        # qa_pairs.csv (25 examples) — never pushed
├── src/
│   ├── dataset.py        # download, flatten, save qa_pairs.csv
│   └── model.py          # pipeline wrappers (extractive + generative)
├── experiments/
│   ├── run_extractive.py # RoBERTa + ELECTRA on 25 examples
│   ├── run_generative.py # Flan-T5-base + Flan-T5-large on 25 examples
│   └── run_decoding.py   # temperature sweep on Flan-T5-large
├── results/              # all CSVs committed
└── notebooks/
    └── exploration.ipynb # scratch only
```

---

## Dataset

**PubMedQA** (`pqa_labeled` config, 1 000 expert-annotated examples)

- `context` — concatenated PubMed abstract sections (BACKGROUND, METHODS, RESULTS, CONCLUSIONS)
- `question` — a biomedical research question
- `reference_answer` — the conclusion sentence from the abstract (verbatim span; used as ground truth)

Loaded via HuggingFace `datasets`, 25 examples used across all experiments.

---

## Models

### Task 1 — Extractive QA

| Model | HuggingFace ID | Size | Why |
|---|---|---|---|
| RoBERTa-base | `deepset/roberta-base-squad2` | ~500 MB | Strong general-domain extractive baseline, fine-tuned on SQuAD2 |
| ELECTRA-base | `deepset/electra-base-squad2` | ~110 MB | Compact discriminator model; frequently outperforms BERT/RoBERTa on span extraction |

Both are general-domain models (pretrained on web text, fine-tuned on SQuAD). Their performance on medical text reveals the domain gap that motivates domain adaptation.

### Task 2 — Generative QA

| Model | HuggingFace ID | Size | Why |
|---|---|---|---|
| Flan-T5-base | `google/flan-t5-base` | ~250 MB | Instruction-tuned encoder-decoder; follows natural-language prompts reliably |
| Flan-T5-large | `google/flan-t5-large` | ~770 MB | Same architecture, ~3× parameters; allows a direct scale comparison |

Flan-T5 is chosen over decoder-only models because it is instruction-tuned (responds predictably to "Answer the question based on the context" prompts), fits comfortably in ≤ 6 GB VRAM, and the base/large pair provides a built-in ablation over model scale.

---

## Environment Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
uv venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

uv pip install -r requirements.txt

# After installing, pin exact versions:
uv pip freeze > requirements.txt
```

> **GPU note (Windows):** if `torch` installs the CPU-only build, install the CUDA variant manually:
> `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## Reproducing Experiments

All scripts are run from the `nlp_week2-qa/` root directory.
The first run downloads PubMedQA automatically; subsequent runs reuse the cached CSV.

```bash
# Task 1 — Extractive QA
python experiments/run_extractive.py
# writes: results/extractive_roberta.csv, extractive_electra.csv, extractive_all.csv

# Task 2 — Generative QA
python experiments/run_generative.py
# writes: results/generative_flan_t5_base.csv, generative_flan_t5_large.csv, generative_all.csv

# Decoding experiment
python experiments/run_decoding.py
# writes: results/decoding_results.csv
```

Expected runtime on a GPU with ≤ 6 GB VRAM: ~5 min extractive, ~15 min generative, ~10 min decoding.

---

## Results

> Run the experiment scripts first, then paste the CSV output into the tables below.
> Contexts are truncated to 120 characters for readability; full text is in the CSVs.

### Task 1 — Extractive QA

#### RoBERTa (`deepset/roberta-base-squad2`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| *run experiments/run_extractive.py to populate* | | |

#### ELECTRA (`deepset/electra-base-squad2`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| *run experiments/run_extractive.py to populate* | | |

#### All models (merged)

| Model | Question | Context (truncated) | Predicted Answer |
|---|---|---|---|
| *run experiments/run_extractive.py to populate* | | | |

---

### Task 2 — Generative QA

#### Flan-T5-base (`google/flan-t5-base`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| *run experiments/run_generative.py to populate* | | |

#### Flan-T5-large (`google/flan-t5-large`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| *run experiments/run_generative.py to populate* | | |

#### All models (merged)

| Model | Question | Context (truncated) | Predicted Answer |
|---|---|---|---|
| *run experiments/run_generative.py to populate* | | | |

---

## Decoding Experiment

Model: `google/flan-t5-large`, 10 examples, `do_sample=True`, temperatures: 0.3 / 0.7 / 1.0

| Question | temp=0.3 | temp=0.7 | temp=1.0 |
|---|---|---|---|
| *run experiments/run_decoding.py to populate* | | | |

**Analysis (to be filled after running):**

At `temperature=0.3` the distribution over vocabulary is sharpened, pushing the model toward its highest-confidence token at each step. Outputs are expected to be shorter, more repetitive, and clinically conservative. At `temperature=1.0` the original logit distribution is unchanged, producing more varied completions — sometimes more informative, sometimes less coherent. The middle value `0.7` is typically a practical sweet spot: enough determinism to stay on topic, enough diversity to avoid degenerate repetition. Medical QA generally prefers low temperature because consistency and accuracy outweigh creativity.

---

## Failure Analysis

Below are three predicted failure modes with mechanistic hypotheses. Update with actual observed failures after running the experiments.

### Failure 1 — Span boundary ambiguity (extractive)

**Hypothesis:** The model predicts a span that contains the correct information but with incorrect start or end boundaries — e.g., extracting "treatment with metformin was associated" instead of "metformin". This is a known failure mode of span-prediction heads: the start and end logits are predicted independently, so the model can select a start token that is locally plausible but inconsistent with the selected end token. Medical sentences tend to be long with nested clauses, which increases the probability of boundary misalignment.

### Failure 2 — Out-of-domain vocabulary (extractive + generative)

**Hypothesis:** Clinical abbreviations and compound noun phrases (e.g., "LVEF", "HbA1c", "T2DM") may be split into subword tokens that carry no meaningful signal in a model pretrained on web text. RoBERTa and ELECTRA were pretrained on BookCorpus and Wikipedia — corpora where such terms are rare. The tokenizer fragments them (e.g., "HbA1c" → ["H", "##b", "##A", "##1", "##c"]), and the embeddings for those fragments are undertrained, causing the attention mechanism to assign low relevance to the correct answer span even when it is present verbatim.

### Failure 3 — Answer requires cross-sentence synthesis (extractive)

**Hypothesis:** Several PubMedQA questions ask for conclusions that are not stated in a single sentence but must be inferred by combining findings from the RESULTS and CONCLUSIONS sections. Extractive models are constrained to return a verbatim span — they cannot synthesise information across sentences. When no single span constitutes a complete answer, the model will fall back on a superficially relevant phrase (often a high-frequency noun phrase near the question's keywords) that is factually incomplete.

---

*Peter Altmayer — NLP Applications in Research and Industry, Uni Mainz, SoSe 2026*
