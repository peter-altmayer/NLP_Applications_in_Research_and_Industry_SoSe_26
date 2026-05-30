# Design: OCR + Translation + Summarization Pipeline

**Date:** 2026-05-30  
**Project:** `nlp_legal_retrieval/` (repurposed for Assignment 2)  
**Course:** NLP Applications in Research and Industry

---

## Overview

Three independent NLP evaluation pipelines, each as a self-contained Python script. Each script loads 100 examples from a HuggingFace dataset using `streaming=True`, runs inference with a pre-trained model, caches results to disk, computes 2–3 metrics, and prints a summary table plus annotated failure examples.

---

## Project Structure

```
nlp_legal_retrieval/
├── part1_ocr.py
├── part2_translation.py
├── part3_summarization.py
├── utils.py
├── requirements.txt
├── README.md
└── results/
    ├── ocr_results.json
    ├── translation_results.json
    └── summarization_results.json
```

Old retrieval files (`src/`, `tests/`, `experiments/`, `notebooks/`, `data/`) are removed. `results/` and `docs/` are kept.

---

## Shared Utilities (`utils.py`)

- `save_cache(path, data)` — serializes list of result dicts to JSON
- `load_cache(path)` — deserializes JSON, returns list or `None` if file missing
- `print_table(rows, headers)` — prints formatted console table using `tabulate`

---

## Part 1 — OCR (`part1_ocr.py`)

**Dataset:** `naver-clova-ix/cord-v2`, test split, `streaming=True`, 100 examples  
**Model:** `microsoft/trocr-base-printed` via HuggingFace `TrOCRProcessor` + `VisionEncoderDecoderModel`  
**Device:** CUDA (RTX 2070 Super)

**Gold text extraction:** The `ground_truth` field is a JSON string containing a nested structure. Concatenate all `nm` (name) and `cnt` (count/price) values from `gt_parse.menu` items into a single string for comparison.

**Metrics:**
- CER (Character Error Rate) — `jiwer.cer()`
- WER (Word Error Rate) — `jiwer.wer()`
- F1 token overlap — computed inline: tokenize both strings, compute set precision/recall/F1

**Caching:** Results saved to `results/ocr_results.json` after first run. Each entry: `{index, ocr_text, gold_text, cer, wer, f1}`.

**Output:**
- 20-row summary table (index, CER, WER, F1, first 40 chars of prediction vs gold)
- Worst 3 examples by CER with full text comparison and failure annotation (character confusion, layout issue, etc.)
- Aggregate mean CER, WER, F1 printed at end

---

## Part 2 — Translation (`part2_translation.py`)

**Dataset:** `wmt/wmt19`, "de-en" config, validation split, `streaming=True`, 100 examples  
**Direction:** EN → DE  
**Model:** `Helsinki-NLP/opus-mt-en-de` (MarianMT) via HuggingFace `pipeline("translation")`  
**Device:** CUDA

**Metrics:**
- BLEU — `sacrebleu.corpus_bleu()`
- chrF — `sacrebleu.corpus_chrf()`
- BERTScore — `bert_score.score()` with `lang="de"`

**Caching:** Results saved to `results/translation_results.json`. Each entry: `{index, source_en, hypothesis_de, reference_de, bleu_sentence, chrf_sentence}`. Corpus-level BLEU/chrF/BERTScore computed at metric time from cached hypotheses + references.

**Output:**
- 20-row table (index, sentence BLEU, chrF, first 60 chars of hypothesis vs reference)
- Corpus-level aggregate scores (BLEU, chrF, BERTScore F1 mean)
- Worst 3 examples with linguistic annotation (idiom, ambiguity, morphology error, hallucination)

---

## Part 3 — Summarization (`part3_summarization.py`)

**Dataset:** `cnn_dailymail`, "3.0.0" config, test split, `streaming=True`, 100 examples  
**Model:** `facebook/bart-large-cnn` via HuggingFace `pipeline("summarization", max_length=130, min_length=30)`  
**Device:** CUDA

**Metrics:**
- ROUGE-1, ROUGE-2, ROUGE-L — `rouge_score.rouge_scorer.RougeScorer`

**Caching:** Results saved to `results/summarization_results.json`. Each entry: `{index, article_snippet, summary, reference, rouge1, rouge2, rougeL}`.

**Output:**
- 20-row table (index, ROUGE-1, ROUGE-2, ROUGE-L, first 80 chars of summary)
- Aggregate mean ROUGE scores
- 2 hallucination examples annotated by type:
  - **Intrinsic hallucination:** model generates a claim that contradicts the source article
  - **Omission hallucination:** model omits a key fact present in the source

---

## README

Sections:
1. Setup (`pip install -r requirements.txt`)
2. Run instructions (`python part1_ocr.py`, etc.)
3. Results tables (one per part, manually populated after runs)
4. Analysis per part: aggregate scores, dominant failure patterns, hallucination annotations

---

## Requirements

```
datasets>=2.19.0
transformers>=4.40.0
torch>=2.0.0
Pillow>=10.0.0
jiwer>=3.0.0
sacrebleu>=2.3.0
bert_score>=0.3.13
rouge_score>=0.1.2
tqdm>=4.66.0
tabulate>=0.9.0
pandas>=2.0.0
```

---

## Compute

All inference on CUDA (RTX 2070 Super, 8GB VRAM). Models run sequentially. Peak VRAM usage: ~1.5GB (BART-large). No quantization needed.

---

## Non-Goals

- No hyperparameter tuning or model comparison
- No web UI or interactive visualisation
- No unit tests (evaluation scripts are the artifact)
- No Jupyter notebooks
