# NLP Pipeline Tasks — OCR, Translation, Summarization

Three NLP evaluation pipelines on real HuggingFace datasets.

## Setup

Install PyTorch with CUDA first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Running

Each script is self-contained. Results are cached to `results/` after the first run.

```bash
python part1_ocr.py           # OCR: CORD-v2 receipts with TrOCR
python part2_translation.py   # Translation: WMT19 EN→DE with MarianMT
python part3_summarization.py # Summarization: CNN/DailyMail with BART-large
```

---

## Part 1 — OCR Results

**Model:** `microsoft/trocr-base-printed`
**Dataset:** `naver-clova-ix/cord-v2` (100 test examples)
**Metrics:** CER (lower = better), WER (lower = better), F1 token overlap (higher = better)

| Metric | Mean Score |
|--------|-----------|
| CER | [INSERT] |
| WER | [INSERT] |
| F1 | [INSERT] |

### Failure Analysis

[INSERT: describe 3 worst examples — character confusion, layout issues, etc.]

---

## Part 2 — Translation Results

**Model:** `Helsinki-NLP/opus-mt-en-de`
**Dataset:** `wmt/wmt19 de-en` validation (100 examples, EN→DE)
**Metrics:** BLEU (higher = better), chrF (higher = better), BERTScore F1 (higher = better)

| Metric | Score |
|--------|-------|
| BLEU | [INSERT] |
| chrF | [INSERT] |
| BERTScore F1 | [INSERT] |

### Failure Analysis

[INSERT: describe 3 worst examples — idiom, ambiguity, morphology, hallucination]

---

## Part 3 — Summarization Results

**Model:** `facebook/bart-large-cnn`
**Dataset:** `cnn_dailymail 3.0.0` test (100 examples)
**Metrics:** ROUGE-1, ROUGE-2, ROUGE-L (all higher = better)

| Metric | Mean Score |
|--------|-----------|
| ROUGE-1 | [INSERT] |
| ROUGE-2 | [INSERT] |
| ROUGE-L | [INSERT] |

### Hallucination Analysis

**Intrinsic hallucination (Example #[INSERT]):**
[INSERT: source claim vs what the model generated that contradicts it]

**Omission hallucination (Example #[INSERT]):**
[INSERT: key fact in source that the summary omitted]
