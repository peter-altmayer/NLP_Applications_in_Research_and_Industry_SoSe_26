# 12-Step Project Framework: OCR, Translation & Summarization Pipelines

**Project:** NLP Pipeline Evaluation - OCR, Translation, Summarization
**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026
**Author:** Peter Altmayer

---

## 1. Problem Definition

**What exactly is the task?**

Three independent NLP evaluation pipelines, each assessing a pre-trained model on a real HuggingFace dataset: (1) OCR of receipt images using TrOCR, (2) English-to-German translation using MarianMT, and (3) news summarization using BART-large. The tasks are well-defined: fixed datasets, fixed models, fixed metrics, no training.

**Who is the user and what do they need?**

The primary audience is the course evaluators and the student as researcher. The goal is not a production system but a reproducible study of pre-trained model performance and failure modes on three distinct NLP tasks.

**What does success look like?**

A working, reproducible pipeline for each task that runs inference, caches results, computes standard metrics, and produces a documented failure analysis per task.

---

## 2. Prior Work & Existing Solutions

**Has this been done before?**

Yes - all three tasks are thoroughly studied. TrOCR is the state of the art for printed-text OCR; MarianMT is a well-established neural MT model family; BART-large-CNN is a benchmark summarization model. No novel contribution is made here - the goal is evaluation and analysis, not a new model.

**Are we in a research or industry setting?**

Neither - this is a course project. The deliverable is a sound comparative study: running established models, measuring them correctly, and explaining what the numbers mean.

**What are the limitations of prior work we address?**

Existing papers report aggregate scores without explaining failure modes. This project adds per-example failure analysis and identifies root causes (model mismatch, data characteristics, metric sensitivity) rather than just reporting numbers.

---

## 3. Data

**Does evaluation data exist?**

Yes - all three datasets are public on HuggingFace: `naver-clova-ix/cord-v2` (CORD-v2, receipt OCR), `wmt/wmt19 de-en` (WMT19, translation), and `abisee/cnn_dailymail 3.0.0` (CNN/DailyMail, summarization).

**Can we use it?**

Yes - all three datasets are publicly released for research use. CORD-v2 is MIT-licensed; WMT19 is a standard research benchmark; CNN/DailyMail is widely used for non-commercial NLP research.

**What does the data look like?**

- CORD-v2: scanned receipt images from Indonesian/Asian retail contexts with bounding-box annotations and structured JSON ground truth. Text is multilingual (English labels, Indonesian values), printed in various fonts and layouts.
- WMT19 de-en: sentence-aligned German/English news text from multiple European news sources. Clean, formal register.
- CNN/DailyMail: English news articles paired with bullet-point reference summaries. Articles average ~800 words; summaries are 3-5 sentences.

**How much do we have and is it enough?**

Each dataset contains thousands to hundreds of thousands of examples; we use 100 per task - sufficient for a reproducible comparative study and failure analysis, but too small for statistical significance claims.

**Are there biases in the data?**

CORD-v2 is biased toward South-East Asian retail receipts, so TrOCR (trained on Western printed text) faces a domain mismatch. WMT19 reflects European news language and may not represent colloquial or technical registers. CNN/DailyMail summaries reflect the editorial style of two specific English-language outlets.

---

## 4. Constraints

**Compute**

One GPU (RTX 2070 Super, 8 GB VRAM). All three scripts run on this hardware within a few minutes per task once models are downloaded. The 100-example limit is set partly by compute time and partly by the project scope.

**Models**

All models are used as-is off HuggingFace - no fine-tuning. This is a deliberate constraint: the project evaluates what pre-trained models can do out of the box.

**Time**

One week. Restricts the scope to evaluation pipelines rather than training or fine-tuning experiments.

**Team**

Solo project - no parallel workstreams.

---

## 5. Architecture & Techniques

**What is the approach?**

Three independent flat Python scripts sharing a `utils.py` cache layer. Each script: loads 100 streamed examples, runs inference with a pre-trained model, caches results to `results/*.json`, computes metrics, and prints a summary table and failure cases.

**OCR preprocessing**

TrOCR is a single-line model. Applying it to full receipt images gives near-zero accuracy (CER ~0.99). The fix is to use CORD-v2's `valid_line` bounding-box annotations to crop individual text-line regions before passing them to TrOCR, then concatenate the outputs. This is the architecturally correct usage.

**Translation**

MarianMT (`Helsinki-NLP/opus-mt-en-de`) is used directly via `MarianMTModel` and `MarianTokenizer`. The HuggingFace `translation` pipeline task was removed in transformers 5.x, so the model is called directly.

**Summarization**

BART-large-CNN (`facebook/bart-large-cnn`) is used directly via `BartForConditionalGeneration` and `BartTokenizer`. Input articles are truncated to 3000 characters before tokenization (BART's hard 1024-token limit). Beam search with `num_beams=4`.

**Tradeoffs**

Using pre-trained models with no fine-tuning keeps the pipeline simple and reproducible but means performance is not optimized for any of the specific datasets. Caching results to JSON means inference only runs once per dataset, making repeated analysis free.

---

## 6. Evaluation

**Metrics**

- OCR: Character Error Rate (CER), Word Error Rate (WER), token-level F1. CER and WER are case-sensitive, which inflates errors due to TrOCR's all-caps output bias; F1 is case-insensitive and is the more informative metric here.
- Translation: corpus BLEU (sacrebleu), corpus chrF, BERTScore F1. BLEU and chrF are surface-form metrics that penalize correct paraphrases; BERTScore captures semantic equivalence.
- Summarization: ROUGE-1, ROUGE-2, ROUGE-L. All are recall-oriented n-gram overlap metrics that do not penalize hallucination directly.

**Do the metrics capture what matters?**

Only partially. CER/WER miss semantically equivalent paraphrases; BLEU and chrF penalize correct idiomatic translations; ROUGE does not detect hallucination. This is why each task also has a qualitative failure analysis: the numbers alone are insufficient.

**Baseline**

There is no learned baseline to compare against - each script evaluates a single model. The comparison is between the model's scores and established literature benchmarks (e.g., BART-large-CNN typically achieves ROUGE-1 ~0.44 on the full CNN/DM test set; our 100-example subset gives 0.37, consistent with sampling variation).

---

## 7. Reliability & Safety

**Can the model hallucinate?**

Yes, especially BART (generative). Two hallucination types were found: omission (key source facts absent from summary) and intrinsic (model selects a framing that inverts the article's emphasis). MarianMT produced a word-sense hallucination translating "pulled over" as "uberfahren" (run over/killed). TrOCR hallucinated tokens (`!!!`, `PERMAS NO`) in low-contrast crop regions.

**Mitigation**

No active mitigation is implemented - this is an evaluation project, not a deployed system. The failure analysis documents the hallucination types and their causes.

**Robustness**

Not stress-tested. All inputs are clean benchmark examples. Real-world robustness (noisy images, unusual fonts, colloquial language) is not evaluated.

---

## 8. Privacy & Legal

**Sensitive data**

CORD-v2 receipt images may contain real commercial transaction data (store names, prices, item names) but no personal health or identity information. WMT19 and CNN/DailyMail are public news text with no personal data.

**Regulations**

No GDPR, HIPAA, or other regulation applies directly to this project - all data is public and non-sensitive.

**Licensing**

| Component | License |
|-----------|---------|
| CORD-v2 | MIT |
| WMT19 | Research benchmark (standard academic use) |
| CNN/DailyMail (`abisee/cnn_dailymail`) | Apache 2.0 |
| `microsoft/trocr-base-printed` | MIT |
| `Helsinki-NLP/opus-mt-en-de` | CC BY 4.0 |
| `facebook/bart-large-cnn` | MIT |

---

## 9. Fairness & Bias

**Does the system perform equally across user groups?**

Not evaluated for this project - the pipeline is offline evaluation only, not a user-facing system.

**Potential bias**

TrOCR performs poorly on CORD-v2's South-East Asian receipts partly because its training data skews toward Western printed text. This represents a real-world performance gap for non-Western use cases that would matter in a deployed OCR product.

---

## 10. Interpretability

**Do we need to explain the model's decisions?**

Not for deployment. For the course evaluation, the failure analysis serves as the explanatory layer: each failure example identifies what went wrong, at what level (character/word), and why.

**How is interpretability provided?**

Results are cached as structured JSON so individual predictions can be inspected. The failure analysis in the README shows model output alongside gold labels with explicit annotations.

---

## 11. Deployment

**Is the system deployed?**

No - this is an offline batch evaluation pipeline. There is no serving infrastructure, API, or user interface.

**Reproducibility**

The pipeline is fully reproducible: results are cached to `results/*.json` after the first run, so subsequent runs load from disk without re-running inference. Environment setup is documented in the README. All three scripts are independently runnable.

---

## 12. Impact

**Positive impact**

This project demonstrates how to evaluate pre-trained NLP models systematically across three distinct tasks, including the importance of preprocessing (OCR bounding-box cropping), the gap between surface-form and semantic metrics (chrF vs. BERTScore), and the distinction between hallucination types in summarization. The findings are directly applicable to any project that reuses these models off the shelf.

**What could go wrong at scale?**

TrOCR's all-caps output and poor performance on non-Western fonts would be a real problem in a production OCR system. MarianMT's word-sense hallucinations (e.g. "pulled over" -> "run over") could cause serious misunderstanding in legal or medical translation contexts. BART's omission hallucinations would be misleading in a summarization tool where users trust the summary to represent the article fully.
