# OCR + Translation + Summarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three self-contained NLP evaluation pipelines (OCR, EN→DE translation, summarization) that each load 100 examples, run inference with a pre-trained model, cache results to disk, compute metrics, and print a summary table with failure analysis.

**Architecture:** Three flat Python scripts sharing a `utils.py` cache/table helper. Each script is independently runnable; cache means model inference only runs once. Results live in `results/*.json`.

**Tech Stack:** Python 3.10+, PyTorch (CUDA), HuggingFace `datasets` + `transformers`, `jiwer`, `sacrebleu`, `bert_score`, `rouge_score`, `tabulate`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `utils.py` | Create | JSON cache helpers + tabulate wrapper |
| `part1_ocr.py` | Create | CORD-v2 → TrOCR → CER/WER/F1 |
| `part2_translation.py` | Create | WMT19 EN→DE → MarianMT → BLEU/chrF/BERTScore |
| `part3_summarization.py` | Create | CNN/DM → BART-large → ROUGE-1/2/L |
| `requirements.txt` | Replace | All dependencies for the three tasks |
| `README.md` | Replace | Setup, run instructions, results tables, analysis |
| `results/` | Keep | Output JSONs live here |
| `tests/test_utils.py` | Create | Tests for pure functions in utils.py |
| `src/`, `tests/__init__.py`, `experiments/`, `notebooks/`, `data/` | Delete | Stale retrieval-project files |

---

## Task 1: Project Cleanup

**Files:**
- Delete: `src/`, `experiments/`, `notebooks/`, `data/`, `tests/__init__.py`, `tests/`
- Modify: `requirements.txt`
- Keep: `results/`, `docs/`

- [ ] **Step 1: Remove old project files**

```powershell
cd "G:\Meine Ablage\digital_system\20_education\22_bachelor\Uni Mainz\6. Semester\NLP Applications in Research and Industry\nlp_legal_retrieval"
Remove-Item -Recurse -Force src, experiments, notebooks, data, tests
```

- [ ] **Step 2: Ensure results/ exists with a .gitkeep**

```powershell
New-Item -ItemType Directory -Force results
New-Item -ItemType File -Force "results\.gitkeep"
```

- [ ] **Step 3: Replace requirements.txt**

Write `requirements.txt` with this exact content:

```
datasets>=2.19.0
transformers>=4.40.0
accelerate>=0.27.0
Pillow>=10.0.0
jiwer>=3.0.0
sacrebleu>=2.3.0
bert-score>=0.3.13
rouge-score>=0.1.2
tqdm>=4.66.0
tabulate>=0.9.0
pytest>=8.0.0
```

Note: PyTorch must be installed separately with the correct CUDA version:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then: `pip install -r requirements.txt`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove retrieval project files, update requirements for pipeline tasks"
```

---

## Task 2: Shared Utilities (`utils.py`)

**Files:**
- Create: `utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_utils.py`:

```python
import json
from pathlib import Path
import pytest


def test_save_and_load_roundtrip(tmp_path):
    from utils import save_cache, load_cache
    data = [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.42}]
    path = tmp_path / "cache.json"
    save_cache(path, data)
    loaded = load_cache(path)
    assert loaded == data


def test_load_cache_missing_returns_none(tmp_path):
    from utils import load_cache
    assert load_cache(tmp_path / "nonexistent.json") is None


def test_save_cache_creates_file(tmp_path):
    from utils import save_cache
    path = tmp_path / "out.json"
    save_cache(path, [{"a": 1}])
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd nlp_legal_retrieval
pytest tests/test_utils.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `utils` does not exist yet.

- [ ] **Step 3: Implement utils.py**

Create `utils.py`:

```python
import json
from pathlib import Path
from tabulate import tabulate


def save_cache(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_cache(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def print_table(rows, headers):
    print(tabulate(rows, headers=headers, tablefmt="github"))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_utils.py -v
```

Expected output:
```
tests/test_utils.py::test_save_and_load_roundtrip PASSED
tests/test_utils.py::test_load_cache_missing_returns_none PASSED
tests/test_utils.py::test_save_cache_creates_file PASSED
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add utils.py with cache helpers and table printer"
```

---

## Task 3: Part 1 — OCR Pipeline (`part1_ocr.py`)

**Files:**
- Create: `part1_ocr.py`

- [ ] **Step 1: Create part1_ocr.py**

```python
import json
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/ocr_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 100


def extract_gold_text(ground_truth_str: str) -> str:
    """Recursively collect all string values from the CORD-v2 gt_parse dict."""
    gt = json.loads(ground_truth_str)

    def collect(obj):
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                out.extend(collect(v))
            return out
        if isinstance(obj, list):
            out = []
            for item in obj:
                out.extend(collect(item))
            return out
        return []

    return " ".join(collect(gt.get("gt_parse", gt)))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_ocr(examples):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed"
    ).to(DEVICE)
    model.eval()

    results = []
    for i, ex in enumerate(tqdm(examples, desc="OCR")):
        image = ex["image"].convert("RGB")
        gold = extract_gold_text(ex["ground_truth"])

        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(DEVICE)

        with torch.no_grad():
            ids = model.generate(pixel_values)
        ocr_text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        cer = jiwer.cer(gold, ocr_text)
        wer = jiwer.wer(gold, ocr_text)
        f1 = token_f1(ocr_text, gold)

        results.append({
            "index": i,
            "ocr_text": ocr_text,
            "gold_text": gold,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "f1": round(f1, 4),
        })

    return results


def main():
    CACHE_PATH.parent.mkdir(exist_ok=True)

    results = load_cache(CACHE_PATH)
    if results is None:
        ds = load_dataset(
            "naver-clova-ix/cord-v2",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        examples = list(ds.take(N))
        results = run_ocr(examples)
        save_cache(CACHE_PATH, results)
        print(f"Cached {len(results)} results to {CACHE_PATH}")
    else:
        print(f"Loaded {len(results)} cached results from {CACHE_PATH}")

    headers = ["#", "CER", "WER", "F1", "OCR (40 chars)", "Gold (40 chars)"]
    rows = [
        [
            r["index"], r["cer"], r["wer"], r["f1"],
            r["ocr_text"][:40], r["gold_text"][:40],
        ]
        for r in results[:20]
    ]
    print("\n=== Summary Table (first 20 examples) ===")
    print_table(rows, headers)

    mean_cer = sum(r["cer"] for r in results) / len(results)
    mean_wer = sum(r["wer"] for r in results) / len(results)
    mean_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\nMean CER: {mean_cer:.4f}  |  Mean WER: {mean_wer:.4f}  |  Mean F1: {mean_f1:.4f}")

    worst = sorted(results, key=lambda x: x["cer"], reverse=True)[:3]
    print("\n=== Worst 3 Examples by CER ===")
    for r in worst:
        print(f"\n--- Example #{r['index']} | CER={r['cer']}  WER={r['wer']}  F1={r['f1']} ---")
        print(f"Gold : {r['gold_text']}")
        print(f"OCR  : {r['ocr_text']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python part1_ocr.py
```

Expected: tqdm progress bar over 100 examples, then a GitHub-flavoured markdown table, aggregate scores, and 3 annotated failure examples. `results/ocr_results.json` must exist after the run.

Run again to confirm caching works:
```bash
python part1_ocr.py
```
Expected: `Loaded 100 cached results from results/ocr_results.json` — no model inference.

- [ ] **Step 3: Commit**

```bash
git add part1_ocr.py results/ocr_results.json
git commit -m "feat(part1): add OCR pipeline with TrOCR, CER/WER/F1 metrics and failure analysis"
```

---

## Task 4: Part 2 — Translation Pipeline (`part2_translation.py`)

**Files:**
- Create: `part2_translation.py`

- [ ] **Step 1: Create part2_translation.py**

```python
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
import sacrebleu
from bert_score import score as bert_score_fn

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/translation_results.json")
DEVICE = 0 if torch.cuda.is_available() else -1
N = 100


def run_translation(examples):
    translator = hf_pipeline(
        "translation_en_to_de",
        model="Helsinki-NLP/opus-mt-en-de",
        device=DEVICE,
    )

    results = []
    for i, ex in enumerate(tqdm(examples, desc="Translation")):
        src = ex["translation"]["en"]
        ref = ex["translation"]["de"]

        hyp = translator(src, max_length=512)[0]["translation_text"]
        chrf_sent = sacrebleu.sentence_chrf(hyp, [ref]).score

        results.append({
            "index": i,
            "source_en": src,
            "hypothesis_de": hyp,
            "reference_de": ref,
            "chrf_sentence": round(chrf_sent, 2),
        })

    return results


def main():
    CACHE_PATH.parent.mkdir(exist_ok=True)

    results = load_cache(CACHE_PATH)
    if results is None:
        ds = load_dataset(
            "wmt/wmt19",
            "de-en",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        examples = list(ds.take(N))
        results = run_translation(examples)
        save_cache(CACHE_PATH, results)
        print(f"Cached {len(results)} results to {CACHE_PATH}")
    else:
        print(f"Loaded {len(results)} cached results from {CACHE_PATH}")

    hypotheses = [r["hypothesis_de"] for r in results]
    references = [r["reference_de"] for r in results]

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    _, _, F1 = bert_score_fn(hypotheses, references, lang="de", verbose=False)
    bs_f1 = F1.mean().item()

    headers = ["#", "chrF", "Hypothesis (60 chars)", "Reference (60 chars)"]
    rows = [
        [
            r["index"], r["chrf_sentence"],
            r["hypothesis_de"][:60], r["reference_de"][:60],
        ]
        for r in results[:20]
    ]
    print("\n=== Summary Table (first 20 examples) ===")
    print_table(rows, headers)

    print(f"\nCorpus BLEU: {bleu:.2f}  |  chrF: {chrf:.2f}  |  BERTScore F1: {bs_f1:.4f}")

    worst = sorted(results, key=lambda x: x["chrf_sentence"])[:3]
    print("\n=== Worst 3 Examples by chrF ===")
    for r in worst:
        print(f"\n--- Example #{r['index']} | chrF={r['chrf_sentence']} ---")
        print(f"Source : {r['source_en']}")
        print(f"Ref    : {r['reference_de']}")
        print(f"Hyp    : {r['hypothesis_de']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python part2_translation.py
```

Expected: tqdm bar over 100 sentences, then summary table, corpus-level BLEU/chrF/BERTScore, worst 3 examples. `results/translation_results.json` must exist.

Run again to confirm caching:
```bash
python part2_translation.py
```
Expected: `Loaded 100 cached results from results/translation_results.json`

- [ ] **Step 3: Commit**

```bash
git add part2_translation.py results/translation_results.json
git commit -m "feat(part2): add EN→DE translation pipeline with MarianMT, BLEU/chrF/BERTScore"
```

---

## Task 5: Part 3 — Summarization Pipeline (`part3_summarization.py`)

**Files:**
- Create: `part3_summarization.py`

- [ ] **Step 1: Create part3_summarization.py**

```python
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/summarization_results.json")
DEVICE = 0 if torch.cuda.is_available() else -1
N = 100


def run_summarization(examples):
    summarizer = hf_pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=DEVICE,
    )
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    results = []
    for i, ex in enumerate(tqdm(examples, desc="Summarization")):
        article = ex["article"]
        reference = ex["highlights"]

        # BART max input is 1024 tokens; 3000 chars is a safe character-level cutoff
        summary = summarizer(
            article[:3000], max_length=130, min_length=30, do_sample=False
        )[0]["summary_text"]

        scores = scorer.score(reference, summary)

        results.append({
            "index": i,
            "article_snippet": article[:200],
            "summary": summary,
            "reference": reference,
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        })

    return results


def main():
    CACHE_PATH.parent.mkdir(exist_ok=True)

    results = load_cache(CACHE_PATH)
    if results is None:
        ds = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split="test",
            streaming=True,
        )
        examples = list(ds.take(N))
        results = run_summarization(examples)
        save_cache(CACHE_PATH, results)
        print(f"Cached {len(results)} results to {CACHE_PATH}")
    else:
        print(f"Loaded {len(results)} cached results from {CACHE_PATH}")

    headers = ["#", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Summary (80 chars)"]
    rows = [
        [r["index"], r["rouge1"], r["rouge2"], r["rougeL"], r["summary"][:80]]
        for r in results[:20]
    ]
    print("\n=== Summary Table (first 20 examples) ===")
    print_table(rows, headers)

    mean_r1 = sum(r["rouge1"] for r in results) / len(results)
    mean_r2 = sum(r["rouge2"] for r in results) / len(results)
    mean_rL = sum(r["rougeL"] for r in results) / len(results)
    print(
        f"\nMean ROUGE-1: {mean_r1:.4f}  |  ROUGE-2: {mean_r2:.4f}  |  ROUGE-L: {mean_rL:.4f}"
    )

    # Two lowest ROUGE-1 examples — prime candidates for hallucination annotation
    worst = sorted(results, key=lambda x: x["rouge1"])[:2]
    print("\n=== 2 Lowest ROUGE-1 Examples (Hallucination Candidates) ===")
    for r in worst:
        print(
            f"\n--- Example #{r['index']} | R1={r['rouge1']}  R2={r['rouge2']}  RL={r['rougeL']} ---"
        )
        print(f"Article (200): {r['article_snippet']}")
        print(f"Reference    : {r['reference']}")
        print(f"Summary      : {r['summary']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python part3_summarization.py
```

Expected: tqdm bar over 100 articles, summary table, mean ROUGE scores, 2 lowest-ROUGE examples for hallucination analysis. `results/summarization_results.json` must exist.

Run again to confirm caching:
```bash
python part3_summarization.py
```
Expected: `Loaded 100 cached results from results/summarization_results.json`

- [ ] **Step 3: Commit**

```bash
git add part3_summarization.py results/summarization_results.json
git commit -m "feat(part3): add summarization pipeline with BART-large, ROUGE-1/2/L metrics"
```

---

## Task 6: README

**Files:**
- Replace: `README.md`

- [ ] **Step 1: Replace README.md**

Write `README.md` with the following content (fill in the `[INSERT ...]` placeholders after running all three scripts):

```markdown
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
python part1_ocr.py          # OCR: CORD-v2 receipts with TrOCR
python part2_translation.py  # Translation: WMT19 EN→DE with MarianMT
python part3_summarization.py  # Summarization: CNN/DailyMail with BART-large
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, run instructions, and results table scaffolding"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** utils.py (Task 2), part1_ocr.py (Task 3), part2_translation.py (Task 4), part3_summarization.py (Task 5), requirements.txt (Task 1), README (Task 6) — all spec sections covered
- [x] **No placeholders:** All code blocks are complete. README has `[INSERT]` markers that are intentional — filled manually after running scripts
- [x] **Type consistency:** `save_cache(path, data)` / `load_cache(path)` / `print_table(rows, headers)` — names are consistent across all tasks
- [x] **streaming=True:** Present in all three dataset loads
- [x] **trust_remote_code=True:** Present for CORD-v2 and WMT19 (required by those datasets); omitted for CNN/DM (not required)
- [x] **Cache check pattern:** Identical in all three scripts — `load_cache` → run if None → `save_cache`
- [x] **CUDA device:** `"cuda"` for direct model use (Task 3), `device=0` for HuggingFace pipelines (Tasks 4, 5) — correct API per usage
