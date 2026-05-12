# NLP Week 3 — RAG Pipeline

**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026

## Description

End-to-end RAG evaluation pipeline comparing sparse (BM25) and dense (DPR) retrieval with two generator models across two open-domain QA datasets. The corpus is constructed from the answer strings of the first 1000 rows of each dataset. Results are evaluated with Exact Match and Token F1 across all combinations of retrieval method, K, model, and dataset.

## Model Choices and Justification

### Generator Models

**`microsoft/Phi-3-mini-4k-instruct` (3.8B)**
A modern instruction-tuned causal LM from Microsoft. Chosen because it is substantially more capable than the Flan-T5 models used in Week 2 while still fitting in fp16 on a consumer GPU (RTX 2070 Super, 8 GB VRAM). Phi-3-mini consistently punches above its weight on factual QA benchmarks relative to its parameter count, making it a good local baseline. The 4k context window is sufficient for K=10 retrieved passages.

**`Qwen/Qwen2.5-7B-Instruct` (7B)**
A strong open-weight instruction-tuned model from Alibaba's Qwen team. Chosen to pair with Phi-3-mini because it represents a different architecture family (grouped-query attention, RoPE) and roughly twice the capacity, allowing us to separate retrieval effects from model-capacity effects in the analysis. We originally targeted `mistralai/Mistral-7B-Instruct-v0.3`, but it is gated on Hugging Face; Qwen2.5-7B-Instruct is fully open, comparable in size/quality, and avoids the licence/token friction. On a free Colab T4 (16 GB VRAM) it is loaded in 4-bit NF4 (`bitsandbytes`) so it fits alongside the DPR encoders and KV cache.

Together these two models cover a meaningful capacity gap (~3.8B vs 7B) and different training recipes (Microsoft vs Alibaba), so differences in the results table are interpretable.

### Retrieval Models

**BM25 (`rank_bm25`)**
A strong non-neural baseline. Included because it requires no GPU, is deterministic, and frequently outperforms dense retrieval on short, factual queries — exactly our domain. Serves as the reference point against which neural retrievers are measured.

**`facebook/dpr-*-single-nq-base` (DPR-NQ)**
The original Dense Passage Retrieval model fine-tuned exclusively on Natural Questions. Expected to perform well on the NQ evaluation split and potentially transfer less well to TriviaQA, providing a concrete case study in dataset-specific retrieval.

**`facebook/dpr-*-multiset-base` (DPR-Multiset)**
The same DPR architecture trained on a mix of four QA datasets: NQ, TriviaQA, WebQuestions, and CuratedTREC. By comparing DPR-NQ against DPR-Multiset we can directly observe whether broader training data improves retrieval generalisation — particularly interesting since TriviaQA is in the multiset training data but not in the NQ-only model.

## Environment Setup

### Google Colab (recommended)

The full pipeline runs end-to-end on a free Colab T4 via `notebooks/colab_rag_pipeline.ipynb`. The notebook mounts Google Drive, clones this repo from GitHub, symlinks `data/` and `results/` to Drive so artifacts survive runtime disconnects, installs deps, and runs all five phases (build → phi3 × 2 datasets → qwen × 2 datasets → merge).

To use it:
1. Push your latest commits to GitHub (`git push`)
2. Open `notebooks/colab_rag_pipeline.ipynb` in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run cells top to bottom

The run-experiment cells are idempotent (skip-if-CSV-exists), so you can re-run after a disconnect and only resume what's missing.

### Local (optional fallback)

```bash
pip install uv
uv venv
# Windows:
.venv\Scripts\activate
uv pip install -r requirements.txt
```

`bitsandbytes` only matters for 4-bit loading of the 7B model — on a local GPU with ≥16 GB VRAM you can drop the `quantize_4bit` flag and run fp16.

## Experiments

```bash
# Phase 1 — build indexes (run once per dataset)
python experiments/build_index.py

# Phase 2 — generate (one CLI call per model × dataset)
python experiments/run_experiment.py --model phi3 --dataset trivia_qa
python experiments/run_experiment.py --model phi3 --dataset natural_questions
python experiments/run_experiment.py --model qwen --dataset trivia_qa --batch_size 2
python experiments/run_experiment.py --model qwen --dataset natural_questions --batch_size 2

# Phase 3 — merge CSVs, compute EM/F1, print table + failure cases
python experiments/merge_results.py
```

## Results

TODO

## Failure Analysis

TODO
