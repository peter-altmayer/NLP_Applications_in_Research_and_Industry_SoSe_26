#!/usr/bin/env python
"""Run HyDE on ~10 queries per dataset. Requires JGU_API_KEY env var.

Usage:
    export JGU_API_KEY=<your_key>
    python scripts/run_hyde_subset.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import transformers

transformers.logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import CACHE_DIR, MODELS, RESULTS_DIR, SEED, HYDE_SAMPLE_SIZE, MSMARCO_SAMPLE_SIZE
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact
from src.eval.runner import run_retriever, save_per_query
from src.methods.dense import DenseRetriever
from src.methods.hyde import HyDERetriever

import pandas as pd

RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "msmarco": load_msmarco(sample_size=MSMARCO_SAMPLE_SIZE, seed=SEED),
    "scifact": load_scifact(),
}

for dataset_name, data in DATASETS.items():
    all_qids = list(data["queries"].keys())
    random.seed(SEED)
    subset_qids = random.sample(all_qids, min(HYDE_SAMPLE_SIZE, len(all_qids)))
    queries = {qid: data["queries"][qid] for qid in subset_qids}
    corpus = data["corpus"]
    qrels = {qid: data["qrels"].get(qid, {}) for qid in subset_qids}

    print(f"\n[HyDE] {dataset_name}: {len(queries)} queries")

    # First stage must be indexed
    first_stage = DenseRetriever(
        model_id=MODELS["dense_general"],
        dataset_name=dataset_name,
        cache_dir=CACHE_DIR,
    )
    first_stage.index(corpus)

    hyde = HyDERetriever(
        first_stage=first_stage,
        dataset_name=dataset_name,
        cache_dir=CACHE_DIR,
    )
    hyde.index(corpus)

    metrics, run = run_retriever("HyDE", hyde, queries, qrels)
    save_per_query("HyDE", dataset_name, run, queries, qrels, RESULTS_DIR)

    print(f"MRR@10: {metrics['MRR@10']:.4f}  NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"Latency: {metrics['Latency_ms']:.1f} ms/query")

    # Print generated hypotheticals for analysis
    hyp_path = CACHE_DIR / "hyde" / f"{dataset_name}__hypotheticals.json"
    if hyp_path.exists():
        import json
        hyps = json.loads(hyp_path.read_text(encoding="utf-8"))
        print(f"\nGenerated {len(hyps)} hypotheticals. See: {hyp_path}")

print("\nDone. Hypotheticals saved to cache/hyde/.")
