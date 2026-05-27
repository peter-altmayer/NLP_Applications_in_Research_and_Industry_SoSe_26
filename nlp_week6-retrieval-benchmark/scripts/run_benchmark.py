#!/usr/bin/env python
"""Run M1-M7 on both datasets. Usage: python scripts/run_benchmark.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import numpy as np
import pandas as pd
import torch
import transformers

transformers.logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import CACHE_DIR, FIRST_STAGE_K, MODELS, RESULTS_DIR, RRF_K, SEED, MSMARCO_SAMPLE_SIZE
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact
from src.eval.runner import bold_best_md, run_retriever, save_per_query
from src.methods.bm25 import BM25Retriever
from src.methods.colbert import ColBERTRetriever
from src.methods.cross_encoder import CrossEncoderRetriever
from src.methods.dense import DenseRetriever
from src.methods.hybrid_rrf import HybridRRFRetriever
from src.methods.tfidf import TFIDFRetriever

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "per_query").mkdir(exist_ok=True)

DATASETS = {
    "msmarco": load_msmarco(sample_size=MSMARCO_SAMPLE_SIZE, seed=SEED),
    "scifact": load_scifact(),
}

for dataset_name, data in DATASETS.items():
    queries = data["queries"]
    corpus = data["corpus"]
    qrels = data["qrels"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | {len(queries)} queries | {len(corpus):,} docs")
    print("=" * 60)

    rows = []
    all_runs = {}

    # M1: BM25
    print("\n[M1] BM25 ...")
    m1 = BM25Retriever(dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m1.index(corpus)
    metrics, run = run_retriever("BM25", m1, queries, qrels)
    rows.append(metrics)
    all_runs["BM25"] = run

    # M2: TF-IDF
    print("\n[M2] TF-IDF ...")
    m2 = TFIDFRetriever(dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m2.index(corpus)
    metrics, run = run_retriever("TF-IDF", m2, queries, qrels)
    rows.append(metrics)
    all_runs["TF-IDF"] = run

    # M3: Dense MiniLM (indexed once, reused by M5 and M7)
    print("\n[M3] Dense MiniLM ...")
    m3 = DenseRetriever(model_id=MODELS["dense_general"], dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m3.index(corpus)
    metrics, run = run_retriever("Dense-MiniLM", m3, queries, qrels)
    rows.append(metrics)
    all_runs["Dense-MiniLM"] = run

    # M4: Dense msmarco-distilbert
    print("\n[M4] Dense msmarco-distilbert ...")
    m4 = DenseRetriever(model_id=MODELS["dense_domain"], dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m4.index(corpus)
    metrics, run = run_retriever("Dense-msmarco", m4, queries, qrels)
    rows.append(metrics)
    all_runs["Dense-msmarco"] = run

    # M5: Hybrid RRF (BM25 + MiniLM, both already indexed)
    print("\n[M5] Hybrid RRF (BM25 + MiniLM) ...")
    m5 = HybridRRFRetriever(retrievers=[m1, m3], k=RRF_K)
    metrics, run = run_retriever("Hybrid-RRF", m5, queries, qrels)
    rows.append(metrics)
    all_runs["Hybrid-RRF"] = run

    # M7: Cross-encoder (first stage = M3, already indexed)
    print("\n[M7] Cross-encoder re-rank (top-100 from MiniLM) ...")
    m7 = CrossEncoderRetriever(first_stage=m3, model_id=MODELS["cross_encoder"], first_stage_k=FIRST_STAGE_K)
    m7.index(corpus)
    metrics, run = run_retriever("CrossEncoder", m7, queries, qrels)
    rows.append(metrics)
    all_runs["CrossEncoder"] = run

    # M6: ColBERT (last — slowest to index)
    print("\n[M6] ColBERT ...")
    m6 = ColBERTRetriever(dataset_name=dataset_name, cache_dir=CACHE_DIR)
    m6.index(corpus)
    metrics, run = run_retriever("ColBERT", m6, queries, qrels)
    rows.append(metrics)
    all_runs["ColBERT"] = run

    # Build results DataFrame
    df = pd.DataFrame(rows).set_index("method")
    metric_cols = [c for c in df.columns if c != "method"]
    df = df[metric_cols]

    # Save CSV
    csv_path = RESULTS_DIR / f"{dataset_name}_results.csv"
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    # Save Markdown with bolded best
    md = bold_best_md(df.reset_index())
    md_path = RESULTS_DIR / f"{dataset_name}_results.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Saved: {md_path}")

    # Save per-query rankings for analysis
    for method_name, run in all_runs.items():
        save_per_query(method_name, dataset_name, run, queries, qrels, RESULTS_DIR)

    print("\n" + df.to_string())
