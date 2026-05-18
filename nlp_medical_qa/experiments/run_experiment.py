"""Main RAG pipeline CLI.

Usage examples
--------------
# 1. Fine-tune V1 from scratch (must do before evaluating V1)
python experiments/run_experiment.py --model v1 --fine-tune

# 2. Fine-tune V3 on biomedical data (must do before evaluating V3)
python experiments/run_experiment.py --model v3 --fine-tune

# 3. Evaluate V2 on BioASQ with dense retrieval, k=5
python experiments/run_experiment.py --model v2 --dataset bioasq --retrieval dense --k 5

# 4. Evaluate V4 on PubMedQA with BM25
python experiments/run_experiment.py --model v4 --dataset pubmedqa --retrieval bm25 --k 5

# 5. Run all combinations (convenience)
python experiments/run_experiment.py --all
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.dataset import (
    SUBSET,
    build_corpus,
    load_bioasq,
    load_bioasq_train_sft,
    load_pubmedqa,
    load_pubmedqa_train_sft,
    load_squad,
    load_squad_train_sft,
)
from src.evaluate import bertscore_batch, evaluate_sample
from src.model import CHECKPOINT_DIR, build_model, load_model
from src.retriever import BM25Retriever, DenseRetriever, compute_retrieval_metrics
from src.privacy import mask_phi

RESULTS = Path(__file__).parent.parent / "results"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_eval(dataset: str, n: int) -> list:
    if dataset == "bioasq":
        return load_bioasq(max_samples=n, split="test")
    elif dataset == "pubmedqa":
        return load_pubmedqa(max_samples=n, split="test")
    elif dataset == "squad":
        return load_squad(split="validation", max_samples=n)
    raise ValueError(dataset)


def _load_index(dataset: str, retrieval: str):
    if retrieval == "bm25":
        p = PROCESSED / f"bm25_{dataset}.pkl"
        if not p.exists():
            _abort_no_index(dataset, retrieval)
        return BM25Retriever.load(p)
    else:
        p = PROCESSED / f"dense_{dataset}"
        if not (Path(str(p) + ".faiss")).exists():
            _abort_no_index(dataset, retrieval)
        return DenseRetriever.load(p)


def _abort_no_index(dataset: str, retrieval: str):
    sys.exit(
        f"Index not found for {dataset}/{retrieval}.\n"
        f"Run: python experiments/build_index.py --datasets {dataset} --retrieval {retrieval}"
    )


# ---------------------------------------------------------------------------
# Fine-tuning routines
# ---------------------------------------------------------------------------

def _fine_tune_v1() -> None:
    print("=== Fine-tuning V1 (from scratch on SQuAD-10k) ===")
    from src.model import TrainableQAModel, V1_TOKENIZER

    model = build_model("v1", device=DEVICE)
    train_data = load_squad_train_sft(max_samples=10_000)
    model.fine_tune(
        train_data,
        output_dir=CHECKPOINT_DIR / "v1",
        epochs=3,
        lr=3e-4,     # higher LR for random init
        batch_size=16,
    )


def _fine_tune_v3() -> None:
    print("=== Fine-tuning V3 (bert-base-uncased on BioASQ yesno + PubMedQA) ===")
    model = build_model("v3", device=DEVICE)

    bio_data = load_bioasq_train_sft()
    pqa_data = load_pubmedqa_train_sft()
    train_data = bio_data + pqa_data
    print(f"  BioASQ yesno: {len(bio_data)}  |  PubMedQA: {len(pqa_data)}  |  Total: {len(train_data)}")

    model.fine_tune(
        train_data,
        output_dir=CHECKPOINT_DIR / "v3",
        epochs=4,
        lr=2e-5,
        batch_size=16,
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model_variant: str,
    dataset: str,
    retrieval: str,
    k: int,
    n: int,
    skip_faithfulness: bool,
) -> Path:
    print(f"\n=== {model_variant.upper()} | {dataset} | {retrieval} | k={k} ===")

    samples = _load_eval(dataset, n)
    retriever = _load_index(dataset, retrieval)
    model = load_model(model_variant, device=DEVICE)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"{model_variant}_{dataset}_{retrieval}_k{k}.csv"

    rows = []
    preds, golds = [], []

    for s in tqdm(samples, desc="Evaluating"):
        question = s["question"]
        gold = s["answer"]
        relevant = s["snippets"]

        # Retrieve
        retrieved = retriever.retrieve(question, k=k)
        context = " ".join(retrieved)

        # PHI masking (illustrative — datasets are already PHI-free)
        context = mask_phi(context)

        # Predict
        pred = model.predict(question, context)

        # Retrieval metrics
        ret_metrics = compute_retrieval_metrics(retrieved, relevant, k=k)

        # Generation metrics (BERTScore computed in batch later)
        gen_metrics = evaluate_sample(
            pred, gold, context, skip_faithfulness=skip_faithfulness
        )

        preds.append(pred)
        golds.append(gold)

        rows.append({
            "model": model_variant,
            "dataset": dataset,
            "retrieval": retrieval,
            "k": k,
            "id": s.get("id", ""),
            "question": question,
            "gold_answer": gold,
            "answer_type": s.get("answer_type", ""),
            "predicted_answer": pred,
            "retrieved_passages": " ||| ".join(retrieved),
            "em": gen_metrics["em"],
            "f1": gen_metrics["f1"],
            "rouge_l": gen_metrics["rouge_l"],
            "faithfulness": gen_metrics["faithfulness"],
            "bertscore": None,      # filled below
            "precision_at_k": ret_metrics["precision_at_k"],
            "recall_at_k": ret_metrics["recall_at_k"],
        })

    # Batch BERTScore
    print("Computing BERTScore …")
    bs_scores = bertscore_batch(preds, golds)
    for row, bs in zip(rows, bs_scores):
        row["bertscore"] = bs

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Quick summary
    mean = lambda col: round(sum(r[col] or 0 for r in rows) / len(rows), 4)
    print(f"  EM={mean('em')}  F1={mean('f1')}  ROUGE-L={mean('rouge_l')}"
          f"  BERTScore={mean('bertscore')}  P@k={mean('precision_at_k')}  R@k={mean('recall_at_k')}")
    print(f"  Saved → {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_MODELS = ["v1", "v2", "v3", "v4"]
ALL_DATASETS = ["bioasq", "pubmedqa", "squad"]
ALL_RETRIEVALS = ["bm25", "dense"]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Medical QA experiment runner")
    parser.add_argument("--model", choices=ALL_MODELS, help="Model variant")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--retrieval", choices=ALL_RETRIEVALS, default="dense")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved passages")
    parser.add_argument("--n", type=int, default=200, help="Eval samples per dataset")
    parser.add_argument("--fine-tune", action="store_true", help="Train V1/V3 and save checkpoint")
    parser.add_argument("--skip-faithfulness", action="store_true", help="Skip slow NLI step")
    parser.add_argument("--all", action="store_true", help="Run all model×dataset×retrieval combos")
    args = parser.parse_args()

    if args.fine_tune:
        if args.model == "v1":
            _fine_tune_v1()
        elif args.model == "v3":
            _fine_tune_v3()
        else:
            sys.exit("--fine-tune only applies to --model v1 or v3")
        return

    if args.all:
        for model in ALL_MODELS:
            for dataset in ALL_DATASETS:
                for retrieval in ALL_RETRIEVALS:
                    try:
                        run_eval(model, dataset, retrieval, args.k, args.n, args.skip_faithfulness)
                    except FileNotFoundError as e:
                        print(f"  SKIP {model}/{dataset}/{retrieval}: {e}")
        return

    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required (or use --all or --fine-tune)")

    run_eval(args.model, args.dataset, args.retrieval, args.k, args.n, args.skip_faithfulness)


if __name__ == "__main__":
    main()
