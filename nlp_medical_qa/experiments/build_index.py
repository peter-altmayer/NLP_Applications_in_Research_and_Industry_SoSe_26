"""Build BM25 and FAISS retrieval indexes for each dataset.

Usage
-----
    python experiments/build_index.py [--datasets bioasq pubmedqa squad] [--retrieval bm25 dense]

Outputs
-------
    data/processed/bm25_<dataset>.pkl
    data/processed/dense_<dataset>.faiss  +  data/processed/dense_<dataset>.pkl
"""
import argparse
import sys
from pathlib import Path

# Make src importable when running from project root or experiments/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    build_corpus,
    load_bioasq,
    load_pubmedqa,
    load_squad,
    SUBSET,
)
from src.retriever import BM25Retriever, DenseRetriever

PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def _load(dataset: str) -> list:
    if dataset == "bioasq":
        # Use full training split as our corpus
        return load_bioasq(max_samples=None, split="train")
    elif dataset == "pubmedqa":
        return load_pubmedqa(max_samples=None, split="train")
    elif dataset == "squad":
        return load_squad(split="train", max_samples=5_000)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")


def build(datasets: list[str], retrieval_types: list[str]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print("Loading samples …")
        samples = _load(ds)
        corpus = build_corpus(samples)
        print(f"  Corpus size: {len(corpus)} unique passages")

        if "bm25" in retrieval_types:
            print("Building BM25 index …")
            bm25 = BM25Retriever(corpus)
            out = PROCESSED / f"bm25_{ds}.pkl"
            bm25.save(out)
            print(f"  Saved → {out}")

        if "dense" in retrieval_types:
            print("Building dense (FAISS) index …")
            dense = DenseRetriever(corpus)
            out = PROCESSED / f"dense_{ds}"
            dense.save(out)
            print(f"  Saved → {out}.faiss / .pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument(
        "--datasets", nargs="+", default=["bioasq", "pubmedqa", "squad"],
        choices=["bioasq", "pubmedqa", "squad"],
        help="Which datasets to index",
    )
    parser.add_argument(
        "--retrieval", nargs="+", default=["bm25", "dense"],
        choices=["bm25", "dense"],
        help="Which retriever types to build",
    )
    args = parser.parse_args()
    build(args.datasets, args.retrieval)
    print("\nDone — all indexes built.")


if __name__ == "__main__":
    main()
