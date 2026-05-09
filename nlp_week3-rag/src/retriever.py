import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(corpus: list[str]) -> BM25Okapi:
    return BM25Okapi([doc.lower().split() for doc in corpus])


def save_bm25(index: BM25Okapi, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_bm25(path: Path) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_bm25(index: BM25Okapi, corpus: list[str], query: str, k: int) -> list[str]:
    scores = index.get_scores(query.lower().split())
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i] for i in top_k]
