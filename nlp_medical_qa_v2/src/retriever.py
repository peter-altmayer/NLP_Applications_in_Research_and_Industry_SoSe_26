"""BM25, dense (FAISS), and oracle retrieval over a passage corpus."""
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(d) for d in corpus])

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = np.argsort(scores)[::-1][:k]
        return [self.corpus[i] for i in top_k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "bm25": self._bm25}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.corpus = state["corpus"]
        obj._bm25 = state["bm25"]
        return obj


# ---------------------------------------------------------------------------
# Dense (FAISS + sentence-transformers)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, corpus: List[str], model_name: str = EMBED_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

        embeddings = self._model.encode(
            corpus, show_progress_bar=True, batch_size=64, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        import faiss

        q_emb = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        _, idxs = self._index.search(q_emb, k)
        return [self.corpus[i] for i in idxs[0] if 0 <= i < len(self.corpus)]

    def save(self, path: Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path) + ".faiss")
        with open(str(path) + ".pkl", "wb") as f:
            pickle.dump({"corpus": self.corpus, "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path: Path) -> "DenseRetriever":
        import faiss
        from sentence_transformers import SentenceTransformer

        path = Path(path)
        with open(str(path) + ".pkl", "rb") as f:
            state = pickle.load(f)

        obj = cls.__new__(cls)
        obj.corpus = state["corpus"]
        obj.model_name = state["model_name"]
        obj._model = SentenceTransformer(obj.model_name)
        obj._index = faiss.read_index(str(path) + ".faiss")
        return obj


# ---------------------------------------------------------------------------
# Oracle retriever (gold-passage baseline — no indexing needed)
# ---------------------------------------------------------------------------

class OracleRetriever:
    """Returns the annotated gold snippets directly.

    Bypasses retrieval entirely to isolate model quality from retrieval quality.
    P@k will be near-perfect by construction; any remaining F1 gap is a model failure.
    """

    def retrieve(self, query: str, k: int = 5, snippets: Optional[List[str]] = None) -> List[str]:
        if not snippets:
            return []
        return snippets[:k]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    retrieved: List[str],
    relevant: List[str],
    k: int,
) -> dict:
    """Precision@k and Recall@k.

    relevant: gold passages for this question (exact text match).
    """
    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    hits = sum(1 for p in retrieved_k if p in relevant_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "hits": hits,
        "k": k,
    }
