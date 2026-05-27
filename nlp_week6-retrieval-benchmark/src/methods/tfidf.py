from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.cache import doc_ids_hash, load_or_compute
from src.methods.base import Retriever


class TFIDFRetriever(Retriever):
    def __init__(self, dataset_name: str, cache_dir: Path):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_matrix = None
        self._doc_ids: list[str] | None = None

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys())
        ids_hash = doc_ids_hash(self._doc_ids)
        cache_path = self.cache_dir / "tfidf" / f"{self.dataset_name}__{ids_hash}.pkl"

        def compute():
            texts = [corpus[did] for did in self._doc_ids]
            vec = TfidfVectorizer()
            mat = vec.fit_transform(texts)
            return (vec, mat)

        self._vectorizer, self._doc_matrix = load_or_compute(cache_path, compute)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._doc_matrix)[0]
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
