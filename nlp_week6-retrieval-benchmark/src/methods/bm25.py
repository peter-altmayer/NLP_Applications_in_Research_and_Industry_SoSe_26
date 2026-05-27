from pathlib import Path

from rank_bm25 import BM25Okapi

from src.cache import doc_ids_hash, load_or_compute
from src.methods.base import Retriever


class BM25Retriever(Retriever):
    def __init__(self, dataset_name: str, cache_dir: Path):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] | None = None

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys())
        ids_hash = doc_ids_hash(self._doc_ids)
        cache_path = self.cache_dir / "bm25" / f"{self.dataset_name}__{ids_hash}.pkl"

        def compute():
            tokenized = [corpus[did].lower().split() for did in self._doc_ids]
            return BM25Okapi(tokenized)

        self._bm25 = load_or_compute(cache_path, compute)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_k = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
