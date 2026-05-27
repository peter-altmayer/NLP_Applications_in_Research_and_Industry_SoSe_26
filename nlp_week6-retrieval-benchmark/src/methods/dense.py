from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.cache import doc_ids_hash, load_or_compute
from src.methods.base import Retriever


class DenseRetriever(Retriever):
    def __init__(self, model_id: str, dataset_name: str, cache_dir: Path):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self._model: SentenceTransformer | None = None
        self._doc_embeddings: np.ndarray | None = None
        self._doc_ids: list[str] | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys())
        texts = [corpus[did] for did in self._doc_ids]
        slug = self.model_id.replace("/", "__")
        ids_hash = doc_ids_hash(self._doc_ids)
        cache_path = self.cache_dir / "embeddings" / f"{slug}__{self.dataset_name}__{ids_hash}.npy"

        def compute() -> np.ndarray:
            return self._get_model().encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        self._doc_embeddings = load_or_compute(cache_path, compute)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q_emb = self._get_model().encode([query], normalize_embeddings=True)
        scores = (q_emb @ self._doc_embeddings.T)[0]
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
