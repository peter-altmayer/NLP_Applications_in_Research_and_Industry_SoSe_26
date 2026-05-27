from sentence_transformers import CrossEncoder

from src.methods.base import Retriever

_MAX_DOC_CHARS = 2000  # ~500 tokens; truncate doc, never query


class CrossEncoderRetriever(Retriever):
    def __init__(
        self,
        first_stage: Retriever,
        model_id: str,
        first_stage_k: int = 100,
    ):
        self.first_stage = first_stage
        self.model_id = model_id
        self.first_stage_k = first_stage_k
        self._model: CrossEncoder | None = None
        self._corpus: dict[str, str] | None = None

    def _model_(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_id, max_length=512)
        return self._model

    def index(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus
        # first_stage must already be indexed externally before calling this

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        candidates = self.first_stage.search(query, self.first_stage_k)
        if not candidates:
            return []
        pairs = [(query, self._corpus[did][:_MAX_DOC_CHARS]) for did, _ in candidates]
        scores = self._model_().predict(pairs)
        ranked = sorted(
            zip([did for did, _ in candidates], scores.tolist()),
            key=lambda x: -x[1],
        )
        return [(did, float(score)) for did, score in ranked[:k]]
