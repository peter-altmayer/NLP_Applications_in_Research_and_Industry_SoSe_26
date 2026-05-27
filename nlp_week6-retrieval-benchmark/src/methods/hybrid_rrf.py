from src.methods.base import Retriever


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRRFRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], k: int = 60):
        self.retrievers = retrievers
        self.k = k

    def index(self, corpus: dict[str, str]) -> None:
        for r in self.retrievers:
            r.index(corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # Retrieve 2× candidates from each component to improve fusion coverage
        rankings = [r.search(query, k * 2) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.k)[:k]
