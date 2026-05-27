from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        """Build/load index for the given corpus."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs sorted by score descending."""
