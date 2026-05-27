import shutil
from pathlib import Path

from src.methods.base import Retriever

_RAGATOUILLE_DEFAULT = Path(".ragatouille") / "colbert" / "indexes"


class ColBERTRetriever(Retriever):
    def __init__(self, dataset_name: str, cache_dir: Path):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.index_dir = self.cache_dir / "colbert" / f"{dataset_name}__index"
        self._rag = None

    def index(self, corpus: dict[str, str]) -> None:
        from ragatouille import RAGPretrainedModel

        if self.index_dir.exists():
            self._rag = RAGPretrainedModel.from_index(str(self.index_dir))
            return

        doc_ids = sorted(corpus.keys())
        texts = [corpus[did] for did in doc_ids]

        rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        rag.index(
            collection=texts,
            document_ids=doc_ids,
            index_name=self.dataset_name,
            max_document_length=256,
            split_documents=False,
        )

        # Move index from ragatouille default location to our cache dir
        default_path = _RAGATOUILLE_DEFAULT / self.dataset_name
        if default_path.exists():
            self.index_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(default_path), str(self.index_dir))

        self._rag = RAGPretrainedModel.from_index(str(self.index_dir))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        results = self._rag.search(query=query, k=k)
        return [(r["document_id"], float(r["score"])) for r in results]
