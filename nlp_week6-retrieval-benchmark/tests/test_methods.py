from unittest.mock import MagicMock, patch

from src.methods.base import Retriever
from src.methods.bm25 import BM25Retriever
from src.methods.colbert import ColBERTRetriever
from src.methods.cross_encoder import CrossEncoderRetriever
from src.methods.dense import DenseRetriever
from src.methods.hybrid_rrf import HybridRRFRetriever, reciprocal_rank_fusion
from src.methods.hyde import HyDERetriever
from src.methods.tfidf import TFIDFRetriever


# ---------------------------------------------------------------------------
# Task 6: Retriever ABC interface
# ---------------------------------------------------------------------------

class _ConstantRetriever(Retriever):
    """Returns docs in reverse lexicographic order with constant scores."""

    def index(self, corpus: dict[str, str]) -> None:
        self._doc_ids = sorted(corpus.keys(), reverse=True)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        ids = self._doc_ids[:k]
        return [(did, float(len(ids) - i)) for i, did in enumerate(ids)]


def test_retriever_abc_interface(tiny_corpus, cache_dir):
    r = _ConstantRetriever()
    r.index(tiny_corpus)
    results = r.search("anything", k=3)
    assert len(results) == 3
    assert all(isinstance(did, str) and isinstance(score, float) for did, score in results)
    # scores must be sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Task 7: M1 BM25
# ---------------------------------------------------------------------------

def test_bm25_returns_sorted_scores(tiny_corpus, cache_dir):
    r = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_ranks_relevant_doc_first(tiny_corpus, cache_dir):
    r = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    top_doc = results[0][0]
    assert top_doc == "d1"  # "quick" and "fox" appear only in d1


def test_bm25_caches_index(tiny_corpus, cache_dir):
    r1 = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r1.index(tiny_corpus)
    # Second retriever loads from cache (no recompute)
    r2 = BM25Retriever(dataset_name="test", cache_dir=cache_dir)
    r2.index(tiny_corpus)
    result1 = r1.search("quick fox", k=3)
    result2 = r2.search("quick fox", k=3)
    assert [did for did, _ in result1] == [did for did, _ in result2]


# ---------------------------------------------------------------------------
# Task 8: M2 TF-IDF
# ---------------------------------------------------------------------------

def test_tfidf_returns_sorted_scores(tiny_corpus, cache_dir):
    r = TFIDFRetriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_ranks_relevant_doc_first(tiny_corpus, cache_dir):
    r = TFIDFRetriever(dataset_name="test", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert results[0][0] == "d1"


# ---------------------------------------------------------------------------
# Task 9: M3 Dense (MiniLM) — marked integration, requires model download
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.integration
def test_dense_minilm_ranks_relevant_first(tiny_corpus, cache_dir):
    r = DenseRetriever(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dataset_name="test_minilm",
        cache_dir=cache_dir,
    )
    r.index(tiny_corpus)
    results = r.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # MiniLM should rank d1 ("quick brown fox") at position 1
    assert results[0][0] == "d1"


@pytest.mark.integration
def test_dense_caches_embeddings(tiny_corpus, cache_dir):
    r = DenseRetriever(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dataset_name="test_cache_check",
        cache_dir=cache_dir,
    )
    r.index(tiny_corpus)
    cache_files = list((cache_dir / "embeddings").glob("*.npy"))
    assert len(cache_files) == 1


# ---------------------------------------------------------------------------
# Task 10: M4 Dense (msmarco-distilbert) — marked integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_dense_msmarco_distilbert_runs(tiny_corpus, cache_dir):
    r = DenseRetriever(
        model_id="sentence-transformers/msmarco-distilbert-base-v3",
        dataset_name="test_msmarco",
        cache_dir=cache_dir,
    )
    r.index(tiny_corpus)
    results = r.search("information retrieval", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # d3 ("information retrieval systems rank documents") should score high
    top_ids = [did for did, _ in results]
    assert "d3" in top_ids


# ---------------------------------------------------------------------------
# Task 11: M5 Hybrid RRF
# ---------------------------------------------------------------------------

def test_rrf_fusion_merges_rankings():
    # Two rankings agree on d1 at top
    r1 = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
    r2 = [("d1", 2.5), ("d3", 1.5), ("d2", 0.5)]
    fused = reciprocal_rank_fusion([r1, r2], k=60)
    # d1 at rank 1 in both → highest RRF score
    assert fused[0][0] == "d1"
    assert all(score_a >= score_b for (_, score_a), (_, score_b) in zip(fused, fused[1:]))


def test_hybrid_rrf_combines_bm25_and_dense(tiny_corpus, cache_dir):
    bm25 = BM25Retriever(dataset_name="test_rrf", cache_dir=cache_dir)
    dense = _DenseStub()  # use a stub to avoid model download in unit tests
    hybrid = HybridRRFRetriever(retrievers=[bm25, dense], k=60)
    hybrid.index(tiny_corpus)
    results = hybrid.search("quick fox", k=5)
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] == "d1"


class _DenseStub(Retriever):
    """Stub dense retriever: returns BM25-like order using TF-IDF for tests without model downloads."""

    def index(self, corpus: dict[str, str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._doc_ids = sorted(corpus.keys())
        texts = [corpus[did] for did in self._doc_ids]
        self._vec = TfidfVectorizer()
        self._mat = self._vec.fit_transform(texts)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vec.transform([query])
        scores = cosine_similarity(q_vec, self._mat)[0]
        top_k_idx = list(np.argsort(-scores)[:k])
        return [(self._doc_ids[i], float(scores[i])) for i in top_k_idx]


# ---------------------------------------------------------------------------
# Task 12: M7 Cross-Encoder Re-Ranker — marked integration (model download)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cross_encoder_reranks(tiny_corpus, cache_dir):
    first_stage = BM25Retriever(dataset_name="test_ce", cache_dir=cache_dir)
    first_stage.index(tiny_corpus)

    reranker = CrossEncoderRetriever(
        first_stage=first_stage,
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        first_stage_k=5,
    )
    reranker.index(tiny_corpus)

    results = reranker.search("information retrieval rank documents", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # d3 "information retrieval systems rank documents by relevance" should be top
    assert results[0][0] == "d3"


# ---------------------------------------------------------------------------
# Task 13: M6 ColBERT — marked integration (model download + indexing)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_colbert_indexes_and_searches(tiny_corpus, cache_dir):
    r = ColBERTRetriever(dataset_name="test_colbert", cache_dir=cache_dir)
    r.index(tiny_corpus)
    results = r.search("information retrieval", k=3)
    assert len(results) >= 1
    # Scores should be sorted descending (ColBERT returns MaxSim scores)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    top_ids = [did for did, _ in results]
    assert "d3" in top_ids  # "information retrieval systems" should score high


# ---------------------------------------------------------------------------
# Task 15: M8 HyDE (mocked — no API key needed)
# ---------------------------------------------------------------------------

def test_hyde_generates_hypothetical_and_retrieves(tiny_corpus, cache_dir):
    first_stage = BM25Retriever(dataset_name="test_hyde_fs", cache_dir=cache_dir)
    first_stage.index(tiny_corpus)

    hyde = HyDERetriever(
        first_stage=first_stage,
        dataset_name="test_hyde",
        cache_dir=cache_dir,
    )
    hyde.index(tiny_corpus)

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Artificial intelligence and machine learning are closely related fields."

    with patch.object(hyde, "_get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_client_fn.return_value = mock_client

        with patch("time.sleep"):  # skip rate-limit sleep
            results = hyde.search("machine learning artificial intelligence", k=3)

    assert len(results) >= 1
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # d2 "machine learning is a subset of artificial intelligence" should rank high
    top_ids = [did for did, _ in results]
    assert "d2" in top_ids


def test_hyde_caches_hypotheticals(tiny_corpus, cache_dir):
    first_stage = BM25Retriever(dataset_name="test_hyde_cache_fs", cache_dir=cache_dir)
    first_stage.index(tiny_corpus)

    hyde = HyDERetriever(
        first_stage=first_stage,
        dataset_name="test_hyde_cache",
        cache_dir=cache_dir,
    )
    hyde.index(tiny_corpus)

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Test hypothetical passage."

    with patch.object(hyde, "_get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_client_fn.return_value = mock_client

        with patch("time.sleep"):
            hyde.search("test query", k=2)
            call_count_after_first = mock_client.chat.completions.create.call_count
            hyde.search("test query", k=2)  # same query — should use cache
            call_count_after_second = mock_client.chat.completions.create.call_count

    assert call_count_after_first == 1
    assert call_count_after_second == 1  # not called again
