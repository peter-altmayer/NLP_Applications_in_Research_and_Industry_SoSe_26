import pytest

TINY_CORPUS = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "machine learning is a subset of artificial intelligence",
    "d3": "information retrieval systems rank documents by relevance",
    "d4": "natural language processing handles text and speech",
    "d5": "deep learning uses neural networks with many layers",
}
TINY_QUERIES = {"q1": "quick fox", "q2": "neural networks deep learning"}
TINY_QRELS = {"q1": {"d1": 1}, "q2": {"d5": 1}}


@pytest.fixture
def tiny_corpus():
    return dict(TINY_CORPUS)


@pytest.fixture
def tiny_queries():
    return dict(TINY_QUERIES)


@pytest.fixture
def tiny_qrels():
    return dict(TINY_QRELS)


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return d
