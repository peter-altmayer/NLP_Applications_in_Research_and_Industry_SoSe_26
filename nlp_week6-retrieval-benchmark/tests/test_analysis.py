from src.analysis.disagreement import find_bm25_beats_dense, find_max_divergence_queries, gold_rank

QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
BM25_RUN = {
    "q1": {"d1": 3.0, "d2": 1.0},  # d1 at rank 1 → gold_rank=1
    "q2": {"d1": 3.0, "d2": 1.0},  # d2 at rank 2 → gold_rank=2
    "q3": {"d3": 3.0, "d1": 1.0},  # d3 at rank 1 → gold_rank=1
}
DENSE_RUN = {
    "q1": {"d2": 3.0, "d1": 1.0},  # d1 at rank 2 → gold_rank=2
    "q2": {"d2": 3.0, "d1": 1.0},  # d2 at rank 1 → gold_rank=1
    "q3": {"d1": 3.0, "d3": 1.0},  # d3 at rank 2 → gold_rank=2
}


def test_gold_rank_found():
    assert gold_rank("q1", QRELS, BM25_RUN) == 1
    assert gold_rank("q2", QRELS, BM25_RUN) == 2


def test_gold_rank_not_found():
    empty_run = {"q1": {"d2": 1.0}}
    assert gold_rank("q1", QRELS, empty_run, max_rank=999) == 999


def test_find_bm25_beats_dense():
    df = find_bm25_beats_dense(BM25_RUN, DENSE_RUN, QRELS, n=5)
    # q1: bm25_rank=1, dense_rank=2 → BM25 wins by 1
    # q3: bm25_rank=1, dense_rank=2 → BM25 wins by 1
    assert len(df) == 2
    assert set(df["query_id"]) == {"q1", "q3"}


def test_find_max_divergence():
    runs = {"BM25": BM25_RUN, "Dense": DENSE_RUN}
    df = find_max_divergence_queries(runs, QRELS, n=3)
    # q1: BM25=1, Dense=2 → spread=1. q2: BM25=2, Dense=1 → spread=1. q3: same
    assert len(df) <= 3
    assert "spread" in df.columns
