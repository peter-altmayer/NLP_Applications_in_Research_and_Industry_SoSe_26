"""Find queries where methods disagree most on the gold document's rank."""
import pandas as pd


def gold_rank(
    qid: str,
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    max_rank: int = 1000,
) -> int:
    gold_docs = {did for did, r in qrels.get(qid, {}).items() if r > 0}
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    for rank, (did, _) in enumerate(ranked, start=1):
        if did in gold_docs:
            return rank
    return max_rank


def find_max_divergence_queries(
    runs: dict[str, dict[str, dict[str, float]]],
    qrels: dict[str, dict[str, int]],
    n: int = 5,
) -> pd.DataFrame:
    """Return top-n queries with the largest gold-rank spread across methods."""
    method_names = list(runs.keys())
    qids = list(qrels.keys())
    rows = []
    for qid in qids:
        ranks = {m: gold_rank(qid, qrels, runs[m]) for m in method_names}
        spread = max(ranks.values()) - min(ranks.values())
        rows.append({"query_id": qid, "spread": spread, **ranks})
    df = pd.DataFrame(rows).sort_values("spread", ascending=False).head(n)
    return df.reset_index(drop=True)


def find_bm25_beats_dense(
    bm25_run: dict[str, dict[str, float]],
    dense_run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    n: int = 5,
) -> pd.DataFrame:
    """Return queries where BM25 rank < dense rank (BM25 wins)."""
    qids = list(qrels.keys())
    rows = []
    for qid in qids:
        bm25_r = gold_rank(qid, qrels, bm25_run)
        dense_r = gold_rank(qid, qrels, dense_run)
        if bm25_r < dense_r:
            rows.append({"query_id": qid, "bm25_rank": bm25_r, "dense_rank": dense_r, "improvement": dense_r - bm25_r})
    df = pd.DataFrame(rows).sort_values("improvement", ascending=False).head(n)
    return df.reset_index(drop=True)
