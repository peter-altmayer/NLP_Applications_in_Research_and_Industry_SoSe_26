import pytrec_eval

_PYTREC_KEYS = {
    "P_1", "P_5", "P_10",
    "recall_10", "recall_50", "recall_100",
    "map_cut_100", "ndcg_cut_10",
}

_DISPLAY = {
    "P_1": "P@1",
    "P_5": "P@5",
    "P_10": "P@10",
    "recall_10": "R@10",
    "recall_50": "R@50",
    "recall_100": "R@100",
    "recip_rank": "MRR@10",
    "map_cut_100": "MAP@100",
    "ndcg_cut_10": "NDCG@10",
}


def evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average all metrics over queries. run may have any number of docs per query."""
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, _PYTREC_KEYS)
    per_query = evaluator.evaluate(run)

    # MRR@10: truncate run to top-10 before computing recip_rank
    run_at10 = {
        qid: dict(sorted(scores.items(), key=lambda x: -x[1])[:10])
        for qid, scores in run.items()
    }
    mrr_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    mrr_per_query = mrr_evaluator.evaluate(run_at10)

    out: dict[str, float] = {}
    for internal_key, display_key in _DISPLAY.items():
        if internal_key == "recip_rank":
            vals = [mrr_per_query[qid]["recip_rank"] for qid in mrr_per_query]
        else:
            vals = [per_query[qid][internal_key] for qid in per_query if internal_key in per_query[qid]]
        out[display_key] = sum(vals) / len(vals) if vals else 0.0

    return out
