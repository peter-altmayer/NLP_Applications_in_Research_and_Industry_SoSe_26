from src.eval.metrics import evaluate

# Two queries:
# q1: d1 is relevant. Run: d1 at rank 1 → MRR=1.0, P@1=1.0, NDCG=1.0
# q2: d2 is relevant. Run: d1 at rank 1 (wrong), d2 at rank 2 → MRR=0.5, P@1=0.0
QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}
RUN = {
    "q1": {"d1": 2.0, "d2": 1.0},
    "q2": {"d1": 2.0, "d2": 1.0},
}


def test_precision_at_1():
    r = evaluate(QRELS, RUN)
    # q1: P@1=1.0, q2: P@1=0.0 → avg=0.5
    assert abs(r["P@1"] - 0.5) < 0.01


def test_precision_at_5():
    r = evaluate(QRELS, RUN)
    # With only 2 docs, P@5 = #{relevant in top-5}/5. q1=1/5=0.2, q2=1/5=0.2 → avg=0.2
    assert abs(r["P@5"] - 0.2) < 0.01


def test_recall_at_10():
    r = evaluate(QRELS, RUN)
    # Both relevant docs are in the 2-doc run → R@10=1.0 for both
    assert abs(r["R@10"] - 1.0) < 0.01


def test_mrr_at_10():
    r = evaluate(QRELS, RUN)
    # q1: first relevant at rank 1 → RR=1.0. q2: first relevant at rank 2 → RR=0.5. avg=0.75
    assert abs(r["MRR@10"] - 0.75) < 0.01


def test_map_at_100():
    r = evaluate(QRELS, RUN)
    # q1: AP = P@1 = 1.0. q2: AP = P@2 = 0.5. avg=0.75
    assert abs(r["MAP@100"] - 0.75) < 0.01


def test_ndcg_at_10():
    r = evaluate(QRELS, RUN)
    # q1: NDCG=1.0. q2: DCG=1/log2(3)=0.6309, idealDCG=1.0, NDCG=0.6309. avg≈0.815
    assert abs(r["NDCG@10"] - 0.815) < 0.01


def test_latency_key_absent():
    # evaluate() does not return Latency_ms — that's added by runner
    r = evaluate(QRELS, RUN)
    assert "Latency_ms" not in r


def test_returns_all_expected_keys():
    r = evaluate(QRELS, RUN)
    expected = {"P@1", "P@5", "P@10", "R@10", "R@50", "R@100", "MRR@10", "MAP@100", "NDCG@10"}
    assert expected.issubset(r.keys())
