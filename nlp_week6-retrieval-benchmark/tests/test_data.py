import pytest
from src.data.load_msmarco import load_msmarco
from src.data.load_scifact import load_scifact


@pytest.mark.integration
def test_load_msmarco_structure():
    data = load_msmarco(sample_size=10, seed=42)
    assert set(data.keys()) == {"queries", "corpus", "qrels"}
    assert len(data["queries"]) > 0
    assert len(data["corpus"]) >= len(data["queries"])  # at least as many docs as queries
    # Every query in qrels must be in queries
    for qid in data["qrels"]:
        assert qid in data["queries"]
    # Every doc in qrels must be in corpus
    for qid, rels in data["qrels"].items():
        for did in rels:
            assert did in data["corpus"], f"{did} missing from corpus"
    # Relevance is binary (0 or 1)
    for qid, rels in data["qrels"].items():
        for did, rel in rels.items():
            assert rel in (0, 1)


@pytest.mark.integration
def test_load_msmarco_sampling_deterministic():
    d1 = load_msmarco(sample_size=10, seed=42)
    d2 = load_msmarco(sample_size=10, seed=42)
    assert list(d1["queries"].keys()) == list(d2["queries"].keys())


@pytest.mark.integration
def test_load_msmarco_at_least_one_relevant_per_query():
    data = load_msmarco(sample_size=20, seed=42)
    for qid in data["queries"]:
        assert qid in data["qrels"], f"query {qid} has no qrels entry"
        assert any(r > 0 for r in data["qrels"][qid].values()), f"query {qid} has no relevant docs"


@pytest.mark.integration
def test_load_scifact_structure():
    data = load_scifact()
    assert set(data.keys()) == {"queries", "corpus", "qrels"}
    assert len(data["queries"]) == 300
    assert len(data["corpus"]) >= 5000
    for qid in data["qrels"]:
        assert qid in data["queries"]
    for qid, rels in data["qrels"].items():
        for did in rels:
            assert did in data["corpus"]
    # SciFact has graded relevance 0/1/2
    all_scores = [r for rels in data["qrels"].values() for r in rels.values()]
    assert all(s in (0, 1, 2) for s in all_scores)
    assert any(s == 2 for s in all_scores), "expected at least one score=2 in SciFact"


@pytest.mark.integration
def test_load_scifact_deterministic():
    d1 = load_scifact()
    d2 = load_scifact()
    assert list(d1["queries"].keys()) == list(d2["queries"].keys())
    assert list(d1["corpus"].keys()) == list(d2["corpus"].keys())
