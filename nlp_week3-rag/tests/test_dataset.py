import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import extract_corpus


def test_extract_corpus_flattens_all_answers():
    records = [
        {"question": "Q1", "answers": ["A1", "A1b"]},
        {"question": "Q2", "answers": ["A2"]},
    ]
    corpus = extract_corpus(records)
    assert "A1" in corpus
    assert "A1b" in corpus
    assert "A2" in corpus
    assert len(corpus) == 3


def test_extract_corpus_deduplicates():
    records = [
        {"question": "Q1", "answers": ["Paris"]},
        {"question": "Q2", "answers": ["Paris"]},
    ]
    corpus = extract_corpus(records)
    assert corpus.count("Paris") == 1


def test_extract_corpus_preserves_order():
    records = [
        {"question": "Q1", "answers": ["B", "A"]},
        {"question": "Q2", "answers": ["C"]},
    ]
    corpus = extract_corpus(records)
    assert corpus == ["B", "A", "C"]
