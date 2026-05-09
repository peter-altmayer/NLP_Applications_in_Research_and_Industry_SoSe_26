import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import build_bm25, load_bm25, retrieve_bm25, save_bm25

CORPUS = [
    "The capital of France is Paris.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital city of Japan.",
    "The Eiffel Tower is located in Paris, France.",
]


def test_retrieve_bm25_returns_k_docs():
    index = build_bm25(CORPUS)
    results = retrieve_bm25(index, CORPUS, "capital of France", k=2)
    assert len(results) == 2


def test_retrieve_bm25_relevance():
    index = build_bm25(CORPUS)
    results = retrieve_bm25(index, CORPUS, "Eiffel Tower Paris", k=1)
    assert "Eiffel Tower" in results[0]


def test_retrieve_bm25_returns_strings():
    index = build_bm25(CORPUS)
    results = retrieve_bm25(index, CORPUS, "Tokyo Japan", k=2)
    assert all(isinstance(r, str) for r in results)


def test_bm25_save_load_roundtrip():
    index = build_bm25(CORPUS)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = Path(f.name)
    save_bm25(index, path)
    loaded = load_bm25(path)
    original = index.get_scores("Paris".split())
    reloaded = loaded.get_scores("Paris".split())
    assert list(original) == list(reloaded)
    path.unlink()


def test_dpr_embeddings_shape():
    pytest.importorskip("transformers")
    from src.retriever import build_dpr_embeddings
    corpus = ["hello world", "foo bar baz"]
    embeddings = build_dpr_embeddings(
        corpus,
        ctx_encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
        device="cpu",
    )
    assert embeddings.shape == (2, 768)


def test_retrieve_dpr_returns_k_docs():
    pytest.importorskip("transformers")
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    from src.retriever import build_dpr_embeddings, retrieve_dpr

    corpus = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital city of Japan.",
    ]
    embeddings = build_dpr_embeddings(
        corpus, "facebook/dpr-ctx_encoder-single-nq-base", device="cpu"
    )
    q_enc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    results = retrieve_dpr(embeddings, corpus, q_enc, q_tok, "capital France", k=2, device="cpu")
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
