import numpy as np
import pytest
from src.cache import doc_ids_hash, load_or_compute


def test_load_or_compute_npy_calls_fn_once(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return np.array([1.0, 2.0, 3.0])

    path = cache_dir / "test.npy"
    result1 = load_or_compute(path, compute)
    result2 = load_or_compute(path, compute)

    assert len(calls) == 1
    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(result1, [1.0, 2.0, 3.0])


def test_load_or_compute_json_idempotent(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return {"key": "value", "nested": [1, 2]}

    path = cache_dir / "test.json"
    r1 = load_or_compute(path, compute)
    r2 = load_or_compute(path, compute)

    assert len(calls) == 1
    assert r1 == r2 == {"key": "value", "nested": [1, 2]}


def test_load_or_compute_pkl_idempotent(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return {"sparse": True, "data": list(range(100))}

    path = cache_dir / "test.pkl"
    r1 = load_or_compute(path, compute)
    r2 = load_or_compute(path, compute)

    assert len(calls) == 1
    assert r1 == r2


def test_doc_ids_hash_order_independent():
    h1 = doc_ids_hash(["c", "a", "b"])
    h2 = doc_ids_hash(["a", "b", "c"])
    assert h1 == h2
    assert len(h1) == 12


def test_doc_ids_hash_different_inputs_differ():
    assert doc_ids_hash(["a", "b"]) != doc_ids_hash(["a", "c"])


def test_load_or_compute_creates_parent_dirs(cache_dir):
    path = cache_dir / "deep" / "nested" / "file.json"

    load_or_compute(path, lambda: {"x": 1})

    assert path.exists()
