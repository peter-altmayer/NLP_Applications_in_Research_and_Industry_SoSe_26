import json
from pathlib import Path
import pytest


def test_save_and_load_roundtrip(tmp_path):
    from utils import save_cache, load_cache
    data = [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.42}]
    path = tmp_path / "cache.json"
    save_cache(path, data)
    loaded = load_cache(path)
    assert loaded == data


def test_load_cache_missing_returns_none(tmp_path):
    from utils import load_cache
    assert load_cache(tmp_path / "nonexistent.json") is None


def test_save_cache_creates_file(tmp_path):
    from utils import save_cache
    path = tmp_path / "out.json"
    save_cache(path, [{"a": 1}])
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]
