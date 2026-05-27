import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np


def doc_ids_hash(doc_ids: list[str]) -> str:
    return hashlib.sha1(",".join(sorted(doc_ids)).encode()).hexdigest()[:12]


def _load(path: Path) -> Any:
    suffix = path.suffix
    if suffix == ".npy":
        return np.load(str(path))
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unknown cache format: {suffix}")


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    if suffix == ".npy":
        np.save(str(path), obj)
    elif suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown cache format: {suffix}")


def load_or_compute(path: Path, compute_fn: Callable[[], Any]) -> Any:
    path = Path(path)
    if path.exists():
        return _load(path)
    result = compute_fn()
    _save(path, result)
    return result
