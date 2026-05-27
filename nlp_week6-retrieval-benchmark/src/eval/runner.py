import time
from pathlib import Path

import pandas as pd

from src.eval.metrics import evaluate
from src.methods.base import Retriever

METRIC_COLS = ["P@1", "P@5", "P@10", "R@10", "R@50", "R@100", "MRR@10", "MAP@100", "NDCG@10"]


def measure_latency(
    retriever: Retriever,
    queries: dict[str, str],
    k: int = 10,
    warmup: int = 3,
    samples: int = 20,
) -> float:
    """Average single-query wall-clock time in ms (warmup excluded)."""
    qids = list(queries.keys())
    for qid in qids[:warmup]:
        retriever.search(queries[qid], k)
    times = []
    for qid in qids[warmup: warmup + samples]:
        t0 = time.perf_counter()
        retriever.search(queries[qid], k)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times) if times else 0.0


def run_retriever(
    name: str,
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 100,
) -> tuple[dict, dict]:
    run: dict[str, dict[str, float]] = {}
    for qid, query in queries.items():
        results = retriever.search(query, k)
        run[qid] = {did: score for did, score in results}

    metrics = evaluate(qrels, run)
    metrics["Latency_ms"] = measure_latency(retriever, queries)
    metrics["method"] = name
    return metrics, run


def save_per_query(
    method_name: str,
    dataset_name: str,
    run: dict[str, dict[str, float]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    results_dir: Path,
    k: int = 100,
) -> None:
    rows = []
    for qid, scores in run.items():
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        for rank, (did, score) in enumerate(ranked, start=1):
            rows.append({
                "query_id": qid,
                "query_text": queries.get(qid, ""),
                "rank": rank,
                "doc_id": did,
                "score": score,
                "is_relevant": int(qrels.get(qid, {}).get(did, 0) > 0),
            })
    df = pd.DataFrame(rows)
    out = results_dir / "per_query" / f"{method_name}_{dataset_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def bold_best_md(df: pd.DataFrame) -> str:
    """Render DataFrame as Markdown table with best value per column bolded."""
    metric_cols = [c for c in df.columns if c != "method"]
    lower_is_better = {"Latency_ms"}

    best = {}
    for col in metric_cols:
        best[col] = df[col].min() if col in lower_is_better else df[col].max()

    lines = []
    lines.append("| Method | " + " | ".join(metric_cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(metric_cols)) + "|")

    for _, row in df.iterrows():
        cells = []
        for col in metric_cols:
            val = row[col]
            fmt = f"{val:.1f}" if col == "Latency_ms" else f"{val:.4f}"
            if abs(val - best[col]) < 1e-9:
                fmt = f"**{fmt}**"
            cells.append(fmt)
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")

    return "\n".join(lines)
