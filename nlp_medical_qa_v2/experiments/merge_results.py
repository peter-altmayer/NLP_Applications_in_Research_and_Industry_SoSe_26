"""Aggregate per-run CSVs into a summary table and failure analysis.

Usage
-----
    python experiments/merge_results.py [--results-dir results/] [--top-n 5]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = Path(__file__).parent.parent / "results"

METRIC_COLS = ["em", "f1", "rouge_l", "bertscore", "faithfulness",
               "precision_at_k", "recall_at_k"]

MODEL_ORDER = ["v1", "v2", "v3", "v4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_csvs(paths) -> list[dict]:
    rows = []
    for csv_path in sorted(paths):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                for m in METRIC_COLS:
                    r[m] = _safe_float(r.get(m))
                rows.append(r)
    return rows


def _load_all(results_dir: Path, include_oracle: bool = False) -> list[dict]:
    paths = [
        p for p in sorted(results_dir.glob("*.csv"))
        if "summary" not in p.name and "oracle_baseline" not in p.name
        and (include_oracle or "_oracle_" not in p.name)
    ]
    return _load_csvs(paths)


def _group(rows: list[dict]) -> dict:
    """key = (model, dataset, retrieval, k)."""
    groups: dict = defaultdict(list)
    for r in rows:
        key = (r["model"], r["dataset"], r["retrieval"], r["k"])
        groups[key].append(r)
    return dict(groups)


def _mean(rows: list[dict], col: str) -> str:
    vals = [r[col] for r in rows if r[col] is not None]
    if not vals:
        return "—"
    return f"{sum(vals) / len(vals):.3f}"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(groups: dict, datasets: list[str]) -> None:
    for dataset in datasets:
        print(f"\n{'─'*80}")
        print(f"  Dataset: {dataset}")
        print(f"{'─'*80}")

        # Collect all (retrieval, k) combos for this dataset
        combos = sorted({
            (retrieval, k)
            for (m, ds, retrieval, k) in groups
            if ds == dataset
        })

        for retrieval, k in combos:
            print(f"\n  Retrieval: {retrieval}  k={k}")
            header = (
                f"  {'Model':<6}  {'N':>5}  "
                f"{'EM':>6}  {'F1':>6}  {'ROUGE-L':>7}  "
                f"{'BERTScore':>9}  {'Faith.':>6}  "
                f"{'P@k':>6}  {'R@k':>6}"
            )
            print(header)
            print("  " + "─" * (len(header) - 2))

            for model in MODEL_ORDER:
                key = (model, dataset, retrieval, k)
                if key not in groups:
                    continue
                rows = groups[key]
                print(
                    f"  {model:<6}  {len(rows):>5}  "
                    f"{_mean(rows,'em'):>6}  {_mean(rows,'f1'):>6}  "
                    f"{_mean(rows,'rouge_l'):>7}  "
                    f"{_mean(rows,'bertscore'):>9}  "
                    f"{_mean(rows,'faithfulness'):>6}  "
                    f"{_mean(rows,'precision_at_k'):>6}  "
                    f"{_mean(rows,'recall_at_k'):>6}"
                )


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def print_failures(groups: dict, top_n: int) -> None:
    print(f"\n{'='*80}")
    print(f"  FAILURE ANALYSIS  —  {top_n} worst samples per model×dataset")
    print(f"{'='*80}")

    # Aggregate all rows by (model, dataset)
    by_model_ds: dict = defaultdict(list)
    for (model, dataset, retrieval, k), rows in groups.items():
        for r in rows:
            by_model_ds[(model, dataset)].append(r)

    for (model, dataset), rows in sorted(by_model_ds.items()):
        # Sort by F1 ascending (worst first)
        worst = sorted(
            [r for r in rows if r["f1"] is not None],
            key=lambda r: r["f1"],
        )[:top_n]

        if not worst:
            continue

        print(f"\n  {model.upper()} / {dataset}  ({len(rows)} total samples)")
        for i, r in enumerate(worst, 1):
            print(f"  [{i}] Q:    {r['question'][:90]}")
            print(f"       Gold: {r['gold_answer'][:60]}")
            print(f"       Pred: {r['predicted_answer'][:60]}")
            print(f"       F1={r['f1']:.3f}  EM={r['em']}  Faith={r['faithfulness']}")
            print()


# ---------------------------------------------------------------------------
# Write aggregated CSV
# ---------------------------------------------------------------------------

def write_summary_csv(groups: dict, out_path: Path) -> None:
    rows_out = []
    for (model, dataset, retrieval, k), rows in groups.items():
        row = {
            "model": model, "dataset": dataset,
            "retrieval": retrieval, "k": k, "n": len(rows),
        }
        for m in METRIC_COLS:
            row[f"mean_{m}"] = _mean(rows, m)
        rows_out.append(row)

    rows_out.sort(key=lambda r: (r["dataset"], r["retrieval"], MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSummary CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# Oracle baseline summary
# ---------------------------------------------------------------------------

def write_oracle_summary(results_dir: Path) -> None:
    oracle_paths = [
        p for p in sorted(results_dir.glob("*.csv"))
        if "_oracle_" in p.name and "summary" not in p.name
    ]
    if not oracle_paths:
        print("No oracle result CSVs found (files matching *_oracle_*.csv).")
        return

    print(f"\nLoading {len(oracle_paths)} oracle result file(s) …")
    rows = _load_csvs(oracle_paths)
    groups = _group(rows)
    datasets = sorted({ds for (_, ds, _, _) in groups})

    print(f"\n{'='*80}")
    print("  ORACLE RETRIEVAL BASELINE")
    print(f"{'='*80}")
    print_summary(groups, datasets)

    out_path = results_dir / "oracle_baseline.csv"
    write_summary_csv(groups, out_path)
    print(f"Oracle baseline CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge results CSVs into summary table")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--top-n", type=int, default=3, help="Failure examples per group")
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--oracle-summary", action="store_true",
                        help="Print oracle baseline table and write oracle_baseline.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.oracle_summary:
        write_oracle_summary(results_dir)
        return

    # Normal summary — excludes oracle runs
    rows = _load_all(results_dir, include_oracle=False)
    if not rows:
        sys.exit(f"No (non-oracle) result CSVs found in {results_dir}. Run experiments first.")

    print(f"Total samples loaded: {len(rows)}")
    groups = _group(rows)
    datasets = sorted({ds for (_, ds, _, _) in groups})

    print_summary(groups, datasets)

    if not args.no_failures:
        print_failures(groups, top_n=args.top_n)

    summary_path = results_dir / "summary.csv"
    write_summary_csv(groups, summary_path)


if __name__ == "__main__":
    main()
