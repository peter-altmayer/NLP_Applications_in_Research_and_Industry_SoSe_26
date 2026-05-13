"""Generate per-config samples and curated failure cases for the README.

Writes:
- results/samples_per_config.md (25 rows per config)
- results/failure_cases.md (10 diverse failure cases across configs)
- results/findings.md (model × retriever × k comparison)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def truncate(s, n=180):
    s = str(s).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def first_gold(gold_json):
    try:
        arr = json.loads(gold_json)
        return arr[0] if arr else ""
    except Exception:
        return str(gold_json)


def samples_md(df: pd.DataFrame) -> str:
    out = ["# 25-Row Samples per Configuration\n"]
    out.append(
        "Each section shows 25 randomly sampled rows (seed=42) from one "
        "(model, dataset, retrieval, k) configuration. Long strings truncated "
        "to ~180 chars for readability.\n"
    )
    groups = df.groupby(["model", "dataset", "retrieval_method", "k"], sort=True)
    for (model, dataset, retr, k), g in groups:
        sample = g.sample(n=min(25, len(g)), random_state=42).reset_index(drop=True)
        em_mean = sample["em"].mean()
        f1_mean = sample["f1"].mean()
        out.append(
            f"\n## {model} · {dataset} · {retr} · k={k}  "
            f"(sample EM={em_mean:.3f}, F1={f1_mean:.3f})\n"
        )
        out.append("| # | Question | Gold (first) | Prediction | EM | F1 |")
        out.append("|---|---|---|---|---|---|")
        for i, row in sample.iterrows():
            out.append(
                "| {i} | {q} | {g} | {p} | {em} | {f1:.2f} |".format(
                    i=i + 1,
                    q=truncate(row["question"], 90),
                    g=truncate(first_gold(row["gold_answers"]), 120),
                    p=truncate(row["prediction"], 90),
                    em=int(row["em"]),
                    f1=row["f1"],
                )
            )
    return "\n".join(out) + "\n"


def pick_failure_cases(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Pick 10 failure cases (EM=0) that span configs and failure modes."""
    fails = df[df["em"] == 0].copy()
    picks = []
    seen_keys = set()
    for _, row in fails.sort_values("f1").iterrows():
        key = (row["model"], row["dataset"], row["retrieval_method"], row["k"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        picks.append(row)
        if len(picks) >= n:
            break
    return pd.DataFrame(picks).reset_index(drop=True)


def failure_md(df: pd.DataFrame) -> str:
    cases = pick_failure_cases(df, 10)
    out = ["# 10 Failure Cases — Stratified Across Configurations\n"]
    out.append(
        "Each case is the lowest-F1 EM=0 row from a distinct "
        "(model, dataset, retrieval, k) combination, to surface a range of "
        "failure modes rather than concentrating on a single weak config.\n"
    )
    for i, row in cases.iterrows():
        gold = json.loads(row["gold_answers"])
        retrieved = json.loads(row["retrieved_docs"]) if row["retrieved_docs"] else []
        out.append(
            f"\n## Case {i + 1} — {row['model']} · {row['dataset']} · "
            f"{row['retrieval_method']} · k={row['k']}"
        )
        out.append(f"- **Question:** {row['question']}")
        out.append(f"- **Gold (first):** {truncate(gold[0] if gold else '', 300)}")
        out.append(f"- **Prediction:** `{truncate(row['prediction'], 200)}`")
        if retrieved:
            out.append(f"- **Top retrieved doc:** {truncate(retrieved[0], 300)}")
        else:
            out.append("- **Top retrieved doc:** _(none — closed-book run)_")
        out.append(f"- **EM=0, F1={row['f1']:.3f}**")
    return "\n".join(out) + "\n"


def findings_md(df: pd.DataFrame) -> str:
    agg = (
        df.groupby(["model", "dataset", "retrieval_method", "k"])
        .agg(em=("em", "mean"), f1=("f1", "mean"), n=("em", "size"))
        .round(3)
        .reset_index()
    )
    out = ["# Findings — Model × Retriever × K\n"]
    out.append("## Full results (mean EM / mean F1)\n")
    out.append("| model | dataset | retrieval | k | EM | F1 | N |")
    out.append("|---|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        out.append(
            f"| {r['model']} | {r['dataset']} | {r['retrieval_method']} | "
            f"{r['k']} | {r['em']:.3f} | {r['f1']:.3f} | {int(r['n'])} |"
        )
    return "\n".join(out) + "\n"


def main():
    csv = RESULTS / "all_results.csv"
    print(f"Loading {csv}")
    df = pd.read_csv(csv)
    print(f"  {len(df)} rows")

    (RESULTS / "samples_per_config.md").write_text(samples_md(df), encoding="utf-8")
    print("  wrote samples_per_config.md")
    (RESULTS / "failure_cases.md").write_text(failure_md(df), encoding="utf-8")
    print("  wrote failure_cases.md")
    (RESULTS / "findings.md").write_text(findings_md(df), encoding="utf-8")
    print("  wrote findings.md")


if __name__ == "__main__":
    main()
