import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import build_results_table, exact_match, token_f1

RESULTS = Path(__file__).parent.parent / "results"


def main():
    csvs = sorted(p for p in RESULTS.glob("*.csv") if p.name != "all_results.csv")
    if not csvs:
        print("No CSVs found in results/ — run run_experiment.py first")
        return

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csvs)} files")

    df["em"] = df.apply(
        lambda r: int(exact_match(str(r["prediction"]), json.loads(r["gold_answers"]))), axis=1
    )
    df["f1"] = df.apply(
        lambda r: token_f1(str(r["prediction"]), json.loads(r["gold_answers"])), axis=1
    )

    df.to_csv(RESULTS / "all_results.csv", index=False)
    print("Saved all_results.csv\n")

    table = build_results_table(df)
    print("=== Results (mean EM / mean F1) ===")
    print(table.to_string())

    print("\n=== Top 10 Failure Cases (EM=0, lowest F1) ===")
    failures = df[df["em"] == 0].nsmallest(10, "f1")
    for _, row in failures.iterrows():
        gold = json.loads(row["gold_answers"])
        retrieved = json.loads(row["retrieved_docs"])
        print(
            f"\n[{row['dataset']} | {row['model']} | {row['retrieval_method']} | K={row['k']}]"
        )
        print(f"  Q:    {row['question']}")
        print(f"  Gold: {gold[:3]}")
        print(f"  Pred: {row['prediction']}")
        if retrieved:
            print(f"  Ret:  {retrieved[0][:120]}")
        print(f"  EM={row['em']}  F1={row['f1']:.3f}")


if __name__ == "__main__":
    main()
