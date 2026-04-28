"""
Generative QA baseline.
Runs google/flan-t5-base and google/flan-t5-large
on 25 PubMedQA examples and writes per-model + merged result CSVs.

Usage:
    python experiments/run_generative.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_processed
from src.model import run_generative

MODELS = {
    "flan_t5_base": "google/flan-t5-base",
    "flan_t5_large": "google/flan-t5-large",
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    examples = load_processed().to_dict(orient="records")
    all_dfs = []

    for short_name, model_name in MODELS.items():
        print(f"\nRunning {short_name} ({model_name})...")
        results = run_generative(model_name, examples, max_new_tokens=128)
        df = pd.DataFrame(results)

        path = RESULTS_DIR / f"generative_{short_name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")

        df.insert(0, "model", short_name)
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(RESULTS_DIR / "generative_all.csv", index=False)
    print(f"\nMerged results saved to {RESULTS_DIR / 'generative_all.csv'}")


if __name__ == "__main__":
    main()
