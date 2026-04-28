"""
Decoding experiment: temperature sweep on google/flan-t5-large.
Runs the same 10 examples at temperature 0.3, 0.7, and 1.0
to demonstrate how generation behaviour changes with temperature.

Usage:
    python experiments/run_decoding.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_processed
from src.model import run_generative

MODEL_NAME = "google/flan-t5-large"
TEMPERATURES = [0.3, 0.7, 1.0]
N_EXAMPLES = 10

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    examples = load_processed().head(N_EXAMPLES).to_dict(orient="records")
    outputs_by_temp: dict[float, list[str]] = {}

    for temp in TEMPERATURES:
        print(f"\nRunning temperature={temp}...")
        results = run_generative(
            MODEL_NAME,
            examples,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp,
        )
        outputs_by_temp[temp] = [r["predicted_answer"] for r in results]

    rows = []
    for i, ex in enumerate(examples):
        row = {
            "question": ex["question"],
            "context": ex["context"],
        }
        for temp in TEMPERATURES:
            row[f"temp_{temp}"] = outputs_by_temp[temp][i]
        rows.append(row)

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "decoding_results.csv"
    df.to_csv(path, index=False)
    print(f"\nDecoding experiment saved to {path}")


if __name__ == "__main__":
    main()
