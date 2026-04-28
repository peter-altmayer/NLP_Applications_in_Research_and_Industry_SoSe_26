import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "qa_pairs.csv"
RAW_PATH = RAW_DIR / "pubmed_qa_raw.json"
N_EXAMPLES = 25


def download_and_process() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading pubmed_qa (pqa_labeled)...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    subset = dataset.select(range(N_EXAMPLES))

    raw_records = [dict(subset[i]) for i in range(len(subset))]
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)
    print(f"Raw data saved to {RAW_PATH}")

    records = []
    for example in subset:
        context = " ".join(example["context"]["contexts"])
        records.append({
            "id": example["pubid"],
            "question": example["question"],
            "context": context,
            "reference_answer": example["long_answer"],
        })

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH} ({len(df)} examples)")
    return df


def load_processed() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        return download_and_process()
    return pd.read_csv(PROCESSED_PATH)


if __name__ == "__main__":
    download_and_process()
