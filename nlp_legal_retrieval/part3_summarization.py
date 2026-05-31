from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/summarization_results.json")
DEVICE = 0 if torch.cuda.is_available() else -1
N = 100


def run_summarization(examples):
    summarizer = hf_pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=DEVICE,
    )
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    results = []
    for i, ex in enumerate(tqdm(examples, desc="Summarization")):
        article = ex["article"]
        reference = ex["highlights"]

        # BART max input is 1024 tokens; 3000 chars is a safe character-level cutoff
        summary = summarizer(
            article[:3000], max_length=130, min_length=30, do_sample=False
        )[0]["summary_text"]

        scores = scorer.score(reference, summary)

        results.append({
            "index": i,
            "article_snippet": article[:200],
            "summary": summary,
            "reference": reference,
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        })

    return results


def main():
    CACHE_PATH.parent.mkdir(exist_ok=True)

    results = load_cache(CACHE_PATH)
    if results is None:
        ds = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split="test",
            streaming=True,
        )
        examples = list(ds.take(N))
        results = run_summarization(examples)
        save_cache(CACHE_PATH, results)
        print(f"Cached {len(results)} results to {CACHE_PATH}")
    else:
        print(f"Loaded {len(results)} cached results from {CACHE_PATH}")

    headers = ["#", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Summary (80 chars)"]
    rows = [
        [r["index"], r["rouge1"], r["rouge2"], r["rougeL"], r["summary"][:80]]
        for r in results[:20]
    ]
    print("\n=== Summary Table (first 20 examples) ===")
    print_table(rows, headers)

    mean_r1 = sum(r["rouge1"] for r in results) / len(results)
    mean_r2 = sum(r["rouge2"] for r in results) / len(results)
    mean_rL = sum(r["rougeL"] for r in results) / len(results)
    print(
        f"\nMean ROUGE-1: {mean_r1:.4f}  |  ROUGE-2: {mean_r2:.4f}  |  ROUGE-L: {mean_rL:.4f}"
    )

    # Two lowest ROUGE-1 examples — prime candidates for hallucination annotation
    worst = sorted(results, key=lambda x: x["rouge1"])[:2]
    print("\n=== 2 Lowest ROUGE-1 Examples (Hallucination Candidates) ===")
    for r in worst:
        print(
            f"\n--- Example #{r['index']} | R1={r['rouge1']}  R2={r['rouge2']}  RL={r['rougeL']} ---"
        )
        print(f"Article (200): {r['article_snippet']}")
        print(f"Reference    : {r['reference']}")
        print(f"Summary      : {r['summary']}")


if __name__ == "__main__":
    main()
