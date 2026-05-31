from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
from bert_score import score as bert_score_fn

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/translation_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 100


def run_translation(examples):
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(DEVICE)
    model.eval()

    results = []
    for i, ex in enumerate(tqdm(examples, desc="Translation")):
        src = ex["translation"]["en"]
        ref = ex["translation"]["de"]

        inputs = tokenizer(
            src, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            translated = model.generate(**inputs)
        hyp = tokenizer.decode(translated[0], skip_special_tokens=True)

        chrf_sent = sacrebleu.sentence_chrf(hyp, [ref]).score

        results.append({
            "index": i,
            "source_en": src,
            "hypothesis_de": hyp,
            "reference_de": ref,
            "chrf_sentence": round(chrf_sent, 2),
        })

    return results


def main():
    CACHE_PATH.parent.mkdir(exist_ok=True)

    results = load_cache(CACHE_PATH)
    if results is None:
        ds = load_dataset(
            "wmt/wmt19",
            "de-en",
            split="validation",
            streaming=True,
        )
        examples = list(ds.take(N))
        results = run_translation(examples)
        save_cache(CACHE_PATH, results)
        print(f"Cached {len(results)} results to {CACHE_PATH}")
    else:
        print(f"Loaded {len(results)} cached results from {CACHE_PATH}")

    hypotheses = [r["hypothesis_de"] for r in results]
    references = [r["reference_de"] for r in results]

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    _, _, F1 = bert_score_fn(hypotheses, references, lang="de", verbose=False)
    bs_f1 = F1.mean().item()

    headers = ["#", "chrF", "Hypothesis (60 chars)", "Reference (60 chars)"]
    rows = [
        [
            r["index"], r["chrf_sentence"],
            r["hypothesis_de"][:60], r["reference_de"][:60],
        ]
        for r in results[:20]
    ]
    print("\n=== Summary Table (first 20 examples) ===")
    print_table(rows, headers)

    print(f"\nCorpus BLEU: {bleu:.2f}  |  chrF: {chrf:.2f}  |  BERTScore F1: {bs_f1:.4f}")

    worst = sorted(results, key=lambda x: x["chrf_sentence"])[:3]
    print("\n=== Worst 3 Examples by chrF ===")
    for r in worst:
        print(f"\n--- Example #{r['index']} | chrF={r['chrf_sentence']} ---")
        print(f"Source : {r['source_en']}")
        print(f"Ref    : {r['reference_de']}")
        print(f"Hyp    : {r['hypothesis_de']}")


if __name__ == "__main__":
    main()
