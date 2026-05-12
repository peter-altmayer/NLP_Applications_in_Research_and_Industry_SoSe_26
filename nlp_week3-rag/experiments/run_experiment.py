import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import generate_answers, load_generator
from src.retriever import load_bm25, load_embeddings, retrieve_bm25, retrieve_dpr

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

GENERATOR_MODELS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

# Models that need 4-bit quantization on 16 GB GPUs (Colab T4)
QUANTIZE_4BIT = {"qwen"}

DPR_Q_ENCODERS = {
    "dpr_nq": "facebook/dpr-question_encoder-single-nq-base",
    "dpr_multiset": "facebook/dpr-question_encoder-multiset-base",
}

K_NONZERO = [1, 5, 10]


def _save(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {len(rows)} rows → {path.name}")


def _make_row(args, retrieval_method, k, rec, pred):
    return {
        "dataset": args.dataset,
        "model": args.model,
        "retrieval_method": retrieval_method,
        "k": k,
        "question": rec["question"],
        "gold_answers": json.dumps(rec["gold_answers"]),
        "retrieved_docs": json.dumps(rec["context_docs"]),
        "prediction": pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(GENERATOR_MODELS))
    parser.add_argument("--dataset", required=True, choices=["trivia_qa", "natural_questions"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset}")

    with open(PROCESSED / f"{args.dataset}_corpus.json", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    corpus = data["corpus"]

    bm25 = load_bm25(PROCESSED / f"{args.dataset}_bm25.pkl")
    embeddings = {k: load_embeddings(PROCESSED / f"{args.dataset}_{k}.npy") for k in DPR_Q_ENCODERS}

    # Load DPR question encoders on CPU to preserve VRAM for the generator
    dpr_encoders = {}
    for enc_key, enc_name in DPR_Q_ENCODERS.items():
        print(f"Loading {enc_key} question encoder (cpu)...")
        tok = DPRQuestionEncoderTokenizer.from_pretrained(enc_name)
        enc = DPRQuestionEncoder.from_pretrained(enc_name)
        enc.eval()
        dpr_encoders[enc_key] = (enc, tok)

    print(f"Loading generator: {GENERATOR_MODELS[args.model]}")
    gen_model, gen_tokenizer = load_generator(
        GENERATOR_MODELS[args.model],
        quantize_4bit=args.model in QUANTIZE_4BIT,
    )

    # ── K=0: no context, run once ──────────────────────────────────────────
    k0_path = RESULTS / f"{args.model}_{args.dataset}_none_k0.csv"
    if k0_path.exists():
        print("K=0: skipping (already exists)")
    else:
        print("\nK=0 (no context)...")
        gen_records = [
            {"question": r["question"], "gold_answers": r["answers"], "context_docs": []}
            for r in records
        ]
        preds = generate_answers(gen_model, gen_tokenizer, gen_records, batch_size=args.batch_size)
        _save([_make_row(args, "none", 0, r, p) for r, p in zip(gen_records, preds)], k0_path)

    # ── K > 0 ──────────────────────────────────────────────────────────────
    for retrieval_key in ["bm25", "dpr_nq", "dpr_multiset"]:
        for k in K_NONZERO:
            out_path = RESULTS / f"{args.model}_{args.dataset}_{retrieval_key}_k{k}.csv"
            if out_path.exists():
                print(f"Skipping {out_path.name}")
                continue

            print(f"\n{retrieval_key} k={k}...")
            gen_records = []
            for rec in records:
                if retrieval_key == "bm25":
                    docs = retrieve_bm25(bm25, corpus, rec["question"], k)
                else:
                    enc, tok = dpr_encoders[retrieval_key]
                    docs = retrieve_dpr(embeddings[retrieval_key], corpus, enc, tok, rec["question"], k, device="cpu")
                gen_records.append({
                    "question": rec["question"],
                    "gold_answers": rec["answers"],
                    "context_docs": docs,
                })

            preds = generate_answers(gen_model, gen_tokenizer, gen_records, batch_size=args.batch_size)
            _save([_make_row(args, retrieval_key, k, r, p) for r, p in zip(gen_records, preds)], out_path)


if __name__ == "__main__":
    main()
