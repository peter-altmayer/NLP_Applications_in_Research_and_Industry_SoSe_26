import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import extract_corpus, load_dataset_records
from src.retriever import build_bm25, build_dpr_embeddings, save_bm25, save_embeddings

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc.nocontext"),
    "natural_questions": ("sentence-transformers/natural-questions", "train"),
}

DPR_CTX_ENCODERS = {
    "dpr_nq": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr_multiset": "facebook/dpr-ctx_encoder-multiset-base",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds_key, (hf_path, split) in DATASETS.items():
        print(f"\n=== {ds_key} ===")
        records = load_dataset_records(hf_path, split, n=1000)
        corpus = extract_corpus(records)
        print(f"  {len(records)} records | {len(corpus)} unique corpus docs")

        corpus_path = PROCESSED / f"{ds_key}_corpus.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump({"records": records, "corpus": corpus}, f, ensure_ascii=False)
        print(f"  Corpus saved → {corpus_path.name}")

        bm25 = build_bm25(corpus)
        save_bm25(bm25, PROCESSED / f"{ds_key}_bm25.pkl")
        print("  BM25 index saved")

        for enc_key, enc_name in DPR_CTX_ENCODERS.items():
            out_path = PROCESSED / f"{ds_key}_{enc_key}.npy"
            if out_path.exists():
                print(f"  {enc_key}: already exists — skipping")
                continue
            print(f"  Building {enc_key} embeddings ({enc_name})...")
            embeddings = build_dpr_embeddings(corpus, enc_name, device=device)
            save_embeddings(embeddings, out_path)
            print(f"  {enc_key}: saved shape {embeddings.shape} → {out_path.name}")


if __name__ == "__main__":
    main()
