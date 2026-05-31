import json
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/ocr_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 100


def extract_gold_text(ground_truth_str: str) -> str:
    """Recursively collect all string values from the CORD-v2 gt_parse dict."""
    gt = json.loads(ground_truth_str)

    def collect(obj):
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                out.extend(collect(v))
            return out
        if isinstance(obj, list):
            out = []
            for item in obj:
                out.extend(collect(item))
            return out
        return []

    return " ".join(collect(gt.get("gt_parse", gt)))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_ocr(examples):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed"
    ).to(DEVICE)
    model.eval()

    results = []
    for i, ex in enumerate(tqdm(examples, desc="OCR")):
        image = ex["image"].convert("RGB")
        gold = extract_gold_text(ex["ground_truth"])

        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(DEVICE)

        with torch.no_grad():
            ids = model.generate(pixel_values)
        ocr_text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        cer = jiwer.cer(gold, ocr_text)
        wer = jiwer.wer(gold, ocr_text)
        f1 = token_f1(ocr_text, gold)

        results.append({
            "index": i,
            "ocr_text": ocr_text,
            "gold_text": gold,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "f1": round(f1, 4),
        })

    return results


def main():
    CACHE_PATH.parent.mkdir(exist_ok=True)

    results = load_cache(CACHE_PATH)
    if results is None:
        ds = load_dataset(
            "naver-clova-ix/cord-v2",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        examples = list(ds.take(N))
        results = run_ocr(examples)
        save_cache(CACHE_PATH, results)
        print(f"Cached {len(results)} results to {CACHE_PATH}")
    else:
        print(f"Loaded {len(results)} cached results from {CACHE_PATH}")

    headers = ["#", "CER", "WER", "F1", "OCR (40 chars)", "Gold (40 chars)"]
    rows = [
        [
            r["index"], r["cer"], r["wer"], r["f1"],
            r["ocr_text"][:40], r["gold_text"][:40],
        ]
        for r in results[:20]
    ]
    print("\n=== Summary Table (first 20 examples) ===")
    print_table(rows, headers)

    mean_cer = sum(r["cer"] for r in results) / len(results)
    mean_wer = sum(r["wer"] for r in results) / len(results)
    mean_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\nMean CER: {mean_cer:.4f}  |  Mean WER: {mean_wer:.4f}  |  Mean F1: {mean_f1:.4f}")

    worst = sorted(results, key=lambda x: x["cer"], reverse=True)[:3]
    print("\n=== Worst 3 Examples by CER ===")
    for r in worst:
        print(f"\n--- Example #{r['index']} | CER={r['cer']}  WER={r['wer']}  F1={r['f1']} ---")
        print(f"Gold : {r['gold_text']}")
        print(f"OCR  : {r['ocr_text']}")


if __name__ == "__main__":
    main()
