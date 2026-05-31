import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer

from utils import save_cache, load_cache, print_table

CACHE_PATH = Path("results/ocr_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 100


def extract_lines_with_boxes(ground_truth_str: str) -> list:
    """Extract (bounding_box, gold_text) pairs from CORD-v2 valid_line annotations."""
    gt = json.loads(ground_truth_str)
    lines = []
    for line in gt.get("valid_line", []):
        words = line.get("words", [])
        if not words:
            continue
        all_x, all_y, texts = [], [], []
        for word in words:
            q = word.get("quad", {})
            all_x += [q.get("x1", 0), q.get("x2", 0), q.get("x3", 0), q.get("x4", 0)]
            all_y += [q.get("y1", 0), q.get("y2", 0), q.get("y3", 0), q.get("y4", 0)]
            t = word.get("text", "").strip()
            if t:
                texts.append(t)
        if all_x and all_y and texts:
            lines.append({
                "box": (min(all_x), min(all_y), max(all_x), max(all_y)),
                "text": " ".join(texts),
            })
    return lines


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
        lines = extract_lines_with_boxes(ex["ground_truth"])
        if not lines:
            continue

        ocr_parts = []
        for line_info in lines:
            x1, y1, x2, y2 = line_info["box"]
            # Small padding so characters at the crop edge aren't cut off
            pad = 4
            crop = image.crop((
                max(0, x1 - pad), max(0, y1 - pad),
                min(image.width, x2 + pad), min(image.height, y2 + pad),
            ))
            pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)
            with torch.no_grad():
                ids = model.generate(pixel_values, max_new_tokens=128)
            ocr_parts.append(processor.batch_decode(ids, skip_special_tokens=True)[0])

        ocr_text = " ".join(ocr_parts)
        gold_text = " ".join(line["text"] for line in lines)

        cer = jiwer.cer(gold_text, ocr_text)
        wer = jiwer.wer(gold_text, ocr_text)
        f1 = token_f1(ocr_text, gold_text)

        results.append({
            "index": i,
            "ocr_text": ocr_text,
            "gold_text": gold_text,
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
