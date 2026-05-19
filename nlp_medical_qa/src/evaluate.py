"""Evaluation metrics: Exact Match, Token-F1, ROUGE-L, BERTScore, faithfulness (NLI)."""
import re
import string
from typing import List, Optional

from rouge_score import rouge_scorer as _rouge_scorer

_ROUGE = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Loaded lazily to avoid importing heavy models at import time
_nli_pipe = None
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


# ---------------------------------------------------------------------------
# Text normalisation (SQuAD-style)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def exact_match(pred: str, gold: str) -> float:
    return float(_normalise(pred) == _normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalise(pred).split()
    gold_toks = _normalise(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    if not pred.strip() or not gold.strip():
        return 0.0
    return _ROUGE.score(gold, pred)["rougeL"].fmeasure


def bertscore_batch(preds: List[str], golds: List[str], lang: str = "en") -> List[float]:
    """Semantic similarity via sentence-transformers cosine similarity.

    bert_score>=0.3.13 calls tokenizer.build_inputs_with_special_tokens() which was
    removed from transformers>=4.40, making it unusable. We use all-MiniLM-L6-v2
    (already required by the dense retriever) as a drop-in replacement.
    """
    if not preds:
        return []
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    pred_embs = _model.encode(preds, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    gold_embs = _model.encode(golds, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    scores = (pred_embs * gold_embs).sum(axis=1)
    return [round(float(s), 4) for s in scores.tolist()]


def faithfulness(answer: str, context: str) -> Optional[float]:
    """NLI-based faithfulness: probability that context entails the answer.

    Returns None when context is empty (not applicable for extractive QA
    with empty retrieval).
    """
    global _nli_pipe
    if not context.strip() or not answer.strip():
        return None

    if _nli_pipe is None:
        from transformers import pipeline
        _nli_pipe = pipeline(
            "text-classification",
            model=NLI_MODEL,
            device=-1,      # CPU; change to 0 on Colab GPU
            truncation=True,
            max_length=512,
        )

    # cross-encoder NLI: premise=context, hypothesis=answer
    inp = f"{context[:400]} [SEP] {answer[:100]}"
    result = _nli_pipe(inp)[0]
    label = result["label"].lower()
    score = result["score"]

    if "entail" in label:
        return round(score, 4)
    elif "contradict" in label:
        return round(1.0 - score, 4)
    else:
        return 0.5   # neutral


# ---------------------------------------------------------------------------
# Per-sample evaluation bundle
# ---------------------------------------------------------------------------

def evaluate_sample(
    pred: str,
    gold: str,
    context: str = "",
    skip_faithfulness: bool = False,
) -> dict:
    return {
        "em": exact_match(pred, gold),
        "f1": round(token_f1(pred, gold), 4),
        "rouge_l": round(rouge_l(pred, gold), 4),
        "faithfulness": None if skip_faithfulness else faithfulness(pred, context),
    }
