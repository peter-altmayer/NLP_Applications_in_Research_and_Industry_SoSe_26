"""Dataset loading and preprocessing for medical QA: BioASQ, PubMedQA, SQuAD."""
import json
import random
from pathlib import Path
from typing import Optional

BIOASQ_JSON = (
    Path(__file__).parent.parent
    / "data" / "raw" / "BioASQ-training13b" / "training13b.json"
)

SEED = 42

# Corpus subset sizes used for indexing and evaluation
SUBSET = {"bioasq": 500, "pubmedqa": 500, "squad": 500}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _flat_answer(raw) -> str:
    """Flatten BioASQ exact_answer to a single string."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):          # list-type: [["a"], ["b"]]
            return ", ".join(item[0] for item in raw if item)
        return raw[0].strip() if len(raw) == 1 else ", ".join(str(x) for x in raw)
    return str(raw).strip()


def _first_str(raw) -> str:
    """Return first element if list, else the string itself."""
    if isinstance(raw, list):
        return raw[0].strip() if raw else ""
    return str(raw).strip()


# ---------------------------------------------------------------------------
# BioASQ
# ---------------------------------------------------------------------------

def load_bioasq(
    path: Path = BIOASQ_JSON,
    max_samples: Optional[int] = None,
    question_types: Optional[list] = None,
    split_ratio: float = 0.8,
    split: str = "test",
) -> list:
    """Load BioASQ Task B from the local JSON file.

    Returns list of dicts with keys:
        id, question, context, answer, ideal_answer, answer_type, snippets
    """
    if question_types is None:
        question_types = ["factoid", "yesno"]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for q in data["questions"]:
        if q["type"] not in question_types:
            continue

        exact = _flat_answer(q.get("exact_answer", ""))
        ideal = _first_str(q.get("ideal_answer", ""))
        snippets = [s["text"] for s in q.get("snippets", []) if s.get("text")]

        # Context = up to 3 snippets joined (fits 512-token window)
        context = " ".join(snippets[:3])

        samples.append({
            "id": q["id"],
            "question": q["body"],
            "context": context,
            "answer": exact,
            "ideal_answer": ideal,
            "answer_type": q["type"],
            "snippets": snippets,
        })

    rng = random.Random(SEED)
    rng.shuffle(samples)

    cut = int(len(samples) * split_ratio)
    samples = samples[:cut] if split == "train" else samples[cut:]

    if max_samples:
        samples = samples[:max_samples]

    return samples


def load_bioasq_train_sft(path: Path = BIOASQ_JSON) -> list:
    """Return yesno samples formatted for fine-tuning (answer always in context)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for q in data["questions"]:
        if q["type"] != "yesno":
            continue
        ideal = _first_str(q.get("ideal_answer", ""))
        exact = _flat_answer(q.get("exact_answer", "")).lower()
        if not ideal or exact not in ("yes", "no"):
            continue
        # Prepend the label so it is always a verbatim span
        context = exact.capitalize() + ". " + ideal
        samples.append({
            "question": q["body"],
            "context": context,
            "answer": exact.capitalize(),   # "Yes" or "No"
        })

    return samples


# ---------------------------------------------------------------------------
# PubMedQA
# ---------------------------------------------------------------------------

def load_pubmedqa(max_samples: Optional[int] = None, split: str = "test") -> list:
    """Load PubMedQA labeled split from HuggingFace (pqa_labeled)."""
    from datasets import load_dataset  # lazy import — not needed at index time

    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)

    all_samples = []
    for item in ds:
        ctx_parts = item["context"]["contexts"]
        context = " ".join(ctx_parts) if isinstance(ctx_parts, list) else ctx_parts
        all_samples.append({
            "id": str(item["pubid"]),
            "question": item["question"],
            "context": context[:2000],
            "answer": item["final_decision"],        # yes / no / maybe
            "ideal_answer": item["long_answer"],
            "answer_type": "yesno",
            "snippets": ctx_parts if isinstance(ctx_parts, list) else [ctx_parts],
        })

    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    cut = int(len(all_samples) * 0.8)
    samples = all_samples[:cut] if split == "train" else all_samples[cut:]

    if max_samples:
        samples = samples[:max_samples]

    return samples


def load_pubmedqa_train_sft() -> list:
    """PubMedQA examples formatted for fine-tuning (answer always in context)."""
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
    samples = []
    for item in ds:
        decision = item["final_decision"].strip().lower()       # yes/no/maybe
        long_ans = item["long_answer"].strip()
        if not long_ans or decision not in ("yes", "no", "maybe"):
            continue
        # Prepend label so it is always findable as a span
        context = decision.capitalize() + ". " + long_ans
        samples.append({
            "question": item["question"],
            "context": context,
            "answer": decision.capitalize(),
        })
    return samples


# ---------------------------------------------------------------------------
# SQuAD
# ---------------------------------------------------------------------------

def load_squad(split: str = "validation", max_samples: Optional[int] = None) -> list:
    """Load SQuAD v1.1 from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("squad", split=split)
    samples = []
    for item in ds:
        ans_texts = item["answers"]["text"]
        answer = ans_texts[0] if ans_texts else ""
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "context": item["context"],
            "answer": answer,
            "ideal_answer": answer,
            "answer_type": "factoid",
            "snippets": [item["context"]],
        })

    rng = random.Random(SEED)
    rng.shuffle(samples)
    if max_samples:
        samples = samples[:max_samples]
    return samples


def load_squad_train_sft(max_samples: int = 10_000) -> list:
    """SQuAD training split formatted for fine-tuning."""
    from datasets import load_dataset

    ds = load_dataset("squad", split="train")
    samples = []
    for item in ds:
        ans_texts = item["answers"]["text"]
        if not ans_texts:
            continue
        answer = ans_texts[0]
        if not answer or answer not in item["context"]:
            continue
        samples.append({
            "question": item["question"],
            "context": item["context"],
            "answer": answer,
        })
        if len(samples) >= max_samples:
            break
    return samples


# ---------------------------------------------------------------------------
# Corpus builder (for retrieval index)
# ---------------------------------------------------------------------------

def build_corpus(samples: list) -> list:
    """Deduplicate and return all unique, non-empty passage texts.

    Includes both the raw snippets/context AND the ideal_answer text.
    For yes/no questions the ideal_answer begins with 'Yes' or 'No', which
    is the exact format V3 was fine-tuned on.  Indexing these texts means
    dense/BM25 retrieval can surface a passage whose first span IS the answer,
    allowing V3 to extract it correctly.
    """
    seen: set = set()
    corpus = []

    def _add(text: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            corpus.append(text)

    for s in samples:
        for passage in s.get("snippets", [s["context"]]):
            _add(passage)
        if s.get("ideal_answer"):
            _add(s["ideal_answer"])

    return corpus
