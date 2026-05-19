"""QA model variants V1–V4.

V1  – 2-layer BERT from scratch, trained on SQuAD
V2  – deepset/roberta-base-squad2  (pretrained general QA, eval-only)
V3  – bert-base-uncased fine-tuned by us on biomedical QA data
V4  – dmis-lab/biobert-base-cased-v1.1-squad (BioBERT specialist, eval-only)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "processed" / "checkpoints"

# Model identifiers
MODEL_NAMES = {
    "v2": "deepset/roberta-base-squad2",
    "v4": "dmis-lab/biobert-base-cased-v1.1-squad",
}
V1_TOKENIZER = "bert-base-uncased"   # vocab reuse; encoder weights are random
V3_BASE = "bert-base-uncased"

MAX_LENGTH = 384


# ---------------------------------------------------------------------------
# Span-finding helper (used during training)
# ---------------------------------------------------------------------------



def _tokenize_sample(
    tokenizer,
    question: str,
    context: str,
    answer: str,
    max_length: int = MAX_LENGTH,
) -> Optional[dict]:
    """Tokenize one (question, context, answer) triple for extractive QA training.

    Returns None if the answer cannot be found in the context.
    """
    # Locate answer in context (case-insensitive)
    ctx_lower = context.lower()
    ans_lower = answer.lower()
    char_start = ctx_lower.find(ans_lower)
    if char_start == -1:
        return None
    char_end = char_start + len(answer)

    encoding = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    sequence_ids = encoding.sequence_ids(0)
    offsets = encoding["offset_mapping"][0].tolist()

    tok_start = tok_end = 0
    for i, (sid, (os, oe)) in enumerate(zip(sequence_ids, offsets)):
        if sid != 1:
            continue
        if os <= char_start < oe:
            tok_start = i
        if os < char_end <= oe:
            tok_end = i
            break

    # If end not found (truncated), skip
    if tok_end < tok_start:
        return None

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "token_type_ids": encoding.get("token_type_ids", torch.zeros(max_length, dtype=torch.long)).squeeze(0)
                          if "token_type_ids" in encoding
                          else torch.zeros(max_length, dtype=torch.long),
        "start_positions": torch.tensor(tok_start, dtype=torch.long),
        "end_positions": torch.tensor(tok_end, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _QADataset(Dataset):
    def __init__(self, features: list):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseQAModel:
    def predict(self, question: str, context: str) -> str:
        raise NotImplementedError

    def predict_batch(self, samples: list, context_key: str = "context") -> List[str]:
        return [self.predict(s["question"], s[context_key]) for s in tqdm(samples, desc="Predicting")]


# ---------------------------------------------------------------------------
# Pipeline-based wrapper (V2, V4)
# ---------------------------------------------------------------------------

class PipelineQAModel(BaseQAModel):
    """Uses the HuggingFace question-answering pipeline. No training needed."""

    def __init__(self, model_name: str, device: int = -1):
        from transformers import pipeline
        self._pipe = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )

    def predict(self, question: str, context: str) -> str:
        if not context.strip():
            return ""
        try:
            result = self._pipe(
                question=question,
                context=context,
                max_answer_len=100,
                handle_impossible_answer=False,
            )
            return result.get("answer", "")
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Trainable wrapper (V1, V3)
# ---------------------------------------------------------------------------

class TrainableQAModel(BaseQAModel):
    """BERT-based extractive QA model that can be trained / fine-tuned."""

    def __init__(
        self,
        model_name_or_config,
        tokenizer_name: str,
        from_scratch: bool = False,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        if from_scratch:
            # V1: tiny encoder, random weights
            config = BertConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_hidden_layers=2,
                hidden_size=256,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=512,
            )
            self.model = BertForQuestionAnswering(config)
        else:
            # V3: start from a pretrained BERT checkpoint
            self.model = BertForQuestionAnswering.from_pretrained(model_name_or_config)

        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        train_data: list,
        output_dir: Path,
        epochs: int = 3,
        lr: float = 3e-5,
        batch_size: int = 16,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """Fine-tune on list of {question, context, answer} dicts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing {len(train_data)} samples …")
        features = []
        skipped = 0
        for s in train_data:
            feat = _tokenize_sample(
                self.tokenizer, s["question"], s["context"], s["answer"], max_length
            )
            if feat is None:
                skipped += 1
            else:
                features.append(feat)

        print(f"  Usable: {len(features)}  Skipped (answer not in context): {skipped}")

        loader = DataLoader(_QADataset(features), batch_size=batch_size, shuffle=True)
        total_steps = len(loader) * epochs
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"  Epoch {epoch + 1} avg loss: {total_loss / len(loader):.4f}")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Checkpoint saved → {output_dir}")

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.model = BertForQuestionAnswering.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, question: str, context: str) -> str:
        if not context.strip():
            return ""

        self.model.eval()
        enc = self.tokenizer(
            question,
            context,
            max_length=MAX_LENGTH,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )

        # Restrict span selection to the context portion (sequence_id == 1).
        # Without this, weak models (V1) pick subword tokens from the question
        # and V3 picks the trailing "?" of the question instead of an answer span.
        seq_ids = enc.sequence_ids(0)
        ctx_positions = [i for i, s in enumerate(seq_ids) if s == 1]
        ctx_start = ctx_positions[0] if ctx_positions else 0
        ctx_end   = ctx_positions[-1] if ctx_positions else len(seq_ids) - 1

        model_input = {k: v.to(self.device) for k, v in enc.items()
                       if k in ("input_ids", "attention_mask", "token_type_ids")}

        with torch.no_grad():
            outputs = self.model(**model_input)

        start_logits = outputs.start_logits[0]
        end_logits   = outputs.end_logits[0]

        # Mask positions outside the context window to -inf
        mask = torch.full_like(start_logits, float("-inf"))
        mask[ctx_start: ctx_end + 1] = 0.0
        start_logits = start_logits + mask
        end_logits   = end_logits   + mask

        start = torch.argmax(start_logits).item()
        end   = torch.argmax(end_logits).item()
        if end < start:
            end = start

        tokens = enc["input_ids"][0][start: end + 1]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(variant: str, device: Optional[str] = None) -> BaseQAModel:
    """Return the correct model object for variant in {v1, v2, v3, v4}.

    For V1 and V3 the returned model still needs fine_tune() or load_checkpoint()
    before inference — call load_model() to get a ready-to-use model.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_idx = 0 if dev == "cuda" else -1

    if variant == "v1":
        return TrainableQAModel(
            model_name_or_config=None,
            tokenizer_name=V1_TOKENIZER,
            from_scratch=True,
            device=dev,
        )
    elif variant == "v2":
        return PipelineQAModel(MODEL_NAMES["v2"], device=dev_idx)
    elif variant == "v3":
        return TrainableQAModel(
            model_name_or_config=V3_BASE,
            tokenizer_name=V3_BASE,
            from_scratch=False,
            device=dev,
        )
    elif variant == "v4":
        return PipelineQAModel(MODEL_NAMES["v4"], device=dev_idx)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from v1, v2, v3, v4.")


def load_model(variant: str, device: Optional[str] = None) -> BaseQAModel:
    """Load a fully ready model.

    V1/V3: loads the saved checkpoint from data/processed/checkpoints/{variant}/.
    V2/V4: downloads from HuggingFace (or cache).
    Raises FileNotFoundError for V1/V3 if no checkpoint exists yet.
    """
    if variant in ("v2", "v4"):
        return build_model(variant, device)

    ckpt = CHECKPOINT_DIR / variant
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint for {variant} at {ckpt}.\n"
            f"Run:  python experiments/run_experiment.py --model {variant} --fine-tune"
        )
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if variant == "v1":
        # Build tiny architecture first; load_checkpoint replaces weights from the saved ckpt
        m = TrainableQAModel(
            model_name_or_config=None,
            tokenizer_name=V1_TOKENIZER,
            from_scratch=True,
            device=dev,
        )
        m.load_checkpoint(ckpt)
    else:
        # V3: load directly from the saved checkpoint — no redundant bert-base-uncased download
        m = TrainableQAModel(
            model_name_or_config=str(ckpt),
            tokenizer_name=str(ckpt),
            from_scratch=False,
            device=dev,
        )
    return m
