import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(corpus: list[str]) -> BM25Okapi:
    return BM25Okapi([doc.lower().split() for doc in corpus])


def save_bm25(index: BM25Okapi, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_bm25(path: Path) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_bm25(index: BM25Okapi, corpus: list[str], query: str, k: int) -> list[str]:
    scores = index.get_scores(query.lower().split())
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i] for i in top_k]


# ── DPR ───────────────────────────────────────────────────────────────────────

def build_dpr_embeddings(
    corpus: list[str],
    ctx_encoder_name: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    import torch
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder_name)
    model = DPRContextEncoder.from_pretrained(ctx_encoder_name).to(device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            out = model(**enc)
            all_embeddings.append(out.pooler_output.cpu().numpy())
    return np.vstack(all_embeddings)


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    np.save(path, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def retrieve_dpr(
    embeddings: np.ndarray,
    corpus: list[str],
    q_encoder,
    q_tokenizer,
    query: str,
    k: int,
    device: str = "cuda",
) -> list[str]:
    import torch
    q_encoder.eval()
    with torch.no_grad():
        enc = q_tokenizer(
            query, return_tensors="pt", truncation=True, max_length=64
        ).to(device)
        q_vec = q_encoder(**enc).pooler_output.cpu().numpy()  # (1, 768)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-8
    scores = (embeddings @ q_vec.T).squeeze() / norms
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i] for i in top_k]
