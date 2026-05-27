import random
from typing import TypedDict

import datasets


class DataBundle(TypedDict):
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_msmarco(
    sample_size: int = 300,
    seed: int = 42,
    distractor_target: int = 5000,
) -> DataBundle:
    random.seed(seed)
    ds = datasets.load_dataset("ms_marco", "v1.1", split="validation")

    all_indices = list(range(len(ds)))
    sampled_indices = random.sample(all_indices, min(sample_size, len(all_indices)))
    sampled_set = set(sampled_indices)

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for query_idx, row_idx in enumerate(sampled_indices):
        row = ds[row_idx]
        qid = f"q_{query_idx}"
        passage_texts = row["passages"]["passage_text"]
        is_selected = row["passages"]["is_selected"]

        qrels[qid] = {}
        for passage_idx, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            did = f"msmarco_{query_idx}_{passage_idx}"
            corpus[did] = text
            if selected == 1:
                qrels[qid][did] = 1

        # Skip queries with no relevant passage
        if not any(r > 0 for r in qrels[qid].values()):
            del qrels[qid]
            continue

        queries[qid] = row["query"]

    # Add distractors from the remaining rows
    other_indices = [i for i in all_indices if i not in sampled_set]
    random.shuffle(other_indices)
    dist_count = 0
    for row_idx in other_indices:
        if dist_count >= distractor_target:
            break
        row = ds[row_idx]
        for passage_idx, text in enumerate(row["passages"]["passage_text"]):
            did = f"msmarco_dist_{row_idx}_{passage_idx}"
            if did not in corpus:
                corpus[did] = text
                dist_count += 1

    return {"queries": queries, "corpus": corpus, "qrels": qrels}
