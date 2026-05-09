from datasets import load_dataset as hf_load


def load_dataset_records(hf_path: str, split: str, n: int = 1000, config: str | None = None) -> list[dict]:
    if config is not None:
        ds = hf_load(hf_path, config, split=split)
    else:
        ds = hf_load(hf_path, split=split)
    subset = ds.select(range(min(n, len(ds))))
    if "trivia_qa" in hf_path:
        return _load_trivia_qa(subset)
    if "natural-questions" in hf_path:
        return _load_natural_questions(subset)
    raise ValueError(f"Unknown dataset path: {hf_path}")


def _load_trivia_qa(subset) -> list[dict]:
    records = []
    for row in subset:
        answers = [row["answer"]["value"]] + list(row["answer"].get("aliases", []))
        records.append({
            "question": row["question"],
            "answers": list(dict.fromkeys(answers)),
        })
    return records


def _load_natural_questions(subset) -> list[dict]:
    # NQ from sentence-transformers has fields: query (str), answer (str)
    records = []
    for row in subset:
        answer = row["answer"]
        answers = answer if isinstance(answer, list) else [answer]
        records.append({
            "question": row["query"],
            "answers": list(dict.fromkeys(answers)),
        })
    return records


def extract_corpus(records: list[dict]) -> list[str]:
    seen: set[str] = set()
    corpus: list[str] = []
    for rec in records:
        for ans in rec["answers"]:
            if ans not in seen:
                seen.add(ans)
                corpus.append(ans)
    return corpus
