"""Pretty-print per-query top-k results across methods."""


def format_query_comparison(
    qid: str,
    query: str,
    runs: dict[str, dict[str, dict[str, float]]],
    corpus: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 5,
    max_chars: int = 120,
) -> str:
    gold_docs = {did for did, r in qrels.get(qid, {}).items() if r > 0}
    lines = [f"Query [{qid}]: {query!r}", ""]
    for method, run in runs.items():
        ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:k]
        lines.append(f"  {method}:")
        for rank, (did, score) in enumerate(ranked, start=1):
            text = corpus.get(did, "")[:max_chars]
            marker = " ✓" if did in gold_docs else ""
            lines.append(f"    {rank}. [{did}]{marker} (score={score:.3f}) {text!r}")
        lines.append("")
    return "\n".join(lines)


def dump_disagreement_examples(
    qids: list[str],
    queries: dict[str, str],
    runs: dict[str, dict[str, dict[str, float]]],
    corpus: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 5,
) -> str:
    parts = []
    for qid in qids:
        parts.append(format_query_comparison(qid, queries.get(qid, ""), runs, corpus, qrels, k))
    return "\n" + ("=" * 70 + "\n").join(parts)
