from typing import TypedDict

import datasets


class DataBundle(TypedDict):
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_scifact() -> DataBundle:
    corpus_ds = datasets.load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = datasets.load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = datasets.load_dataset("BeIR/scifact-qrels", split="test")

    corpus: dict[str, str] = {
        str(row["_id"]): (row["title"] + " " + row["text"]).strip()
        for row in corpus_ds
    }

    all_queries: dict[str, str] = {str(row["_id"]): row["text"] for row in queries_ds}

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score

    # Only include queries that appear in qrels (test split)
    queries = {qid: q for qid, q in all_queries.items() if qid in qrels}

    return {"queries": queries, "corpus": corpus, "qrels": qrels}
