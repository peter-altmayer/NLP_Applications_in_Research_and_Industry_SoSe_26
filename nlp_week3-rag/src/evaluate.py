import re
import string


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def exact_match(pred: str, gold_list: list[str]) -> bool:
    pred_norm = normalize(pred)
    return any(normalize(g) == pred_norm for g in gold_list)


def token_f1(pred: str, gold_list: list[str]) -> float:
    pred_tokens = normalize(pred).split()
    best_f1 = 0.0
    for gold in gold_list:
        gold_tokens = normalize(gold).split()
        if not pred_tokens and not gold_tokens:
            f1 = 1.0
        elif not pred_tokens or not gold_tokens:
            f1 = 0.0
        else:
            common = {
                t: min(pred_tokens.count(t), gold_tokens.count(t))
                for t in set(pred_tokens) & set(gold_tokens)
            }
            num_common = sum(common.values())
            if num_common == 0:
                f1 = 0.0
            else:
                precision = num_common / len(pred_tokens)
                recall = num_common / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def build_results_table(df):
    import pandas as pd
    agg = (
        df.groupby(["retrieval_method", "k", "model", "dataset"])
        .agg(em=("em", "mean"), f1=("f1", "mean"))
        .round(3)
    )
    agg["em/f1"] = agg["em"].map("{:.3f}".format) + " / " + agg["f1"].map("{:.3f}".format)
    return agg["em/f1"].unstack(["model", "dataset"])
