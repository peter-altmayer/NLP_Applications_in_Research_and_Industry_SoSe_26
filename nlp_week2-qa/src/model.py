from transformers import pipeline

PROMPT_TEMPLATE = (
    "Answer the question based on the context below.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)


def run_extractive(model_name: str, examples: list[dict]) -> list[dict]:
    pipe = pipeline("question-answering", model=model_name)
    results = []
    for ex in examples:
        out = pipe(question=ex["question"], context=ex["context"])
        results.append({
            "question": ex["question"],
            "context": ex["context"],
            "predicted_answer": out["answer"],
            "score": round(out["score"], 4),
        })
    return results


def run_generative(model_name: str, examples: list[dict], **gen_kwargs) -> list[dict]:
    pipe = pipeline("text2text-generation", model=model_name)
    results = []
    for ex in examples:
        prompt = PROMPT_TEMPLATE.format(
            context=ex["context"],
            question=ex["question"],
        )
        out = pipe(prompt, **gen_kwargs)
        results.append({
            "question": ex["question"],
            "context": ex["context"],
            "predicted_answer": out[0]["generated_text"],
        })
    return results
