import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_TEMPLATE = (
    "Answer the question based on the context below.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)


def run_extractive(model_name: str, examples: list[dict]) -> list[dict]:
    print(f"  Loading tokenizer and model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(DEVICE)
    model.eval()

    results = []
    for ex in examples:
        inputs = tokenizer(
            ex["question"],
            ex["context"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        start = outputs.start_logits.argmax().item()
        end = outputs.end_logits.argmax().item()
        # clamp end so it is never before start
        end = max(start, end)
        tokens = inputs["input_ids"][0, start: end + 1]
        answer = tokenizer.decode(tokens, skip_special_tokens=True)
        score = torch.softmax(outputs.start_logits, dim=-1)[0, start].item()

        results.append({
            "question": ex["question"],
            "context": ex["context"],
            "predicted_answer": answer,
            "score": round(score, 4),
        })
    return results


def run_generative(model_name: str, examples: list[dict], **gen_kwargs) -> list[dict]:
    print(f"  Loading tokenizer and model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    results = []
    for ex in examples:
        prompt = PROMPT_TEMPLATE.format(
            context=ex["context"],
            question=ex["question"],
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({
            "question": ex["question"],
            "context": ex["context"],
            "predicted_answer": answer,
        })
    return results
