import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_generator(model_name: str, quantize_4bit: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto"}
    if quantize_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def build_prompt(question: str, context_docs: list[str]) -> list[dict]:
    if context_docs:
        context_text = "\n".join(f"{i + 1}. {doc}" for i, doc in enumerate(context_docs))
        system = (
            "Answer the question using only the provided context. "
            "Be concise — a few words or one short phrase. "
            "If the context does not contain the answer, say 'I don't know'."
        )
        user = f"Context:\n{context_text}\n\nQuestion: {question}"
    else:
        system = "Answer the question concisely — a few words or one short phrase."
        user = f"Question: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answers(
    model,
    tokenizer,
    records: list[dict],
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> list[str]:
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    predictions = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_prompt(r["question"], r["context_docs"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in batch
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            pred = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            predictions.append(pred)

    return predictions
