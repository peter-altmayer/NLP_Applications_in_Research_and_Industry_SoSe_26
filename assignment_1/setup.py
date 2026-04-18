'''
NLP applications in reaserch and industry - Assignment 1
Name: Peter Altmayer
Date: 2026-04-18
'''

from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch


# ── Step 1: Load Dataset ────────────────────────────────────────────────────
# Load 5 000 training examples from SQuAD (Stanford Question Answering Dataset).
# Using a slice keeps training fast while still being representative.
squad = load_dataset("squad", split="train[:5000]")

# Split into 80 % train / 20 % test so we can evaluate after each epoch.
squad = squad.train_test_split(test_size=0.2)

# Inspect one example to understand the data structure:
#   context  – the passage the model must read to find the answer
#   question – what we want the model to answer
#   answers  – dict with 'answer_start' (char offset into context) and 'text'
print(squad["train"][0])


# ── Step 2: Preprocess ──────────────────────────────────────────────────────
# Load the DistilBERT tokenizer.  It converts raw text into token IDs that the
# model understands and aligns with the model's vocabulary.
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    """Tokenize question+context pairs and compute token-level answer positions."""

    # Strip accidental whitespace from each question string.
    questions = [q.strip() for q in examples["question"]]

    # Tokenize the question (1st sequence) and context (2nd sequence) together.
    #
    # truncation="only_second"   → when the combined length exceeds max_length,
    #                              only the context (2nd seq) is truncated —
    #                              the question is always kept in full.
    # return_offsets_mapping=True → each token gets a (char_start, char_end)
    #                              tuple so we can convert char-level answer
    #                              offsets into token-level positions.
    # padding="max_length"       → pad every sample to exactly 384 tokens so
    #                              all tensors in a batch share the same shape.
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Pop offset_mapping out of the inputs dict so the model never sees it —
    # it's only needed here to locate the answer tokens.
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]

        # Character-level start and end of the answer inside the context string.
        start_char = answer["answer_start"][0]
        end_char   = start_char + len(answer["text"][0])

        # sequence_ids() returns:
        #   None  → special tokens ([CLS], [SEP], padding)
        #   0     → tokens belonging to the question (1st sequence)
        #   1     → tokens belonging to the context  (2nd sequence)
        sequence_ids = inputs.sequence_ids(i)

        # Walk forward past the question tokens and the first [SEP] to find
        # the first context token.
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        # Walk forward until we leave the context tokens.
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1  # index of the last context token

        # If the answer span falls entirely outside the (possibly truncated)
        # context window, point both positions to the [CLS] token (index 0).
        # This signals an unanswerable question to the loss function.
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Scan right from context_start until we pass the answer's first char.
            # The token just before that crossing covers start_char.
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # Scan left from context_end until we pass the answer's last char.
            # The token just after that crossing covers end_char.
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    # Store the computed token positions so Trainer can use them as labels.
    inputs["start_positions"] = start_positions
    inputs["end_positions"]   = end_positions
    return inputs


# Apply preprocess_function to every example in one batched pass.
# batched=True means the function receives a dict of lists (one list per column)
# which is much faster than processing one example at a time.
# remove_columns drops the original text columns; the model only needs tensors.
tokenized_squad = squad.map(
    preprocess_function,
    batched=True,
    remove_columns=squad["train"].column_names,
)

# DefaultDataCollator simply stacks tensors into batches.
# Unlike other collators it does NOT add extra padding because we already
# padded everything to max_length above.
data_collator = DefaultDataCollator()


# ── Step 3: Train ───────────────────────────────────────────────────────────
# Load DistilBERT with a QA head: two linear layers on top of the encoder that
# independently predict the start and end token positions of the answer span.
model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",  # directory where checkpoints are saved
    eval_strategy="epoch",             # run evaluation at the end of every epoch
    learning_rate=2e-5,                # small LR typical for fine-tuning BERT-family models
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,                 # L2 regularisation to reduce overfitting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    processing_class=tokenizer,        # tokenizer is stored with the model checkpoint
    data_collator=data_collator,
)

from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = get_last_checkpoint("my_awesome_qa_model")
if last_checkpoint is not None:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found — starting training from scratch.")
    trainer.train()

# Save the final model weights and tokenizer to output_dir so inference can
# load directly from "my_awesome_qa_model" rather than a checkpoint subfolder.
trainer.save_model()


# ── Step 4: Inference ───────────────────────────────────────────────────────
# Load the fine-tuned model and its tokenizer from the saved checkpoint.
tokenizer_inf = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Fall back to the last checkpoint if the final model hasn't been saved yet.
model_dir = get_last_checkpoint("my_awesome_qa_model") or "my_awesome_qa_model"
model_inf = AutoModelForQuestionAnswering.from_pretrained(model_dir)

question = "How many programming languages does BLOOM support?"
context  = ("BLOOM has 176 billion parameters and can generate text in 46 natural "
            "languages and 13 programming languages.")

# Tokenize the question+context pair; return_tensors="pt" gives PyTorch tensors.
inputs = tokenizer_inf(question, context, return_tensors="pt")

# Forward pass — torch.no_grad() skips gradient computation (not needed for inference).
with torch.no_grad():
    outputs = model_inf(**inputs)

# The model outputs one logit per token for the start position and one for the end.
# argmax picks the token index with the highest score.
answer_start_index = outputs.start_logits.argmax()
answer_end_index   = outputs.end_logits.argmax()

# Slice the input_ids to extract the predicted answer tokens, then decode to text.
predicted_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print("Predicted answer:", tokenizer_inf.decode(predicted_tokens))
