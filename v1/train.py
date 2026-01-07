import argparse
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# V1 Training Configuration
MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "TGoddessana/common-linux-commands-kr-en"
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MAX_LENGTH = 256
LORA_R = 8
LORA_ALPHA = 16
TEST_SPLIT = 0.1


def load_hf_dataset() -> list[dict]:
    print(f"Loading dataset from HuggingFace: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(dataset)} total samples")
    return dataset.to_list()


def create_chat_messages(question: str, command: str) -> list[dict]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": command}
    ]


def prepare_dataset(raw_data: list[dict]) -> Dataset:
    processed_data = []

    for item in raw_data:
        korean_q = item.get("korean_question", "")
        english_q = item.get("english_question", "")
        command = item.get("command", "")

        if not command:
            continue

        if korean_q:
            processed_data.append({
                "messages": create_chat_messages(korean_q, command),
                "language": "korean"
            })

        if english_q:
            processed_data.append({
                "messages": create_chat_messages(english_q, command),
                "language": "english"
            })

    print(f"Created {len(processed_data)} training samples (Korean + English)")
    return Dataset.from_list(processed_data)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for command generation (v1)")
    parser.add_argument("--output_dir", type=str, default="./commandgemma-adapters")
    args = parser.parse_args()

    # Load and prepare data
    print("\n=== Loading Data ===")
    raw_data = load_hf_dataset()
    dataset = prepare_dataset(raw_data)

    dataset_splits = dataset.train_test_split(test_size=TEST_SPLIT, shuffle=True, seed=42)
    print(f"Train: {len(dataset_splits['train'])}, Test: {len(dataset_splits['test'])}")

    # Load model and tokenizer
    print(f"\n=== Loading Model: {MODEL_NAME} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if torch.backends.mps.is_available():
        print("Using Apple MPS backend")
        model_kwargs["torch_dtype"] = torch.float32
    elif torch.cuda.is_available():
        print("Using CUDA backend")
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        print("Using CPU backend")
        model_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    if torch.backends.mps.is_available():
        base_model = base_model.to("mps")
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        max_length=MAX_LENGTH,
        packing=False,
        optim="adamw_torch",
        report_to="tensorboard",
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Train
    print("\n=== Starting Training ===")
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        peft_config=lora_config,
    )

    trainer.train()

    # Save
    print(f"\n=== Saving Model to {args.output_dir} ===")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
