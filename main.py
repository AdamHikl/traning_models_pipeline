import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "Qwen/Qwen3-14B"
OUTPUT_DIR = "./qwen3_lora_text"
DATASET_PATH = "./traniningData"
EPOCHS = 2
BATCH_SIZE = 1
LR = 2e-4
MAX_LENGTH = 512

# -------------------------------
# BITSANDBYTES CONFIG (QLoRA)
# -------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# -------------------------------
# LOAD MODEL (GPU GUARANTEED)
# -------------------------------
print("Loading model in 4-bit (QLoRA)…")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# APPLY LoRA
# -------------------------------
# Prepare model for k-bit training (required for QLoRA)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# For 4-bit training, use enable_input_require_grads instead of gradient_checkpointing
model.enable_input_require_grads()
model.config.use_cache = False

# -------------------------------
# LOAD DATASET
# -------------------------------
print("Loading dataset…")

dataset = load_dataset(
    "json",
    data_files=os.path.join(
        DATASET_PATH,
        "20251208_1952_murderOfRogerAckroyd_Hercule_Poirot_raw.json",
    ),
)

def format_example(example):
    prompt = (
        f"Scene: {example['scene']}\n"
        f"Interlocutor: {example['interlocutor']}\n"
        f"Mood: {example['mood']}\n\n"
        f"{example['character_input']}"
    )

    response = (
        f"Reasoning: {example['character_reasoning']}\n\n"
        f"{example['character_response']}"
    )

    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

# -------------------------------
# TRAINING ARGUMENTS (CUDA SAFE)
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    save_strategy="epoch",
    bf16=True,               # ✅ correct for RTX 30/40/50
    fp16=False,
    logging_steps=1,
    dataloader_pin_memory=False,
    report_to="none",
    no_cuda=False,
)

# -------------------------------
# TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# -------------------------------
# VERIFY GPU (DO NOT REMOVE)
# -------------------------------
print("Model device:", next(model.parameters()).device)

# -------------------------------
# TRAIN
# -------------------------------
trainer.train()

# -------------------------------
# SAVE LoRA ADAPTER ONLY
# -------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete! LoRA saved to:", OUTPUT_DIR)
