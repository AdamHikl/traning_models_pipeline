import os
import time
import glob
import torch
import argparse
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
# ARGUMENT PARSING
# -------------------------------
parser = argparse.ArgumentParser(description="Train a LoRA model on persona dataset")
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen3-8B", help="Base model name (e.g., Qwen/Qwen3-8B)")
args = parser.parse_args()

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = args.model
DATASET_PATH = "./trainingData"

# Automatically scan the trainingData folder for all JSONL files
jsonl_pattern = os.path.join(DATASET_PATH, "*.jsonl")
DATASET_FILES = [os.path.basename(f) for f in sorted(glob.glob(jsonl_pattern))]

if not DATASET_FILES:
    raise ValueError(f"No JSONL files found in {DATASET_PATH}")

print(f"Found {len(DATASET_FILES)} JSONL file(s) in {DATASET_PATH}:")
for f in DATASET_FILES:
    print(f"  - {f}")

EPOCHS = 2
BATCH_SIZE = 1
LR = 2e-4
MAX_LENGTH = 512

# Extract character name from the first dataset filename
# Expected format: YYYYMMDD_HHMM_novelName_Character_Name_dataset.jsonl
dataset_parts = DATASET_FILES[0].replace("_dataset.jsonl", "").split("_")
# Character name is everything after the novel name (skip timestamp parts)
character_name = "_".join(dataset_parts[3:])  # e.g., "Hercule_Poirot"

# Create timestamp for output directory
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Create output directory with format: ./qwen3_lora_text/character_model_timestamp_
# Extract just the model name (e.g., "Qwen3-14B") from the full path
model_short_name = MODEL_NAME.split("/")[-1]
OUTPUT_DIR = f"./output_adapters"

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
print(f"Loading dataset from {len(DATASET_FILES)} file(s)…")

# Create full paths for all dataset files
dataset_file_paths = [os.path.join(DATASET_PATH, f) for f in DATASET_FILES]

dataset = load_dataset(
    "json",
    data_files=dataset_file_paths,
)

def format_example(example):
    # Build ChatML format from instruction, input, and output fields
    # Handle both new 'messages' format and legacy format
    if 'messages' in example:
        messages = example['messages']
    elif 'user' in example and 'assistant' in example:
        # New raw format: {user, think, assistant}
        # Format the assistant response with <think> tags
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': example['user']},
            {'role': 'assistant', 'content': f"<think>\n{example.get('think', '')}\n</think>\n{example['assistant']}"}
        ]
    else:
        print("Legacy format detected")
        # Convert legacy format key-values to messages list
        messages = [
            {'role': 'system', 'content': example.get('instruction', '')},
            {'role': 'user', 'content': example.get('input', '')},
            {'role': 'assistant', 'content': example.get('output', '')}
        ]
    
    text = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

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
print("\nStarting training...")
start_time = time.time()

trainer.train()

end_time = time.time()
training_duration = end_time - start_time

# -------------------------------
# SAVE LoRA ADAPTER ONLY
# -------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Print training duration
hours = int(training_duration // 3600)
minutes = int((training_duration % 3600) // 60)
seconds = int(training_duration % 60)

print("\n" + "="*50)
print("Training complete! LoRA saved to:", OUTPUT_DIR)
print(f"Total training time: {hours}h {minutes}m {seconds}s ({training_duration:.2f} seconds)")
print("="*50)
