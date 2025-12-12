import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "Qwen/Qwen3-14B"  # HuggingFace FP16 model
OUTPUT_DIR = "./qwen3_lora_text"
DATASET_PATH = "./traniningData"  # folder containing data.json
EPOCHS = 2
BATCH_SIZE = 1 
LR = 2e-4
MAX_LENGTH = 1024

# -------------------------------
# LOAD MODEL IN 4-BIT (QLoRA)
# -------------------------------
print("Loading model in 4-bit (QLoRA)…")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# APPLY LoRA
# -------------------------------
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_cfg)
print(model.print_trainable_parameters())

# -------------------------------
# LOAD DATASET
# dataset format:
# [
#   {
#     "scene": "...", "interlocutor": "...", "character_input": "...",
#     "character_reasoning": "...", "character_response": "...", "mood": "..."
#   },
#   ...
# ]
# -------------------------------
print("Loading dataset…")
dataset = load_dataset("json", data_files=os.path.join(DATASET_PATH, "20251208_1952_murderOfRogerAckroyd_Hercule_Poirot_raw.json"))

def format_example(example):
    prompt = f"Scene: {example['scene']}\nInterlocutor: {example['interlocutor']}\nMood: {example['mood']}\n\n{example['character_input']}"
    response = f"Reasoning: {example['character_reasoning']}\n\n{example['character_response']}"
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# -------------------------------
# TRAINING ARGUMENTS
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,  # simulates batch size 4
    learning_rate=LR,
    save_strategy="epoch",
    fp16=True,
    logging_steps=10,
    report_to="none",
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
# TRAIN
# -------------------------------
trainer.train()

# Save LoRA adapter only
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete! LoRA saved to:", OUTPUT_DIR)
