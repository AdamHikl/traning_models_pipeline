import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"   # <-- your base model
LORA_PATH = "./output_adapters"           # <-- folder with adapter_model.safetensors
OUT_DIR = "./merged_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",      # IMPORTANT: merge on CPU
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, LORA_PATH)

# Merge LoRA → base weights
model = model.merge_and_unload()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✅ LoRA merged and saved to", OUT_DIR)