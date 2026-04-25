"""
NEXUS-7 Local Strategist (SFT) Training Pipeline

This module implements Supervised Fine-Tuning (SFT) for the NEXUS-7 
LLM using Unsloth's optimized training stack. It transforms raw news 
data into a "Lead Quantitative Strategist" persona via LoRA adapters, 
enabling deep reasoning and volatility prediction on consumer-grade GPUs.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import sys

# System Path Integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

# --- 1. MODEL ARCHITECTURE & QUANTIZATION ---
max_seq_length = 2048 # Sufficient context for HFT event-driven synthesis
dtype = None          # Auto-calibrate based on hardware (BF16/FP16)
load_in_4bit = True   # Aggressive quantization for VRAM efficiency

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Ministral-3-3B-Instruct-2512",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    use_gradient_checkpointing = "unsloth", 
)

# --- 2. PEFT / LoRA ADAPTATION CONFIGURATION ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Rank of the adaptation matrices
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Dropout=0 is typically superior for speed
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# --- 3. DATASET SCHEMATIZATION ---
def formatting_prompts_func(examples):
    """
    Maps instruction/input/output triplets into the NEXUS prompt template.
    
    Args:
        examples (dict): Dataset features.
        
    Returns:
        dict: Processed text blocks.
    """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Structure that links Reasoning (Chain of Thought) with explicit Action
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output} <|end_of_text|>"
        texts.append(text)
    return { "text" : texts, }

# Load and hydrate instructional dataset
dataset = load_dataset("json", data_files=str(DATA_DIR / "final_finetune_ready.json"), split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# --- 4. OPTIMIZED TRAINING HYPERPARAMETERS ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4, # Effective batch size = 8
        warmup_steps = 5,
        max_steps = -1, 
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# --- 5. EXECUTION & PERSISTENCE ---
print("[SYSTEM] Commencing NEXUS-7 SFT Training...")
trainer_stats = trainer.train()

# Save adapter weights for runtime deployment
model.save_pretrained("crypto_trader_lora") 
tokenizer.save_pretrained("crypto_trader_lora")

# Export GGUF for edge-inference (Ollama/Llama.cpp compatibility)
print("[INFO] Exporting GGUF for edge deployment...")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")
