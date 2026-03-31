import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

# 1. MODEL CONFIGURATION
max_seq_length = 2048 # Sufficient for HFT analysis
dtype = None # Auto-selected based on GPU (bfloat16 or float16)
load_in_4bit = True # VRAM-efficient 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Ministral-3-3B-Instruct-2512",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# 2. LoRA PARAMETERS
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Rank/Flexibility
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Optimization for speed
    bias = "none",    # Optimization for stability
    use_gradient_checkpointing = "unsloth", # VRAM optimization
    random_state = 3407,
)

# 3. DATASET FORMATTING
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Structure that links Reasoning and Action
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output} <|end_of_text|>"
        texts.append(text)
    return { "text" : texts, }

# Load and map dataset
dataset = load_dataset("json", data_files=str(DATA_DIR / "final_finetune_ready.json"), split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. TRAINING ARGUMENTS
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Adjust to 1 if VRAM is insufficient
        gradient_accumulation_steps = 4, # Virtual batch size increase
        warmup_steps = 5,
        max_steps = -1, # Use -1 for epoch-based training
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

# 5. TRAINING START
trainer_stats = trainer.train()

# 6. SAVE MODEL (LoRA / GGUF)
model.save_pretrained("crypto_trader_lora") # Saves adapter weights only
tokenizer.save_pretrained("crypto_trader_lora")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m") # GGUF for Ollama