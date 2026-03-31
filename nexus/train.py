import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

# 1. CONFIGURATION
class Config:
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    NUM_CLASSES = 3  # 0: Hold, 1: Short, 2: Long

# 2. DATASET STRUCTURE
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 3. TRAINING FUNCTION
def train_standard_deberta(train_df, val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Using standard HuggingFace sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=Config.NUM_CLASSES
    ).to(device)

    train_ds = NewsDataset(train_df, tokenizer, Config.MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    weights = torch.tensor([1.0, 3.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # Save model
    model.save_pretrained("standard_deberta_nexus")
    print("Training complete. Model saved.")


def prepare_nexus_data(json_path):
    # 1. Load Data
    print(f"Loading data: {json_path}")
    df = pd.read_json(json_path)
    
    # 2. Column Validation
    required_cols = ['text', 'label', 'original_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Error: Required column '{col}' missing from dataset.")

    # 3. Group-Based Splitting
    # Ensures all rows with the same 'original_id' are in the same split to prevent leakage.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=42)
    
    train_idx, val_idx = next(gss.split(df, groups=df['original_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # 4. Statistics Report
    print("\n" + "="*30)
    print("DATA SPLIT REPORT")
    print("="*30)
    print(f"Total Rows: {len(df)}")
    print(f"Train Rows: {len(train_df)} ({train_df['original_id'].nunique()} unique news articles)")
    print(f"Validation Rows: {len(val_df)} ({val_df['original_id'].nunique()} unique news articles)")
    
    print("\nClass Distribution (Train):")
    # 0: Hold, 1: Short, 2: Long
    print(train_df['label'].value_counts(normalize=True).sort_index())
    
    return train_df, val_df

train_df, val_df = prepare_nexus_data(str(DATA_DIR / "nexus_elite_dataset_v5.json"))

train_standard_deberta(train_df, val_df)