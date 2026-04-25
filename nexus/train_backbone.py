"""
NEXUS Standard Neural Trainer (DeBERTa-v3 Backend)

This module implements the production training pipeline for the NEXUS 
signal classification model. It utilizes a weighted Cross-Entropy Loss 
to address class imbalance (prioritizing active trades over HOLD noise) 
and features group-based splitting to prevent data leakage between 
related news events.
"""

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

# Project Path Configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class Config:
    """Hyperparameter and Architecture Configuration for NEXUS Training."""
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    NUM_CLASSES = 3  # Labels: 0: HOLD, 1: SHORT, 2: LONG

class NewsDataset(Dataset):
    """
    PyTorch Dataset for tokenizing and batching market news data.
    """
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

def train_standard_deberta(train_df, val_df):
    """
    Orchestrates the fine-tuning of the DeBERTa backbone.
    
    Args:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Initialize sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=Config.NUM_CLASSES
    ).to(device)

    train_ds = NewsDataset(train_df, tokenizer, Config.MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Imbalance correction: Apply 3x weight to active trade signals (LONG/SHORT)
    weights = torch.tensor([1.0, 3.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"[EPOCH {epoch+1}/{Config.EPOCHS}]")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Execution pass with embedded loss calculation
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # Serialize fine-tuned weights
    model.save_pretrained("standard_deberta_nexus")
    print(f"[SUCCESS] Model serialized to 'standard_deberta_nexus'. Avg Loss: {total_loss/len(train_loader):.4f}")

def prepare_nexus_data(json_path):
    """
    Loads and splits the dataset using Group-Based Leakage prevention.
    
    Args:
        json_path (str): Source dataset path.
        
    Returns:
        tuple: (train_df, val_df)
    """
    print(f"[SYSTEM] Ingesting elite dataset: {json_path}")
    df = pd.read_json(json_path)
    
    required_cols = ['text', 'label', 'original_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[FATAL] Dataset missing required column: {col}")

    # Prevent Data Leakage: Split by news article ID rather than individual rows.
    # This ensures that different assets mentioned in the SAME news piece 
    # don't appear in both train and validation sets.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['original_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Telemetry Report
    print("\n" + "="*40)
    print("NEXUS DATA DISTRIBUTION REPORT")
    print("="*40)
    print(f"Total Samples: {len(df)}")
    print(f"Unique Events: {df['original_id'].nunique()}")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print("\nLabel Breakdown (Training):")
    # Mapping indices: 0: HOLD, 1: SHORT, 2: LONG
    print(train_df['label'].value_counts(normalize=True).sort_index())
    print("="*40)
    
    return train_df, val_df

if __name__ == "__main__":
    # Orchestrate the standard training pipeline
    t_df, v_df = prepare_nexus_data(str(DATA_DIR / "nexus_elite_dataset_v5.json"))
    train_standard_deberta(t_df, v_df)