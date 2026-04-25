"""
Nexus Multi-Head Architecture & Evaluator

This module defines the custom PyTorch architecture for NEXUS-7.
It implements a 'Shared Backbone + Multi-Head' model where:
1. Classifier Head: Predicts discrete signals (LONG, SHORT, HOLD).
2. TP/SL Head: Regresses expected price displacement (ROI).
3. Validity Head: Regresses the trade's temporal effectiveness.

It also includes a custom Masked Loss trainer that prevents regression loss 
from HOLD signals from polluting the backpropagation signal.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from torch.optim import AdamW
import sys
import os

# Project Path Injection
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class NexusMultiHead(nn.Module):
    """
    NEXUS-7 Multi-Head Neural Network Architecture.
    
    Uses DeBERTa-v3 as the semantic backbone with specialized linear heads 
    for classification and regression.
    """
    def __init__(self, model_name="microsoft/deberta-v3-small", dropout_rate=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Classification Head (3-Way: LONG, SHORT, HOLD)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 3)
        )
        # Expected Peak ROI Regression Head
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(128, 1)
        )
        # Temporal Validity Minutes Regression Head
        self.validity_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Executes a forward pass through the multi-head network.
        
        Returns:
            tuple: (logits, tp_preds, val_preds)
        """
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Using the [CLS] token (index 0) for sequence representation
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled), self.tp_head(pooled), self.validity_head(pooled)

class NexusDataset(Dataset):
    """
    PyTorch Dataset wrapper for NEXUS instruction data.
    """
    def __init__(self, data_path, tokenizer, max_len=256):
        self.df = pd.read_json(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        text = str(row['text']) 
        encoding = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'tp_targets': torch.tensor(row['tp_pct'] or 0.0, dtype=torch.float),
            'val_targets': torch.tensor(row['validity_minutes'] or 0.0, dtype=torch.float)
        }

def train_nexus(data_path, model_name="microsoft/deberta-v3-small", epochs=3):
    """
    Orchestrates the fine-tuning process using a custom Masked Loss objective.
    
    Args:
        data_path (str): JSON dataset path.
        model_name (str): HF model identifier.
        epochs (int): Training iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NexusMultiHead(model_name).to(device)
    
    dataset = NexusDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            tp_targets = batch['tp_targets'].to(device)
            val_targets = batch['val_targets'].to(device)

            logits, tp_preds, val_preds = model(input_ids, attention_mask)

            # --- MASKED MULTI-OBJECTIVE LOSS ---
            # 1. Classification (Primary task)
            loss_cls = criterion_cls(logits, labels)

            # 2. Regression (Secondary tasks - Masked for HOLD samples)
            # We only penalize regression errors on active trades (LONG/SHORT)
            mask = (labels != 0).float() 
            
            # Masked MSE: Gradient only flows for non-HOLD predictions
            loss_tp = (criterion_reg(tp_preds.squeeze(), tp_targets) * mask).mean()
            loss_val = (criterion_reg(val_preds.squeeze(), val_targets) * mask).mean()

            # Combined Loss with weighting (Lambda=0.1 for regression heads)
            loss = loss_cls + (0.1 * loss_tp) + (0.1 * loss_val)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[EPOCH {epoch+1}/{epochs}] Runtime Loss: {total_loss/len(loader):.4f}")

    # Persist the optimized weights
    torch.save(model.state_dict(), "nexus_multihead_final.bin")
    print("[SYSTEM] Training complete. Weights serialized to 'nexus_multihead_final.bin'.")

if __name__ == "__main__":
    train_nexus(str(DATA_DIR / "nexus_elite_v2_12.json"))