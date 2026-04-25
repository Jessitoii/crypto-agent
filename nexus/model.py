"""
NEXUS-V2 Production Model Architecture (Dual-Core Inference)

This module defines the core neural architecture for the Crypto-Agent's 
decision-making engine. It features a sophisticated dual-pathway 
transformer-based design:

1. Gate Pathway: Determines if an event is a "Structural Shock" (binary).
2. Direction Pathway: Predicts market direction (LONG/SHORT) if gated.
3. Regression Heads: Estimate Take Profit (TP) and Validity Timeframes.

Architecture highlights: Layer Fusion (L8, L10, L12), Deep Reasoning Blocks 
(Stacked Decoders), and Temperature-calibrated Inference.
"""

import os
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from scipy.optimize import minimize
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class NexusLayerFusion(nn.Module):
    """
    Implements cross-layer information flow for deep transformer representations.
    
    Concatenates hidden states from different depth levels (L8, L10, L12) 
    to capture both intermediate syntactic and final semantic features.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        # 3x hidden size input due to concatenation of 3 layers
        self.proj = nn.Linear(hidden_size * 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, all_hidden_states):
        # Specific depth selection based on empirical performance in v2.0
        l8 = all_hidden_states[8]
        l10 = all_hidden_states[10]
        l12 = all_hidden_states[11]

        # Inter-layer feature fusion
        fused = torch.cat([l8, l10, l12], dim=-1) 
        x = self.proj(fused)
        x = self.norm(x)
        return self.dropout(x)

class DeepReasoningBlock(nn.Module):
    """
    Stacked Transformer Decoder Layers for iterative market logic synthesis.
    
    Uses a learnable 'Reasoning Query' to extract context from token embeddings 
    through multi-head cross-attention.
    """
    def __init__(self, hidden_size, num_layers=4, num_heads=12, dropout=0.3):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.reasoning_tower = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, token_embeddings, mask=None):
        batch_size = token_embeddings.size(0)
        query = self.query.to(token_embeddings.device).expand(batch_size, -1, -1)
        key_padding_mask = (mask == 0).to(token_embeddings.device) if mask is not None else None
        x = self.reasoning_tower(tgt=query, memory=token_embeddings, memory_key_padding_mask=key_padding_mask)
        return x.squeeze(1)

class NexusV2Production(nn.Module):
    """
    The main production model featuring the Dual-Core asynchronous architecture.
    """
    def __init__(self, backbone_name="microsoft/deberta-v3-base"):
        super().__init__()
        # Parallel encoders to prevent gradient interference between Gate and Direction tasks
        self.gate_encoder = AutoModel.from_pretrained(backbone_name, output_hidden_states=True)
        self.dir_encoder = AutoModel.from_pretrained(backbone_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        hidden_size = self.gate_encoder.config.hidden_size

        self.gate_fusion = NexusLayerFusion(hidden_size)
        self.dir_fusion = NexusLayerFusion(hidden_size)
        self.gate_reasoner = DeepReasoningBlock(hidden_size)
        self.dir_reasoner = DeepReasoningBlock(hidden_size)

        self.gate_out = nn.Linear(hidden_size, 1)
        self.dir_out = nn.Linear(hidden_size, 1)

        # Non-linear regression heads for trade parameter estimation
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Hardtanh(min_val=-1.0, max_val=1.0) # Prevents extreme outliers
        )
        self.val_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )

    def forward(self, input_ids, attention_mask):
        """
        Executes the hierarchical prediction pass.
        
        Args:
            input_ids (torch.Tensor): Tokenized news indices.
            attention_mask (torch.Tensor): Mask for padding tokens.
            
        Returns:
            dict: Multi-task logits and context vectors.
        """
        # 1. GATE PATHWAY: Structural Shock Detection
        gate_out = self.gate_encoder(input_ids=input_ids, attention_mask=attention_mask)
        gate_hidden = self.gate_fusion(gate_out.hidden_states)
        gate_ctx = self.gate_reasoner(gate_hidden, attention_mask)
        gate_logit = self.gate_out(gate_ctx)
        gate_prob = torch.sigmoid(gate_logit)

        # 2. DIRECTION PATHWAY: Gated Directional Prediction
        dir_out = self.dir_encoder(input_ids=input_ids, attention_mask=attention_mask)
        dir_hidden = self.dir_fusion(dir_out.hidden_states)

        # Apply attention-based gating to isolate relevant tokens for direction
        gate_mask = gate_prob.unsqueeze(-1) 
        dir_hidden_gated = dir_hidden * gate_mask 

        dir_ctx = self.dir_reasoner(dir_hidden_gated, attention_mask)
        dir_logit = self.dir_out(dir_ctx)

        # 3. REGRESSION PATHWAY: Trade Execution Parameters
        tp_out = self.tp_head(gate_ctx) * 5.0 # Max 5% ROI targeted
        val_out = self.val_head(gate_ctx) * 20.0 # Max 20 min validity targeted

        return {
            'gate': gate_logit,
            'direction': dir_logit,
            'tp': tp_out,
            'validity': val_out,
            'gate_ctx': gate_ctx,
            'dir_ctx': dir_ctx
        }

    def predict(self, texts, threshold=0.5, temperature=0.5301):
        """
        High-level inference wrapper with temperature calibration.
        
        Args:
            texts (Union[str, List[str]]): News headlines to analyze.
            threshold (float): Confidence threshold for structural shock gate.
            temperature (float): Calibrated scalar for probability smoothing.
            
        Returns:
            Union[dict, List[dict]]: Final trade decisions and parameters.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Apply temperature scaling to logits before probability conversion
            gate_probs = torch.sigmoid(outputs['gate'] / temperature).cpu().numpy()
            dir_probs = torch.sigmoid(outputs['direction'] / temperature).cpu().numpy()
            tp_preds = outputs['tp'].cpu().numpy()
            val_preds = outputs['validity'].cpu().numpy()

        results = []
        for i in range(len(texts)):
            g_prob = float(gate_probs[i][0])
            d_prob = float(dir_probs[i][0])

            gate_decision = 1 if g_prob >= threshold else 0

            # Conditional trade parameter extraction
            if gate_decision == 1:
                direction = 'LONG' if d_prob >= 0.5 else 'SHORT'
                dir_conf = d_prob if d_prob >= 0.5 else (1 - d_prob)
                tp_val = round(float(tp_preds[i][0]), 2)
                validity_val = round(float(val_preds[i][0]), 1)
            else:
                direction = 'WAIT' # Passive stance
                dir_conf, tp_val, validity_val = 0.0, 0.0, 0.0

            results.append({
                'text': texts[i],
                'gate': gate_decision,
                'gate_confidence': round(g_prob, 4),
                'direction': direction,
                'direction_confidence': round(float(dir_conf), 4),
                'take_profit_pct': tp_val,
                'validity_minutes': validity_val,
            })

        return results if len(results) > 1 else results[0]

class NexusPredictor:
    """
    Standard Production Predictor wrapper for sequence classification backends.
    """
    def __init__(self, model_path="standard_deberta_nexus"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            print("[WARN] Local tokenizer not found, reverting to DeBERTa-base default.")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.labels = {0: "HOLD", 1: "SHORT", 2: "LONG"}

    def predict(self, news_text, symbol):
        """
        Performs inference on a single news event.
        
        Args:
            news_text (str): Content of the news.
            symbol (str): Target asset ticker.
            
        Returns:
            dict: Decision object.
        """
        # Inject context tokens into the sequence
        formatted_text = f"[N] {news_text} [C] {symbol}"
        
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            
            # Confidence estimation via Softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0][prediction].item()

        return {
            "decision": self.labels[prediction],
            "confidence": round(confidence, 4),
            "label_id": prediction
        }
