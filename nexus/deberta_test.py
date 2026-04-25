"""
SetFit DeBERTa Validation Script

This utility evaluates the performance of a fine-tuned SetFit/DeBERTa model 
on a held-out test set. 

It calculates accuracy and provides a detailed classification report 
(Precision, Recall, F1-Score) across the core signal classes: 
HOLD, SHORT, and LONG.
"""

from setfit import SetFitModel
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# --- MODEL & DATA HYDRATION ---
# model_path should point to the localized SetFit checkpoint
model_path = "nexus-ai-v2-core" 
model = SetFitModel.from_pretrained(model_path)
df = pd.read_json('data/nexus_elite_v2.json')

# --- TEST SET PREPARATION ---
# Extraction of a randomized validation slice
test_df = df.sample(n=2000, random_state=42)
test_texts = test_df['text'].tolist()
y_true = test_df['label'].tolist() 

# --- INFERENCE EXECUTION ---
print("[SYSTEM] Running vectorized inference on test slice...")
raw_preds = model.predict(test_texts)

# --- POST-PROCESSING: MAPPING LOGIC ---
# Standardize string predictions into categorical integers
label_map = {"HOLD": 0, "SHORT": 1, "LONG": 2}

y_pred = []
for p in raw_preds:
    if isinstance(p, str):
        y_pred.append(label_map[p])
    else:
        y_pred.append(int(p))

# --- PERFORMANCE AUDIT ---
acc = accuracy_score(y_true, y_pred)
print("\n" + "="*40)
print(f"NEXUS AI v2 CORE ACCURACY: {acc:.2%}")
print("="*40)

# Multi-class performance breakdown
print(classification_report(y_true, y_pred, target_names=["HOLD", "SHORT", "LONG"]))