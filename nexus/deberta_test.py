from setfit import SetFitModel
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Model and Data
# model_path should point to the directory containing the saved SetFit model
model_path = "nexus-ai-v2-core" 
model = SetFitModel.from_pretrained(model_path)
df = pd.read_json('data/nexus_elite_v2.json')

# 2. Prepare Test Dataset
# Using a random sample for validation
test_df = df.sample(n=2000, random_state=42)
test_texts = test_df['text'].tolist()
y_true = test_df['label'].tolist() 

# 3. Run Predictions
print("[INFO] Running predictions...")
raw_preds = model.predict(test_texts)

# 4. CRITICAL STEP: String -> Integer Conversion
# Map string labels back to numeric values if necessary
label_map = {"HOLD": 0, "SHORT": 1, "LONG": 2}

y_pred = []
for p in raw_preds:
    if isinstance(p, str):
        y_pred.append(label_map[p])
    else:
        y_pred.append(int(p))

# 5. Evaluation Results
acc = accuracy_score(y_true, y_pred)
print("\n" + "="*40)
print(f"NEXUS AI v2 ACCURACY: {acc:.2%}")
print("="*40)

# Generate classification report with technical label names
print(classification_report(y_true, y_pred, target_names=["HOLD", "SHORT", "LONG"]))