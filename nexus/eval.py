import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from torch.optim import AdamW

# 1. MODEL TANIMI (SEQUENTIAL HEADS)
class NexusMultiHead(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small", dropout_rate=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 3)
        )
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(128, 1)
        )
        self.validity_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled), self.tp_head(pooled), self.validity_head(pooled)

# 2. DATASET VE MASKELENMİŞ KAYIP (LOSS) MANTIĞI
class NexusDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        self.df = pd.read_json(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        # Text temizliği ve context injection
        text = str(row['text']) 
        encoding = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'tp_targets': torch.tensor(row['tp_pct'] or 0.0, dtype=torch.float),
            'val_targets': torch.tensor(row['validity_minutes'] or 0.0, dtype=torch.float)
        }

# 3. EĞİTİM DÖNGÜSÜ (CUSTOM TRAINER)
def train_nexus(data_path, model_name="microsoft/deberta-v3-small", epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NexusMultiHead(model_name).to(device)
    
    dataset = NexusDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Loss Fonksiyonları
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

            # --- MASKELENMİŞ LOSS HESABI ---
            # 1. Sınıflandırma Kaybı (Her zaman hesaplanır)
            loss_cls = criterion_cls(logits, labels)

            # 2. Regresyon Kaybı (Sadece LONG (2) veya SHORT (1) ise hesaplanır)
            mask = (labels != 0).float() # HOLD (0) olanları maskele
            
            # Maskelenmiş MSE: (tahmin - gerçek)^2 * mask -> HOLD ise 0 olur
            loss_tp = (criterion_reg(tp_preds.squeeze(), tp_targets) * mask).mean()
            loss_val = (criterion_reg(val_preds.squeeze(), val_targets) * mask).mean()

            # Toplam Kayıp (Lambda katsayıları: Regresyonu başta düşük tutuyoruz)
            loss = loss_cls + (0.1 * loss_tp) + (0.1 * loss_val) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "nexus_multihead_final.bin")
    print("Model kaydedildi.")

if __name__ == "__main__":
    train_nexus("data/nexus_elite_v2_12.json")