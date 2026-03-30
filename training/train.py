import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# 1. KONFİGÜRASYON
class Config:
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    NUM_CLASSES = 3  # 0: Hold, 1: Short, 2: Long

# 2. DATASET YAPISI
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

# 3. EĞİTİM FONKSİYONU
def train_standard_deberta(train_df, val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # HuggingFace'in hazır sınıflandırma modelini kullanıyoruz
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
            
    # Modeli kaydet
    model.save_pretrained("standard_deberta_nexus")
    print("Eğitim tamamlandı ve model kaydedildi.")


def prepare_nexus_data(json_path):
    # 1. Veriyi Yükle
    print(f"Veri yükleniyor: {json_path}")
    df = pd.read_json(json_path)
    
    # 2. Gerekli Sütun Kontrolü
    required_cols = ['text', 'label', 'original_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Hata: Veri setinde '{col}' sütunu bulunamadı!")

    # 3. Group-Based Splitting (Sızıntıyı Önlemek İçin Şart)
    # Aynı 'original_id'ye sahip tüm satırlar aynı sete gider.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=42)
    
    train_idx, val_idx = next(gss.split(df, groups=df['original_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # 4. İstatistikleri Raporla
    print("\n" + "="*30)
    print("VERİ BÖLME RAPORU")
    print("="*30)
    print(f"Toplam Satır: {len(df)}")
    print(f"Eğitim Satır: {len(train_df)} ({train_df['original_id'].nunique()} eşsiz haber)")
    print(f"Validasyon Satır: {len(val_df)} ({val_df['original_id'].nunique()} eşsiz haber)")
    
    print("\nSınıf Dağılımı (Train):")
    # 0: Hold, 1: Short, 2: Long
    print(train_df['label'].value_counts(normalize=True).sort_index())
    
    return train_df, val_df

train_df, val_df = prepare_nexus_data('data/nexus_elite_dataset_v5.json')

train_standard_deberta(train_df, val_df)