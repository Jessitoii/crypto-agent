from setfit import SetFitModel
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# 1. Modeli ve Veriyi Yükle
model_path = "nexus-ai-v2-core" # Kafa (head) olan final klasörün
model = SetFitModel.from_pretrained(model_path)
df = pd.read_json('data/nexus_elite_v2.json')

# 2. Test Setini Hazırla
test_df = df.sample(n=2000, random_state=42)
test_texts = test_df['text'].tolist()
y_true = test_df['label'].tolist() # Bunlar [0, 1, 2]

# 3. Tahminleri Al
print("[*] Tahminler yapılıyor...")
raw_preds = model.predict(test_texts)

# 4. KRİTİK ADIM: String -> Integer Dönüşümü
# Eğer model "HOLD" döndürüyorsa sayıya çeviriyoruz.
label_map = {"HOLD": 0, "SHORT": 1, "LONG": 2}

# predict çıktısı zaten 0,1,2 ise olduğu gibi bırakır, string ise map'ler
y_pred = []
for p in raw_preds:
    if isinstance(p, str):
        y_pred.append(label_map[p])
    else:
        y_pred.append(int(p))

# 5. GERÇEK SONUÇLARI YAZDIR
acc = accuracy_score(y_true, y_pred)
print("\n" + "="*40)
print(f"NEXUS AI v2 GERÇEK ACCURACY: {acc:.2%}")
print("="*40)

# target_names vererek 0,1,2'nin ne olduğunu rapora anlatıyoruz
print(classification_report(y_true, y_pred, target_names=["HOLD", "SHORT", "LONG"]))