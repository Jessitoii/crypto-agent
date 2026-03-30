import json
import sys
from collections import Counter

def analyze(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Eğer list değil dict ise içindeki listeyi bul
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                data = v
                break

    total = len(data)
    print(f"Toplam kayıt: {total}")
    print(f"\n--- İlk kayıt (örnek yapı) ---")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    # Label dağılımı
    labels = [str(d.get("label", d.get("signal", d.get("action", "?")))) for d in data]
    dist = Counter(labels)
    print(f"\n--- Label dağılımı ---")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    # Key listesi
    print(f"\n--- Tüm keyler ---")
    print(list(data[0].keys()))

    # Coin dağılımı (varsa)
    coins = [d.get("coin", d.get("symbol", None)) for d in data]
    coins = [c for c in coins if c]
    if coins:
        top = Counter(coins).most_common(10)
        print(f"\n--- Top 10 coin ---")
        for k, v in top:
            print(f"  {k}: {v}")

    # Tarih aralığı (varsa)
    dates = [d.get("timestamp", d.get("date", d.get("time", None))) for d in data]
    dates = [d for d in dates if d]
    if dates:
        print(f"\n--- Tarih aralığı ---")
        print(f"  İlk: {min(dates)}")
        print(f"  Son: {max(dates)}")

analyze(sys.argv[1])