import os
from src.config import DATA_DIR

folder = str(DATA_DIR / "backtest_results")
for f in sorted(os.listdir(folder)):
    if f.endswith(".txt"):
        path = os.path.join(folder, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        print(f"\n{'='*40}")
        print(f"FILE: {f}")
        print('='*40)
        print(content[:800])  # İlk 800 karakter yeterli