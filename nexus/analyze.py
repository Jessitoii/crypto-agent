import json
import sys
from collections import Counter

def analyze(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Resolve list if data is a dictionary
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    total = len(data)
    print(f"Total records: {total}")
    print(f"\n--- First record structure sample ---")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    labels = [str(d.get("label", d.get("signal", d.get("action", "?")))) for d in data]
    dist = Counter(labels)
    print(f"\n--- Label distribution ---")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    print(f"\n--- All keys ---")
    print(list(data[0].keys()))

    coins = [d.get("coin", d.get("symbol", None)) for d in data]
    coins = [c for c in coins if c]
    if coins:
        top = Counter(coins).most_common(10)
        print(f"\n--- Top 10 coins ---")
        for k, v in top:
            print(f"  {k}: {v}")

    dates = [d.get("timestamp", d.get("date", d.get("time", None))) for d in data]
    dates = [d for d in dates if d]
    if dates:
        print(f"\n--- Date range ---")
        print(f"  Start: {min(dates)}")
        print(f"  End: {max(dates)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python analyze.py <path_to_json_dataset>")