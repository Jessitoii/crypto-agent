"""
ML Dataset Distribution Analysis Utility

This module provides diagnostic functions for auditing JSON-based training datasets.
It calculates label distributions, ticker frequency, and chronological ranges 
to ensure data balance and quality before fine-tuning.
"""

import json
import sys
from collections import Counter

def analyze(path):
    """
    Performs a structural and statistical audit on a JSON dataset.
    
    Args:
        path (str): File system path to the JSON dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Automatically resolve nested list structures (common in raw exports)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    total = len(data)
    print(f"Total Observations Detected: {total}")
    print(f"\n--- Representative Schema Sample ---")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    # Calculate Signal Distribution (LONG/SHORT/HOLD parity)
    labels = [str(d.get("label", d.get("signal", d.get("action", "?")))) for d in data]
    dist = Counter(labels)
    print(f"\n--- Signal Distribution ---")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    print(f"\n--- Feature Keys Detected ---")
    print(list(data[0].keys()))

    # Calculate Asset Concentration
    coins = [d.get("coin", d.get("symbol", None)) for d in data]
    coins = [c for c in coins if c]
    if coins:
        top = Counter(coins).most_common(10)
        print(f"\n--- Top 10 High-Concentration Assets ---")
        for k, v in top:
            print(f"  {k}: {v}")

    # Determine Temporal Coverage
    dates = [d.get("timestamp", d.get("date", d.get("time", None))) for d in data]
    dates = [d for d in dates if d]
    if dates:
        print(f"\n--- Chronological Bounds ---")
        print(f"  Inception: {min(dates)}")
        print(f"  Conclusion: {max(dates)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python diagnostic_auditor.py <path_to_json_dataset>")