"""
Backtest Results Aggregator & Reporter

This utility scans the backtest results directory and provides a formatted 
summary of all forensic reports. It serves as the primary diagnostic 
entry point for reviewing historical model performance.
"""

import os
from src.config import DATA_DIR

def report_all_backtests():
    """Iterates through and previews all backtest result logs."""
    folder = str(DATA_DIR / "backtest_results")
    
    # Iterate through sorted results for chronological analysis
    for f in sorted(os.listdir(folder)):
        if f.endswith(".txt"):
            path = os.path.join(folder, f)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            
            print(f"\n{'='*60}")
            print(f"DIAGNOSTIC REPORT: {f}")
            print('='*60)
            # Display initial telemetry data (first 800 chars)
            print(content[:800])

if __name__ == "__main__":
    report_all_backtests()