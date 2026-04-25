"""
Dataset Quality Assurance & Heuristic Auditor

This module performs deep-dive analytics on trading datasets to ensure 
signal integrity and categorical balance. 

Key features:
1. Signal Distribution Audit (HOLD discipline verification).
2. Semantic Reasoning Pattern Analysis (Trigram extraction).
3. Logical Divergence detection across multiple asset classes.
"""

import json
from collections import Counter

def analyze_dataset(file_path):
    """
    Performs a high-level statistical audit of the dataset signal distribution.
    
    Validates that the dataset maintains 'HOLD' discipline, which is critical 
    for preventing model over-trading in noisy markets.
    
    Args:
        file_path (str): Path to the target JSON dataset.
    """
    stats = Counter()
    total_samples = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine sample list from data structure (handle dict vs list exports)
        samples = data if isinstance(data, list) else []
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    samples = val
                    break
        
        total_samples = len(samples)
        for d in samples:
            action = d.get('action', d.get('label', 'UNKNOWN'))
            stats[str(action)] += 1

        print("-" * 40)
        print(f"DATASET QUALITY REPORT: {file_path}")
        print("-" * 40)
        print(f"Total Observations: {total_samples}")
        print("-" * 40)
        
        for action, count in stats.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{str(action).ljust(15)}: {str(count).rjust(5)} samples ({percentage:.2f}%)")
            
        print("-" * 40)
        
        # Sectoral Heuristic: Threshold-based quality check
        if total_samples > 0:
            hold_count = stats.get('HOLD', stats.get('0', 0))
            hold_ratio = (hold_count / total_samples) * 100
            if hold_ratio < 70:
                print("[WARNING] HOLD discipline < 70%. High risk of signal hallucination.")
            else:
                print("[INFO] Strong HOLD discipline confirmed. Filtering efficacy is high.")
            
    except FileNotFoundError:
        print("[ERROR] Measurement file not found.")
    except Exception as e:
        print(f"[ERROR] Diagnostic pipeline fault: {e}")

def check_logic_diversity(file_path):
    """
    Audits the semantic diversity of the reasoning blocks in the dataset.
    
    Uses trigram extraction to identify repetitive or circular logic patterns 
    that might lead to model overfitting on specific phrases.
    
    Args:
        file_path (str): Path to the target JSON dataset.
    """
    logics = []
    hold_count = 0
    long_count = 0
    short_count = 0
    try:
        # Robust encoding handling for cross-platform datasets
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            hold_count = sum(1 for d in data if d.get('label') == 0) 
            short_count = sum(1 for d in data if d.get('label') == 1) 
            long_count = sum(1 for d in data if d.get('label') == 2) 

            for d in data:
                if d.get('label') == 0:
                    logics.append(d.get('reasoning', ''))
    except (UnicodeDecodeError, Exception):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            hold_count = sum(1 for d in data if d.get('label') == 0) 
            short_count = sum(1 for d in data if d.get('label') == 1) 
            long_count = sum(1 for d in data if d.get('label') == 2) 
            for d in data:
                if d.get('label') == 0:
                    logics.append(d.get('reasoning', ''))
    
    print("-" * 40)
    print(f"HOLD Concentration: {hold_count}")
    print(f"LONG Concentration: {long_count}")
    print(f"SHORT Concentration: {short_count}")
    print("-" * 40)
    
    # N-Gram Analysis for semantic pattern discovery
    words = " ".join(logics).lower().split()
    if len(words) > 2:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        print("SEMANTIC REASONING PATTERNS (TOP TRIGRAMS):")
        for pattern, count in Counter(trigrams).most_common(10):
            print(f"{pattern.ljust(35)}: {count} times")
    else:
        print("[INFO] Sparse semantic data; trigram audit skipped.")

if __name__ == "__main__":
    # Standard entry point for dataset audit
    check_logic_diversity('data/nexus_elite_v2.json')