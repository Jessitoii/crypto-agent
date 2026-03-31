import json
from collections import Counter

def analyze_dataset(file_path):
    stats = Counter()
    total_samples = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine sample list from data structure
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
        print(f"DATASET ANALYSIS REPORT: {file_path}")
        print("-" * 40)
        print(f"Total Samples: {total_samples}")
        print("-" * 40)
        
        for action, count in stats.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{str(action).ljust(15)}: {str(count).rjust(5)} samples ({percentage:.2f}%)")
            
        print("-" * 40)
        
        # Threshold-based quality check
        if total_samples > 0:
            hold_count = stats.get('HOLD', stats.get('0', 0))
            hold_ratio = (hold_count / total_samples) * 100
            if hold_ratio < 70:
                print("[WARNING] HOLD ratio below 70%. Model might be over-aggressive.")
            else:
                print("[INFO] Strong HOLD discipline detected. Model is learning to filter noise.")
            
    except FileNotFoundError:
        print("[ERROR] Measurement file not found.")
    except Exception as e:
        print(f"[ERROR] Unexpected diagnostic failure: {e}")

def check_logic_diversity(file_path):
    logics = []
    hold_count = 0
    long_count = 0
    short_count = 0
    try:
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
    print(f"Hold Count: {hold_count}")
    print(f"Long Count: {long_count}")
    print(f"Short Count: {short_count}")
    print("-" * 40)
    
    words = " ".join(logics).lower().split()
    if len(words) > 2:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        print("MOST COMMON REASONING PATTERNS (TRIGRAMS):")
        for pattern, count in Counter(trigrams).most_common(10):
            print(f"{pattern.ljust(35)}: {count} times")
    else:
        print("[INFO] Insufficient data for trigram analysis.")

if __name__ == "__main__":
    check_logic_diversity('data/nexus_elite_v2.json')