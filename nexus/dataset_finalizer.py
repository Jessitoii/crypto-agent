"""
Elite Dataset Sanitization Utility

This module performs final post-processing on the Elite Nexus dataset. 
It ensures all target profit percentages are absolute (positive) values, 
standardizing the regression targets for the multi-head architecture.
"""

import json 

def sanitize_elite_dataset(input_path, output_path):
    """
    Cleans and standardizes trade outcome targets in the dataset.
    
    Args:
        input_path (str): Path to the v4 raw dataset.
        output_path (str): Path to save the v5 refined dataset.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refined_data = []

    for entry in data:
        # Standardize TP as absolute displacement from entry
        entry["tp_pct"] = abs(entry["tp_pct"])
        refined_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    sanitize_elite_dataset("data/nexus_elite_dataset_v4.json", "data/nexus_elite_dataset_v5.json")