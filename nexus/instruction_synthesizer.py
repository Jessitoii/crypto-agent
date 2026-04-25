"""
Fine-Tuning Instruction Orchestrator

This module transforms synthetic teacher-generated reasoning data into 
the final instructional format required for NEXUS-7 Supervised 
Fine-Tuning (SFT).

It maps raw analysis into a structured "Lead Strategist" persona, 
categorizing volatility and synthesizing reasoning into a production-ready 
dataset schema.
"""

import json
import re
import os

# --- SYSTEM SETTINGS ---
INPUT_FILE = "data/synthetic_finetune_data.jsonl" 
OUTPUT_FILE = "data/final_finetune_ready.json"

# Standard Instruction Manifest for the Lead Strategist persona
FINAL_INSTRUCTION = """Acting as a Lead Event-Driven Quantitative Strategist, your task is to synthesize unstructured crypto news with multi-dimensional market metrics.      

Evaluation Protocol:
1) Catalyst DNA: Classify news as 'High-Impact Catalyst', 'Lagging' (Priced-in), or 'Noise'.
2) Sentiment-Technical Confluence: Cross-examine news sentiment with Funding Rates and RSI to identify exhaustion or overextension.
3) Size-Adjusted Impact: Scale volatility expectations based on Market Cap and Category.
4) Reasoning: Provide a 2-3 sentence logic bridge focusing on liquidity grab, 'Sell the News' or trend continuation.

Output Format:
Analysis: [Your Synthesis]
Action: [LONG/SHORT/HOLD]
Expected Volatility: [Low/Medium/High]"""

def get_volatility_category(peak_pct):
    """
    Categorizes trade volatility based on historical peak displacement.
    
    Args:
        peak_pct (float): Max historical ROI during event window.
        
    Returns:
        str: 'High', 'Medium', or 'Low'.
    """
    try:
        val = abs(float(peak_pct))
        if val >= 2.5: return "High"
        if val >= 1.0: return "Medium"
        return "Low"
    except Exception:
        return "Low"

def transform_data():
    """
    Orchestrates the transformation from synthetic JSONL to instructional JSON.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found! Dataset ingestion halted.")
        return

    final_list = []
    processed_count = 0

    print(f"[SYSTEM] Recalibrating instruction dataset: {INPUT_FILE}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            
            try:
                entry = json.loads(line)
                
                # Deconstruct current teacher-generated output structure
                output_text = entry.get('output', '')
                lines = output_text.split("\n")
                
                analysis_line = next((l for l in lines if l.startswith("Analysis:")), "Analysis: N/A")
                action_line = next((l for l in lines if l.startswith("Action:")), "Action: HOLD")
                
                # Extract numeric peak value for volatility classification
                peak_line = next((l for l in lines if l.startswith("Peak:")), "")
                match = re.search(r"Peak:\s*(-?[\d.]+)", peak_line)
                
                if match:
                    peak_numeric = match.group(1)
                    vol_cat = get_volatility_category(peak_numeric)
                else:
                    vol_cat = "Low"

                # Re-construct entry following the production strategist manifest
                new_entry = {
                    "instruction": FINAL_INSTRUCTION,
                    "input": entry.get('input', ''), 
                    "output": f"{analysis_line}\n{action_line}\nExpected Volatility: {vol_cat}"
                }
                
                final_list.append(new_entry)
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"[WARNING] Serialization failure on line {line_num}: {e}")
                continue

    # Export to standard JSON array format for fine-tuning frameworks
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] {processed_count} instructions synthesized and exported to {OUTPUT_FILE}.")

if __name__ == "__main__":
    transform_data()