"""
AI-Assisted Dataset Relevance Auditor

This module utilizes Large Language Models (Gemini/Llama) to audit the 
causal relationship between news events and trading actions in a dataset.

It filters out 'noisy' data where the price movement was likely unrelated 
to the news, ensuring that the final training set contains only 
high-signal, fundamentally-sound observations.
"""

import json
import os
from groq import AsyncGroq
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
import re
from google import genai
from google.genai import types

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

# Execution Mode: "GEMINI" or "GROQ"
MODE = "GEMINI"

# --- SYSTEM CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQCLOUD_API_KEY")
INPUT_FILE = str(DATA_DIR / "hold_data.json")
OUTPUT_FILE = str(DATA_DIR / "hold_data_reasoning.json")
IRREVELANT_OUTPUT_FILE = str(DATA_DIR / "nexus_elite_v2_12_ultra_pure_groq_irrelevant.json")
MODEL = "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Asynchronous LLM Clients
client = AsyncGroq(api_key=GROQ_API_KEY)
gclient = genai.Client(api_key=GEMINI_API_KEY)

def save_progress(data, filename):
    """
    Persists current progress to disk to mitigate data loss from interruptions.
    
    Args:
        data (list): Current list of processed records.
        filename (str): Target filesystem path.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()

async def check_relevance(news_text, label, symbol):
    """
    Invokes an LLM to evaluate the fundamental causality of a news event.
    
    The prompt enforces a "Financial Logic Auditor" persona to reject news 
    that is merely descriptive of price action or lacks market momentum.
    
    Args:
        news_text (str): Raw news content.
        label (int): Integer label (0=HOLD, 1=SHORT, 2=LONG).
        symbol (str): Asset ticker.
        
    Returns:
        bool: True if the news is evaluated as [RELEVANT] to the action.
    """
    action_map = {1: "SHORT (Price Drop)", 2: "LONG (Price Rise)", 0: "HOLD"}
    intended_action = action_map.get(label, "UNKNOWN")

    prompt = f"""
    You are a senior cryptocurrency analyst and financial logic auditor. Your task is to question the causality between the news presented to you and the action taken. Don't just do a simple word match; evaluate the fundamental weight of the news on the market.

    INPUTS:
    NEWS: {news_text}
    ACTION: {intended_action}
    COIN: {symbol}

    ANALYSIS PROTOCOL:
    1. NARRATIVE WEIGHT: Does this news carry structural momentum? Distinguish noise from impact.
    2. CAUSAL DIRECTION: Is there a clear financial link? Reject technical analysis excuses.
    3. CHRONOLOGICAL TRAP: Is the news the CAUSE or the RESULT? Reject situation reports.
    4. LOGICAL CONSISTENCY: Does the action match economic reality (e.g., Reject LONG on Token Unlocks).

    DECISION MECHANISM:
    [RELEVANT]: Direct and logical catalyst.
    [IRRELEVANT]: Weak, reversed, or noisy link.

    ANSWER FORMAT: Just write [RELEVANT] or [IRRELEVANT].
    """
    
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            # Route request based on selected provider mode
            if MODE == "GROQ":
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=10
                    ),
                    timeout=30.0
                )
                response = completion.choices[0].message.content.strip()
                return "[RELEVANT]" in response

            elif MODE == "GEMINI":
                # Offload synchronous Google client to a thread to maintain async flow
                def run_gemini():
                    return gclient.models.generate_content(
                        model="gemma-3-27b-it", 
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0)
                    )

                res = await asyncio.wait_for(
                    asyncio.to_thread(run_gemini),
                    timeout=30.0
                )
                response = res.text.strip()
                return "[RELEVANT]" in response

        except asyncio.TimeoutError:
            print(f"[TIMEOUT] Network latency exceeded. Retrying ({retries+1}/{max_retries})")
            retries += 1
            await asyncio.sleep(2)

        except Exception as e:
            error_msg = str(e)
            # Handle Rate Limiting with dynamic backoff
            if "429" in error_msg:
                retries += 1
                ms_match = re.search(r"try again in (\d+)ms", error_msg)
                sec_match = re.search(r"try again in (\d+)s", error_msg)
                wait_time = 1.0
                if ms_match: wait_time = float(ms_match.group(1)) / 1000.0
                elif sec_match: wait_time = float(sec_match.group(1))
                wait_time += 0.5
                print(f"[RATE LIMIT] Provider throttled. Waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"[ERROR] Logic Error: {e}")
                retries += 1 
                await asyncio.sleep(1)
                
    return False

async def process_dataset():
    """
    Orchestrates the batch processing and cleaning of the dataset.
    """
    perfected_data = []
    irrelevant_data = []
    
    # Reload existing progress if available
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                perfected_data = json.load(f)
        except Exception:
            pass

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[SYSTEM] Auditing {len(data)} observations for fundamental causality...")
    
    try:
        relevant_count = 0
        irrelevant_count = 0
        
        for i, entry in enumerate(tqdm(data)):
            news_text = entry.get('text', '')
            # Extract ticker symbol from the internal labeling format
            symbol = news_text.split('[C]')[-1].strip() if '[C]' in news_text else "General"
            
            is_relevant = await check_relevance(news_text, entry['label'], symbol)
            
            if is_relevant:
                relevant_count += 1
                perfected_data.append(entry)
            else:
                irrelevant_count += 1
                irrelevant_data.append(entry)

            # Atomic checkpoint saving
            save_progress(perfected_data, OUTPUT_FILE)
            save_progress(irrelevant_data, IRREVELANT_OUTPUT_FILE)
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Snapshot captured. Saving work-in-progress...")
        save_progress(perfected_data, OUTPUT_FILE)
        save_progress(irrelevant_data, IRREVELANT_OUTPUT_FILE)
        return

    print(f"\n[SUCCESS] Dataset Audit Concluded.")
    print(f"[STATS] Total: {len(data)} | Validated: {relevant_count} | Noise: {irrelevant_count}")

if __name__ == "__main__":
    if not GROQ_API_KEY and MODE == "GROQ":
        print("[ERROR] Provider API Key (Groq) is missing from environment.")
    else:
        try:
            asyncio.run(process_dataset())
        except KeyboardInterrupt:
            pass