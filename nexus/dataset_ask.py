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
MODE = "GEMINI"

# --- SETTINGS ---
GROQ_API_KEY = os.getenv("GROQCLOUD_API_KEY")
INPUT_FILE = str(DATA_DIR / "hold_data.json")
OUTPUT_FILE = str(DATA_DIR / "hold_data_reasoning.json")
IRREVELANT_OUTPUT_FILE = str(DATA_DIR / "nexus_elite_v2_12_ultra_pure_groq_irrelevant.json")
MODEL = "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

client = AsyncGroq(api_key=GROQ_API_KEY)
gclient = genai.Client(api_key=GEMINI_API_KEY)

def save_progress(data, filename):
    """Writes data to disk to prevent data loss."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()

async def check_relevance(news_text, label, symbol):
    """
    Asks LLM about the relevance of the news to the coin.
    """
    action_map = {1: "SHORT (Price Drop)", 2: "LONG (Price Rise)", 0: "HOLD"}
    intended_action = action_map.get(label, "UNKNOWN")

    prompt = f"""
    You are a senior cryptocurrency analyst and financial logic auditor. Your task is to question the causality between the news presented to you and the action taken. Don't just do a simple word match; evaluate the fundamental weight of the news on the market.

    INPUTS:

    NEWS: {news_text}

    ACTION: {intended_action}

    COIN: {symbol}

    ANALYSIS PROTOCOL (Don't Be Robotic, Reason!):

    NARRATIVE WEIGHT: Does this news carry 'momentum' that will trigger market makers and bots? Is it a simple announcement or a structural change? "Which coin do you think will skyrocket?" polls or general comments on Twitter are noise. Distinguish the noise.

    CAUSAL DIRECTION: There must be a financial logic between the news and the action. If the news strengthens the coin, go LONG; if it weakens it, go SHORT. However, if the news is positive but the action is given as a SHORT order using 'technical indicators' (RSI, BTC Trend, etc.) as an excuse, this is not data measuring the impact of the news; it is technical analysis data. Consider anything that does not reflect the impact of the news as [IRRELEVANT].

    CHRONOLOGICAL TRAP: Is the news the CAUSE or the RESULT of the price movement? 'Bitcoin dropped to $90,000' is not news, it's a situation report. If the news describes a price movement that has already occurred and does not present a new catalyst, this data is not predictive.

    LOGICAL CONSISTENCY: Giving a 'LONG' order in anticipation of a bull market during events that are clearly negative (increased supply), such as 'Token Unlock', is a logical fallacy. The supply of goods puts price pressure on the market. Reject such data that contains 'optimism' but contradicts economic realities.

    DECISION MECHANISM: Choose only one of these two options:

    [RELEVANT]: The news is a direct and logical catalyst for the stated action. Without the news, this action would not be expected.

    [IRRELEVANT]: The link between the news and the action is weak, reversed, noisy, or the news is merely a report of consequences.

    ANSWER FORMAT: Just write [RELEVANT] or [IRRELEVANT]. Do not provide any other explanation.
    """
    
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            # Wrap API call with asyncio.wait_for to trigger retry on timeout
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
                # gclient is synchronous, so wrapping with asyncio.to_thread for non-blocking execution
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
            print(f"[TIMEOUT] Request timed out. Retrying ({retries+1}/{max_retries})")
            retries += 1
            await asyncio.sleep(2)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                retries += 1
                ms_match = re.search(r"try again in (\d+)ms", error_msg)
                sec_match = re.search(r"try again in (\d+)s", error_msg)
                wait_time = 1.0
                if ms_match: wait_time = float(ms_match.group(1)) / 1000.0
                elif sec_match: wait_time = float(sec_match.group(1))
                wait_time += 0.5
                print(f"[RATE LIMIT] Waiting {wait_time:.2f}s... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                print(f"[ERROR] {e}")
                retries += 1 
                await asyncio.sleep(1)
                
    return False

async def process_dataset():
    processed_count = 0
    perfected_data = []
    irrelevant_data = []
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                perfected_data = json.load(f)
        except Exception:
            pass

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[INFO] Analyzing total {len(data)} entries...")
    
    try:
        relevant_count = 0
        irrelevant_count = 0
        
        for i, entry in enumerate(tqdm(data)):
            # --- OPTIMIZATION ---
            # Skip entry if already processed logic could be added here
            
            news_text = entry.get('text', '')
            symbol = news_text.split('[C]')[-1].strip() if '[C]' in news_text else "General"
            
            is_relevant = await check_relevance(news_text, entry['label'], symbol)
            
            if is_relevant:
                relevant_count += 1
                perfected_data.append(entry)
            else:
                irrelevant_count += 1
                irrelevant_data.append(entry)

            # Checkpoint saving
            save_progress(perfected_data, OUTPUT_FILE)
            save_progress(irrelevant_data, IRREVELANT_OUTPUT_FILE)
            
    except KeyboardInterrupt:
        print("\n[USER INTERRUPT] Process stopped by user.")
        print("[SYSTEM] Saving current progress...")
        save_progress(perfected_data, OUTPUT_FILE)
        save_progress(irrelevant_data, IRREVELANT_OUTPUT_FILE)
        print("[SUCCESS] Data saved. Exiting.")
        return

    print(f"\n[SUCCESS] Operation completed.")
    print(f"[STATS] Original: {len(data)} | Cleaned: {relevant_count} | Irrelevant: {irrelevant_count}")

if __name__ == "__main__":
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("[ERROR] API Key missing.")
    else:
        try:
            asyncio.run(process_dataset())
        except KeyboardInterrupt:
            pass