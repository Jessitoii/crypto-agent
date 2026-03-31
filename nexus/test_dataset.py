import asyncio
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TARGET_CHANNELS, API_ID, API_HASH
from binance_client import BinanceExecutionEngine
from utils import find_coins, get_top_100_map
from price_buffer import PriceBuffer
from training.quant import NexusTechScoreGate

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

coin_map = get_top_100_map()
tech_gate = NexusTechScoreGate()

async def get_technicals_and_outcomes(ctx, pair, msg_ts):
    """
    Fetches historical klines to calculate:
    1. Technicals at moment of news (RSI, Trend, etc.)
    2. Future outcomes (Max Gain/Loss in next 20m, 1h)
    """
    start_ts_buffer = msg_ts - (100 * 60) # 100 mins before
    end_ts_outcome = msg_ts + (60 * 60)   # 60 mins after
    
    klines = await ctx.real_exchange.client.futures_klines(
        symbol=pair.upper(),
        interval='1m',
        startTime=int(start_ts_buffer * 1000),
        endTime=int(end_ts_outcome * 1000),
        limit=200 
    )
    
    if not klines or len(klines) < 100:
        return None

    msg_index = -1
    parsed_klines = []
    
    for i, k in enumerate(klines):
        k_ts = k[0] / 1000
        parsed_klines.append({
            'ts': k_ts,
            'o': float(k[1]),
            'h': float(k[2]),
            'l': float(k[3]),
            'c': float(k[4]),
            'v': float(k[5])
        })
        if abs(k_ts - msg_ts) < 60:
            msg_index = i
            
    if msg_index == -1 or msg_index < 14:
        return None

    # --- 1. CALCULATE TECHNICALS ---
    past_candles = parsed_klines[:msg_index+1]
    closes = [c['c'] for c in past_candles]
    prices = np.array(closes)
    
    if len(prices) < 15: return None
    
    deltas = np.diff(prices)
    seed = deltas[-14:]
    up = seed[seed >= 0].sum() / 14
    down = -seed[seed < 0].sum() / 14
    rs = up / down if down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    if len(prices) >= 60:
        mom_1h = (prices[-1] - prices[-60]) / prices[-60] * 100
    else:
        mom_1h = 0.0
        
    vol_z = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100 if len(prices) >= 20 else 0.0
    btc_trend = 0.0 
    
    technicals = {
        'rsi': rsi,
        'momentum_1h': mom_1h,
        'vol_z': vol_z,
        'btc_trend': btc_trend,
        'close': past_candles[-1]['c']
    }

    # --- 2. CALCULATE OUTCOMES ---
    future_candles = parsed_klines[msg_index+1:]
    
    outcome_20m = future_candles[:20]
    if not outcome_20m:
        max_gain_20m = 0.0
        max_loss_20m = 0.0
    else:
        entry_price = past_candles[-1]['c']
        highs = [c['h'] for c in outcome_20m]
        lows = [c['l'] for c in outcome_20m]
        max_price = max(highs)
        min_price = min(lows)
        
        max_gain_20m = (max_price - entry_price) / entry_price * 100
        max_loss_20m = (min_price - entry_price) / entry_price * 100

    outcomes = {
        'max_gain_20m': max_gain_20m,
        'max_loss_20m': max_loss_20m,
        'future_candles': future_candles 
    }

    return technicals, outcomes

async def generate_dataset():
    print("[SYSTEM] Starting Dataset Generation (15 Day Window)...")
    
    # 1. Setup Context
    ctx = type('obj', (object,), {})
    ctx.real_exchange = BinanceExecutionEngine("", "")
    await ctx.real_exchange.connect()
    
    # 2. Telegram Auth
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    dir = dir.replace("src", "data")
    dir = dir.replace("training", "")
    SESSION_PATH = os.path.join(dir, "crypto_agent_session")
    
    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    await client.connect()
    
    # 3. Iterate
    results = []
    start_date = datetime.now(timezone.utc) - timedelta(days=15)
    count = 0
    
    for channel in TARGET_CHANNELS:
        print(f"[SCAN] Analyzing channel: {channel}")
        async for message in client.iter_messages(channel, offset_date=start_date, reverse=True):
            if not message.text: continue
            
            msg_text = message.text
            msg_ts = message.date.timestamp()
            
            # Identify Coin
            detected_pairs = find_coins(msg_text, coin_map=coin_map)
            
            if not detected_pairs:
                continue
                
            for pair in detected_pairs:
                try:
                    if pair == "USDT": continue

                    print(f"Processing {pair} at {message.date}...")
                    
                    data = await get_technicals_and_outcomes(ctx, pair, msg_ts)
                    if not data:
                        print(f"[WARNING] Insufficient data for {pair}")
                        continue
                        
                    technicals, outcomes = data
                    
                    # Coin Info
                    clean_symbol = pair.lower().replace("usdt", "")
                    c_data = coin_map.get(clean_symbol, {})
                    cap_str = "UNKNOWN"
                    if c_data:
                        m_cap = c_data.get("cap", 0)
                        if m_cap > 1_000_000_000:
                            cap_str = f"${m_cap / 1_000_000_000:.2f} BILLION"
                        elif m_cap > 1_000_000:
                            cap_str = f"${m_cap / 1_000_000:.2f} MILLION"
                    
                    entry = {
                        "msg_text": msg_text,
                        "msg_ts": msg_ts,
                        "msg_dt": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": pair.upper(),
                        "coin_full_name": c_data.get("name", "Unknown") if c_data else "Unknown",
                        "cap_str": cap_str,
                        "technicals": technicals,
                        "outcomes": outcomes
                    }
                    
                    results.append(entry)
                    count += 1
                    
                    if count % 10 == 0:
                        print(f"[INFO] Current Progress: {count} samples collected.")
                        
                except Exception as e:
                    print(f"[ERROR] Processing failure for {pair}: {e}")
                    
    # 4. Save
    output_path = os.path.join(dir, "offline_test_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"[SUCCESS] Dataset completed. {len(results)} samples saved to {output_path}")
    
    await client.disconnect()
    await ctx.real_exchange.client.close_connection()

if __name__ == "__main__":
    asyncio.run(generate_dataset())
