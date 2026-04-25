"""
NEXUS Offline Dataset Generator (Forensic Archiver)

This module automates the creation of high-fidelity offline test datasets 
by scraping historical Telegram signals and enriching them with 
concurrent technical indicators and forward-looking market outcomes.

It builds the ground-truth foundation for backtesting the NEXUS 
neural decision engine.
"""

import asyncio
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
import sys

# Project Path Resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_CHANNELS, API_ID, API_HASH
from binance_client import BinanceExecutionEngine
from utils import find_coins, get_top_100_map
from price_buffer import PriceBuffer
from technical_gate import NexusTechScoreGate

# Technical Forensic Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

coin_map = get_top_100_map()
tech_gate = NexusTechScoreGate()

async def get_technicals_and_outcomes(ctx, pair, msg_ts):
    """
    Reconstructs the precise technical state and future outcomes for an event.
    
    Args:
        ctx (object): Bot context for API access.
        pair (str): Target trading pair.
        msg_ts (float): Event timestamp.
        
    Returns:
        tuple: (technicals_dict, outcomes_dict) if data is sufficient.
    """
    # 1. Forensic Window Definition (100m lookback + 60m forward tracking)
    start_ts_buffer = msg_ts - (100 * 60) 
    end_ts_outcome = msg_ts + (60 * 60)   
    
    klines = await ctx.real_exchange.client.futures_klines(
        symbol=pair.upper(),
        interval='1m',
        startTime=int(start_ts_buffer * 1000),
        endTime=int(end_ts_outcome * 1000),
        limit=200 
    )
    
    if not klines or len(klines) < 100: return None

    # Identify the exact candle containing the news event
    msg_index = -1
    parsed_klines = []
    for i, k in enumerate(klines):
        k_ts = k[0] / 1000
        parsed_klines.append({
            'ts': k_ts, 'o': float(k[1]), 'h': float(k[2]),
            'l': float(k[3]), 'c': float(k[4]), 'v': float(k[5])
        })
        if abs(k_ts - msg_ts) < 60: msg_index = i
            
    if msg_index == -1 or msg_index < 14: return None

    # --- 1. HISTORICAL TECHNICAL ANALYSIS ---
    past_candles = parsed_klines[:msg_index+1]
    prices = np.array([c['c'] for c in past_candles])
    
    if len(prices) < 15: return None
    
    # Precise RSI Calculation (Standard 14-period)
    deltas = np.diff(prices)
    seed = deltas[-14:]
    up, down = seed[seed >= 0].sum() / 14, -seed[seed < 0].sum() / 14
    rs = up / down if down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Multi-timeframe momentum and volatility metrics
    mom_1h = (prices[-1] - prices[-60]) / prices[-60] * 100 if len(prices) >= 60 else 0.0
    vol_z = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100 if len(prices) >= 20 else 0.0
    
    technicals = {
        'rsi': rsi, 'momentum_1h': mom_1h, 'vol_z': vol_z,
        'btc_trend': 0.0, 'close': past_candles[-1]['c']
    }

    # --- 2. FORWARD PERFORMANCE AUDITING ---
    future_candles = parsed_klines[msg_index+1:]
    outcome_20m = future_candles[:20]
    
    if not outcome_20m:
        max_g, max_l = 0.0, 0.0
    else:
        entry_price = past_candles[-1]['c']
        max_g = (max([c['h'] for c in outcome_20m]) - entry_price) / entry_price * 100
        max_l = (min([c['l'] for c in outcome_20m]) - entry_price) / entry_price * 100

    return technicals, {'max_gain_20m': max_g, 'max_loss_20m': max_l, 'future_candles': future_candles}

async def generate_dataset():
    """
    Orchestrates the global dataset extraction pipeline.
    """
    print("[SYSTEM] Initializing Forensic Dataset Generation Pipeline...")
    
    ctx = type('obj', (object,), {'real_exchange': BinanceExecutionEngine("", "")})
    await ctx.real_exchange.connect()
    
    # Path resolution for session persistence
    SESSION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "crypto_agent_session")
    
    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    await client.connect()
    
    results = []
    start_date = datetime.now(timezone.utc) - timedelta(days=15)
    count = 0
    
    for channel in TARGET_CHANNELS:
        print(f"\n[SCAN] Scouring Signal History: {channel}")
        async for message in client.iter_messages(channel, offset_date=start_date, reverse=True):
            if not message.text: continue
            
            detected = find_coins(message.text, coin_map=coin_map)
            if not detected: continue
                
            for pair in detected:
                try:
                    if pair == "USDT": continue
                    sys.stdout.write(f"\r[INFO] Mining: {pair} @ {message.date.strftime('%H:%M')}")
                    sys.stdout.flush()
                    
                    data = await get_technicals_and_outcomes(ctx, pair, message.date.timestamp())
                    if not data: continue
                        
                    technicals, outcomes = data
                    
                    # Enrich with categorical metadata
                    clean_symbol = pair.lower().replace("usdt", "")
                    c_data = coin_map.get(clean_symbol, {})
                    m_cap = c_data.get("cap", 0)
                    cap_str = f"${m_cap/1e9:.2f}B" if m_cap > 1e9 else f"${m_cap/1e6:.2f}M" if m_cap > 1e6 else "N/A"
                    
                    results.append({
                        "msg_text": message.text,
                        "msg_ts": message.date.timestamp(),
                        "msg_dt": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": pair.upper(),
                        "coin_full_name": c_data.get("name", "Unknown"),
                        "cap_str": cap_str,
                        "technicals": technicals,
                        "outcomes": outcomes
                    })
                    count += 1
                    
                    if count % 10 == 0:
                        logger.info(f"Forensic dataset growth: {count} samples serialized.")
                        
                except Exception as e:
                    print(f"\n[ERROR] Event extraction failure for {pair}: {e}")
                    
    # Final Persistence
    output_path = os.path.join(os.path.dirname(SESSION_PATH), "offline_test_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Pipeline concluded. {len(results)} samples archived to {output_path}")
    
    await client.disconnect()
    if hasattr(ctx.real_exchange, 'client'): await ctx.real_exchange.client.close_connection()

if __name__ == "__main__":
    asyncio.run(generate_dataset())

