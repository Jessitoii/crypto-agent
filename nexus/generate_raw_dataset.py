"""
Raw Dataset Ingestion Engine (Sequential Miner)

This module implements the primary data mining pipeline for constructing 
training datasets from Telegram news archives.

It sequentially scrapes historical messages, detects mentioned assets, 
calculates concurrent BTC momentum, and audits the market outcome using 
direct Binance API calls for each event.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
import aiofiles
import random

# Project Path Resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_CHANNELS, API_ID, API_HASH
from binance_client import BinanceExecutionEngine
from utils import find_coins, get_top_100_map, coin_categories
from price_buffer import PriceBuffer

# --- MINING CONFIGURATION ---
LOOKBACK_DAYS = 175
OBSERVATION_WINDOW = 20
MIN_ROI_THRESHOLD = 0.5 
STOP_LOSS_LIMIT = 0.5
OUTPUT_FILE = "hold_data.jsonl"
COIN_MAP = get_top_100_map()

async def get_market_outcome(ctx, pair, msg_ts, btc_trend): 
    """
    Analyzes the market performance of an asset following a news event.
    
    Args:
        ctx (object): Bot context with active exchange client.
        pair (str): Target trading pair (e.g., 'BTCUSDT').
        msg_ts (float): Unix timestamp of the news event.
        btc_trend (float): Concurrent 1-hour BTC momentum.
        
    Returns:
        dict: Processed event data if it qualifies as 'HOLD' noise, else None.
    """
    try:
        start_ms = int(msg_ts * 1000)
        
        # 1. Forensic Technical Analysis (100-minute historical lookback)
        klines_hist = await ctx.real_exchange.client.futures_klines(
            symbol=pair.upper(), interval='1m', startTime=start_ms - 6000000, endTime=start_ms, limit=100
        )
        if not klines_hist: return None
        
        # Hydrate PriceBuffer for RSI calculation
        pb = PriceBuffer()
        for k in klines_hist: pb.update_candle(float(k[4]), k[0]/1000, True)
        rsi_val = pb.calculate_rsi(14)
        entry_price = float(klines_hist[-1][4])
        
        # Calculate localized momentum
        price_1h_ago = float(klines_hist[-60][4]) if len(klines_hist) >= 60 else float(klines_hist[0][4])
        change_1h = ((entry_price - price_1h_ago) / price_1h_ago) * 100

        # 2. Asset Classification Metadata
        clean_symbol = pair.upper().replace("USDT", "")
        coin_info = COIN_MAP.get(clean_symbol.lower(), {})
        mcap = coin_info.get("cap", 0)
        category = coin_categories.get(clean_symbol.upper(), "Unknown")

        # 3. Forward Outcome Analysis (20-minute post-news window)
        after_klines = await ctx.real_exchange.client.futures_klines(
            symbol=pair.upper(), interval='1m', startTime=start_ms, limit=OBSERVATION_WINDOW + 1
        )
        max_high, min_low = 0.0, 0.0
        
        for i, k in enumerate(after_klines):
            h_move = ((float(k[2]) - entry_price) / entry_price) * 100
            l_move = ((float(k[3]) - entry_price) / entry_price) * 100
            if h_move > max_high: max_high = h_move
            if l_move < min_low: min_low = l_move
 
        data_template = {
            "symbol": pair.upper(), "mcap": f"{mcap/1e9:.2f}B", "cat": category,
            "rsi": round(rsi_val, 2), "btc_trend": btc_trend,
            "mom": {"1h": round(change_1h, 2)}
        }

        # Filtering Logic: Only capture 'Noise' data (where no significant move occurred)
        if max_high >= MIN_ROI_THRESHOLD and abs(min_low) < STOP_LOSS_LIMIT:
            return None
        elif abs(min_low) >= MIN_ROI_THRESHOLD and max_high < STOP_LOSS_LIMIT:
            return None
            
        # Fetch funding rates for market state context
        funding = await ctx.real_exchange.client.futures_funding_rate(symbol=pair.upper(), limit=1)
        funding_rate = float(funding[0]['fundingRate']) if funding else 0.01

        return {
            **data_template,
            "action": "HOLD",
            "peak_pct": 0,
            "peak_min": 0,
            "funding": funding_rate
        }
    except Exception: return None

async def get_btc_trend(ctx, msg_ts):
    """
    Calculates the 1-hour BTC momentum at the time of a news event.
    
    Args:
        ctx (object): Bot context.
        msg_ts (float): Target Unix timestamp.
        
    Returns:
        float: BTC percentage change.
    """
    try:
        start_ms = int(msg_ts * 1000)
        klines = await ctx.real_exchange.client.futures_klines(
            symbol="BTCUSDT", interval='1m', startTime=start_ms - 3600000, endTime=start_ms, limit=61
        )
        if not klines: return 0.0
        start_p, end_p = float(klines[0][4]), float(klines[-1][4])
        return round(((end_p - start_p) / start_p) * 100, 2)
    except Exception: return 0.0

async def main():
    """
    Main orchestration for sequential dataset generation.
    """
    ctx = type('obj', (object,), {'real_exchange': BinanceExecutionEngine("", "")})
    await ctx.real_exchange.connect()

    # Initialize Telegram scraper
    client = TelegramClient(os.path.join("data", "crypto_agent_session"), API_ID, API_HASH)
    await client.connect()

    start_date = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    processed, found = 0, 0

    try:
        async with aiofiles.open(OUTPUT_FILE, mode='a', encoding='utf-8') as f:
            random.shuffle(TARGET_CHANNELS)
            for channel in TARGET_CHANNELS:
                print(f"\n[SYSTEM] Indexing messages for channel: {channel}")
                all_msgs = await client.get_messages(channel, offset_date=start_date, limit=20000)
                chan_total = len(all_msgs)
                
                random.shuffle(all_msgs)
                for i, message in enumerate(all_msgs):
                    processed += 1
                    percent = ((i + 1) / chan_total) * 100
                    sys.stdout.write(f"\r[MINING] {percent:.2f}% | Scanned: {processed} | Total Samples: {found} | Source: {channel}")
                    sys.stdout.flush()

                    if not message.text or len(message.text) < 20: continue
                    detected = find_coins(message.text, COIN_MAP)
                    if not detected: continue
                    
                    btc_trend = await get_btc_trend(ctx, message.date.timestamp())
                    
                    for pair in detected:
                        res = await get_market_outcome(ctx, pair, message.date.timestamp(), btc_trend)
                        if res:
                            entry = {"ts": message.date.isoformat(), "news": message.text, "data": res}
                            await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            await f.flush()
                            os.fsync(f.fileno()) # Atomic write for data safety
                            found += 1
                            print(f"\n[DIAMOND] [{res['action']}] {pair} | {message.date.strftime('%Y-%m-%d %H:%M')}")
                        await asyncio.sleep(0.01) # API rate limit protection

    finally:
        print("\n[SYSTEM] Terminating active mining sessions...")
        await client.disconnect()
        if hasattr(ctx.real_exchange, 'client'): await ctx.real_exchange.client.close_connection()
        print(f"[SUCCESS] Mining completed. {found} records written to {OUTPUT_FILE}.")

if __name__ == "__main__":
    asyncio.run(main())