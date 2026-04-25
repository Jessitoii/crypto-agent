"""
Market Data Mining and Preparation Utility

This module is responsible for bootstrapping the local market cache by 
downloading historical 1-minute klines (candle data) for all relevant 
Binance Futures pairs.

It implements batch-based downloading with rate-limit protection and 
local pickle caching to optimize subsequent backtesting and training runs.
"""

import asyncio
import os
import pandas as pd
import sys
from binance import AsyncClient
from datetime import datetime, timedelta, timezone

from utils import get_top_100_map, check_is_stablecoin

# Pre-fetch top market assets for validation
COIN_MAP = get_top_100_map()

# Curated list of high-liquidity Binance Futures pairs
MANUAL_BINANCE_FUTURES_TICKERS = [
    # Major Market Leaders
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    # Layer 1 & 2 Ecosystems
    "NEARUSDT", "ATOMUSDT", "ALGOUSDT", "FTMUSDT", "APTUSDT", "SUIUSDT", "SEIUSDT", "OPUSDT", "ARBUSDT", "HBARUSDT",
    "INJUSDT", "LDOUSDT", "TIAUSDT", "STXUSDT", "EGLDUSDT", "FILUSDT", "ICPUSDT", "RUNEUSDT", "GRTUSDT", "AAVEUSDT",
    # AI & DePIN Infrastructure
    "FETUSDT", "RENDERUSDT", "TAOUSDT", "NEARUSDT", "AGIXUSDT", "WLDUSDT", "ARKMUSDT", "THETAUSDT",
    # Sectoral Memecoins
    "DOGEUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "1000BONKUSDT", "1000FLOKIUSDT", "WIFUSDT", "PEOPLEUSDT", "MEMEUSDT",
    "POPCATUSDT", "BOMEUSDT", "1000LUNCUSDT", "1000RATSUSDT",
    # DeFi Aggregators & Protocols
    "UNIUSDT", "SUSHIUSDT", "DYDXUSDT", "CRVUSDT", "MKRUSDT", "SNXUSDT", "PENDLEUSDT", "ENAUSDT", "ETHFIUSDT",
    "JUPUSDT", "PYTHUSDT", "STRKUSDT", "AXSUSDT", "IMXUSDT", "GALAUSDT", "BEAMXUSDT", "SANDUSDT", "MANAUSDT",
    # Legacy Assets & High-Volume Mid-caps
    "LTCUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT", "TRXUSDT", "VETUSDT", "NEOUSDT", "QTUMUSDT", "EOSUSDT", "IOTAUSDT",
    "ZECUSDT", "DASHUSDT", "XMRUSDT", "ONTUSDT", "ZILUSDT", "BATUSDT", "ENJUSDT", "KNCUSDT", "ANKRUSDT", "OCEANUSDT",
    "CHZUSDT", "ALICEUSDT", "FLOWUSDT", "KAVAUSDT", "GMXUSDT", "ORDIUSDT", "1000SATSUSDT", "GASUSDT", "TRBUSDT"
]

BASE_DIR = "data/market_cache"
KLINES_DIR = f"{BASE_DIR}/klines"
FUNDING_DIR = f"{BASE_DIR}/funding"

# Ensure directory structure exists
for d in [KLINES_DIR, FUNDING_DIR]:
    if not os.path.exists(d): os.makedirs(d)

async def download_symbol_data(client, symbol):
    """
    Downloads historical 1-minute klines for a given asset from Binance.
    
    Caches the results in a binary pickle format for rapid loading during 
    simulation or training.
    
    Args:
        client (AsyncClient): Authenticated or public Binance AsyncClient.
        symbol (str): Target ticker (e.g., 'BTCUSDT').
        
    Returns:
        bool: True if download was successful or data already existed.
    """
    try:
        kline_path = f"{KLINES_DIR}/{symbol}_1m.pkl"
        
        if not os.path.exists(kline_path):
            klines = []
            # Historical generator handles pagination and rate-limits internally
            gen = await client.futures_historical_klines_generator(symbol, "1m", "1 year ago UTC")
            async for k in gen:
                # Store only essential OHLCV data to minimize memory footprint
                klines.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[7])])
            
            if klines:
                df_k = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df_k.to_pickle(kline_path)
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Mining failed for {symbol}: {e}")
        return False

async def main():
    """
    Main orchestration logic for the data mining service.
    
    Implements a batched execution model to prevent exchange rate-limiting 
    errors during massive historical data collection.
    """
    client = await AsyncClient.create()
    
    # Identify tickers requiring initial or delta downloads
    symbols = [s for s in MANUAL_BINANCE_FUTURES_TICKERS if not os.path.exists(f"{KLINES_DIR}/{s}_1m.pkl")]
    
    total_symbols = len(symbols)
    print(f"[SYSTEM] Initializing 1-Year Market Mining for {total_symbols} assets.")

    # Control batch size to manage concurrent network I/O
    batch_size = 3 
    for i in range(0, total_symbols, batch_size):
        batch = symbols[i : i + batch_size]
        tasks = [download_symbol_data(client, s) for s in batch]
        await asyncio.gather(*tasks)
        
        # UI Progress Indicator
        progress = min(i + batch_size, total_symbols)
        percent = (progress / total_symbols) * 100
        sys.stdout.write(f"\r[MINING] Cumulative Progress: {percent:.2f}% [{progress}/{total_symbols}]")
        sys.stdout.flush()
        
        # Anti-ban sleep interval
        await asyncio.sleep(0.5)

    await client.close_connection()
    print("\n[SYSTEM] Historical Mining Operation Concluded.")

if __name__ == "__main__":
    asyncio.run(main())