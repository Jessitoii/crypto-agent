import asyncio
import os
import pandas as pd
import sys
from binance import AsyncClient
from datetime import datetime, timedelta, timezone

from utils import get_top_100_map, check_is_stablecoin
COIN_MAP = get_top_100_map()
MANUAL_BINANCE_FUTURES_TICKERS = [
    # Major Coins
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    # Layer 1 & 2
    "NEARUSDT", "ATOMUSDT", "ALGOUSDT", "FTMUSDT", "APTUSDT", "SUIUSDT", "SEIUSDT", "OPUSDT", "ARBUSDT", "HBARUSDT",
    "INJUSDT", "LDOUSDT", "TIAUSDT", "STXUSDT", "EGLDUSDT", "FILUSDT", "ICPUSDT", "RUNEUSDT", "GRTUSDT", "AAVEUSDT",
    # AI & Infrastructure
    "FETUSDT", "RENDERUSDT", "TAOUSDT", "NEARUSDT", "AGIXUSDT", "WLDUSDT", "ARKMUSDT", "THETAUSDT",
    # Memecoins
    "DOGEUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "1000BONKUSDT", "1000FLOKIUSDT", "WIFUSDT", "PEOPLEUSDT", "MEMEUSDT",
    "POPCATUSDT", "BOMEUSDT", "1000LUNCUSDT", "1000RATSUSDT",
    # DeFi
    "UNIUSDT", "SUSHIUSDT", "DYDXUSDT", "CRVUSDT", "MKRUSDT", "SNXUSDT", "PENDLEUSDT", "ENAUSDT", "ETHFIUSDT",
    "JUPUSDT", "PYTHUSDT", "STRKUSDT", "AXSUSDT", "IMXUSDT", "GALAUSDT", "BEAMXUSDT", "SANDUSDT", "MANAUSDT",
    # Legacy & Others
    "LTCUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT", "TRXUSDT", "VETUSDT", "NEOUSDT", "QTUMUSDT", "EOSUSDT", "IOTAUSDT",
    "ZECUSDT", "DASHUSDT", "XMRUSDT", "ONTUSDT", "ZILUSDT", "BATUSDT", "ENJUSDT", "KNCUSDT", "ANKRUSDT", "OCEANUSDT",
    "CHZUSDT", "ALICEUSDT", "FLOWUSDT", "KAVAUSDT", "GMXUSDT", "ORDIUSDT", "1000SATSUSDT", "GASUSDT", "TRBUSDT"
]
BASE_DIR = "data/market_cache"
KLINES_DIR = f"{BASE_DIR}/klines"
FUNDING_DIR = f"{BASE_DIR}/funding"

for d in [KLINES_DIR, FUNDING_DIR]:
    if not os.path.exists(d): os.makedirs(d)

async def download_symbol_data(client, symbol):
    """
    Downloads 1 year of kline and funding data for a given symbol.
    """
    try:
        # 1. Candle Data (1m Klines)
        kline_path = f"{KLINES_DIR}/{symbol}_1m.pkl"
        if not os.path.exists(kline_path):
            klines = []
            gen = await client.futures_historical_klines_generator(symbol, "1m", "1 year ago UTC")
            async for k in gen:
                klines.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[7])])
            
            if klines:
                df_k = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df_k.to_pickle(kline_path)
        
        return True
    except Exception as e:
        print(f"\n[ERROR] {symbol} fetch failed: {e}")
        return False

async def main():
    client = await AsyncClient.create()
    
    # Filter symbols that already have cached data
    symbols = [s for s in MANUAL_BINANCE_FUTURES_TICKERS if not os.path.exists(f"{KLINES_DIR}/{s}_1m.pkl")]
    
    total_symbols = len(symbols)
    print(f"[SYSTEM] Starting 1-year kline mining for {total_symbols} coins.")

    batch_size = 3 # Controlled speed to prevent rate limiting
    for i in range(0, total_symbols, batch_size):
        batch = symbols[i : i + batch_size]
        tasks = [download_symbol_data(client, s) for s in batch]
        await asyncio.gather(*tasks)
        
        progress = min(i + batch_size, total_symbols)
        percent = (progress / total_symbols) * 100
        sys.stdout.write(f"\r[MINING] Progress: {percent:.2f}% [{progress}/{total_symbols}]")
        sys.stdout.flush()
        
        await asyncio.sleep(0.5)

    await client.close_connection()
    print("\n[SYSTEM] Mining completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())