import requests
from ddgs import DDGS
import asyncio
import re
from config import DANGEROUS_TICKERS, AMBIGUOUS_COINS

def get_top_pairs(limit=50):
    """Fetches top volume USDT pairs from Binance in the last 24h."""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url).json()
        
        # Filter USDT pairs (excluding leveraged tokens and stablecoins)
        filtered = [
            x for x in response 
            if x['symbol'].endswith('USDT') 
            and 'UPUSDT' not in x['symbol'] 
            and 'DOWNUSDT' not in x['symbol']
            and x['symbol'] not in ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT']
        ]
        
        # Sort by quoteVolume and take top X
        sorted_pairs = sorted(filtered, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]
        
        return [x['symbol'].lower() for x in sorted_pairs]
    except Exception as e:
        print(f"ERROR: Could not fetch pair list! {e}")
        return ['btcusdt', 'ethusdt', 'bnbusdt', 'solusdt']

def get_top_100_map():
    """Returns a map of top 100 coins by market cap from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": "false"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        coin_data = {}
        for coin in data:
            coin_data[coin['name'].lower()] = {
                'symbol': coin['symbol'].lower(),
                'cap': coin['market_cap'] if coin['market_cap'] else 0
            }
            # Also index by symbol for quick access
            coin_data[coin['symbol'].lower()] = {
                'symbol': coin['symbol'].lower(),
                'cap': coin['market_cap'] if coin['market_cap'] else 0,
                'name': coin['name']
            }
            
        return coin_data

    except Exception as e:
        print(f"[ERROR] CoinGecko request failed: {e}")
        return {}

def search_web_sync(query):
    """Performs synchronous search on DuckDuckGo (to be run in thread)."""
    try:
        results = DDGS().text(query, max_results=2)
        if not results:
            return "No search results found."
        
        summary = "WEB SEARCH RESULTS:\n"
        for res in results:
            summary += f"- {res['title']}: {res['body']}\n"
        
        print(f"Search completed for: {query}")
        print(summary)
        return summary
    except Exception as e:
        return f"Search Error: {e}"

async def perform_research(query):
    """Performs non-blocking research using a thread pool."""
    return await asyncio.to_thread(search_web_sync, query)

def clean_coin_map(raw_map):
    """Cleans raw coin map into {Symbol: Name} format."""
    clean_map = {}
    if not raw_map: 
        return {}
    
    for key, value in raw_map.items():
        if isinstance(value, dict):
            symbol = value.get('symbol', '').upper()
            name = value.get('name', '').lower()
            if symbol: clean_map[symbol] = name
        else:
            clean_map[str(key).upper()] = str(value).lower()
    return clean_map

def find_coins(msg, coin_map=None):
    if not msg: 
        return []
    
    coin_map = clean_coin_map(coin_map)
    detected_symbols = set()
    
    msg_lower = msg.lower()
    msg_upper = msg.upper()
    
    context_pattern = r'(protocol|network|token|coin|dao|chain|finance|labs|swap)'

    for symbol, full_name in coin_map.items():
        if check_is_stablecoin(symbol):
            continue

        # CASE 1: DANGEROUS TICKERS (e.g., THE, IT, IS)
        # Rule: Only match if uppercase and near a context word.
        if symbol in DANGEROUS_TICKERS:
            strict_pattern = rf'\b{symbol}\b\s+{context_pattern}'
            if re.search(strict_pattern, msg_upper, re.IGNORECASE):
                detected_symbols.add(symbol)
            continue

        # CASE 2: AMBIGUOUS COINS (e.g., LINK, GAS, NEAR)
        if symbol in AMBIGUOUS_COINS:
            if rf'\b{symbol}\b' in msg_upper: 
                if re.search(rf'\b{symbol}\b', msg_upper):
                    detected_symbols.add(symbol)
                    continue
            
            special_name = AMBIGUOUS_COINS[symbol].lower()
            if special_name in msg_lower:
                detected_symbols.add(symbol)
            continue

        # CASE 3: STANDARD COINS
        if re.search(rf'\b{symbol}\b', msg_upper):
            detected_symbols.add(symbol)
        
        elif full_name and len(full_name) > 2:
            if rf' {full_name} ' in f' {msg_lower} ': 
                detected_symbols.add(symbol)

    return [f"{s}USDT" for s in detected_symbols]

def check_is_stablecoin(symbol):
    try:
        return "stablecoin" in coin_categories.get(symbol, "").lower()
    except Exception:
        return False
    
coin_categories = {
    # --- TOP 10 & MAJORS ---
    'BTC': 'Layer-1 (Store of Value)',
    'ETH': 'Layer-1 (Smart Contract Platform)',
    'SOL': 'Layer-1 (High Performance)',
    'BNB': 'Exchange Token / Layer-1',
    'XRP': 'Layer-1 (Payments)',
    'ADA': 'Layer-1',
    'AVAX': 'Layer-1',
    'TRX': 'Layer-1',
    'DOGE': 'Meme Coin (OG)',
    'DOT': 'Layer-0 (Interoperability)',
    'LINK': 'Oracle (Infrastructure)',
    'LTC': 'Layer-1 (Payments)',
    'BCH': 'Layer-1 (Payments)',
    'NEAR': 'Layer-1 (AI focus)',
    'MATIC': 'Layer-2 (Polygon)', 
    'POL': 'Layer-2 (Polygon)',
    'DAI': 'Stablecoin (Decentralized)',
    'UNI': 'DeFi (DEX)',
    'LEO': 'Exchange Token',
    'WBTC': 'Wrapped Asset',

    # --- STABLECOINS ---
    'USDT': 'Stablecoin',
    'USDC': 'Stablecoin',
    'FDUSD': 'Stablecoin',
    'TUSD': 'Stablecoin',
    'USDE': 'Stablecoin (Ethena)',
    'PYUSD': 'Stablecoin (PayPal)',
    'USDS': 'Stablecoin',
    'GUSD': 'Stablecoin',

    # --- ARTIFICIAL INTELLIGENCE (AI) & DATA ---
    'FET': 'AI & Big Data',
    'RNDR': 'AI & Rendering', 
    'RENDER': 'AI & Rendering',
    'TAO': 'AI (Decentralized Intelligence)',
    'WLD': 'AI & Identity',
    'ARKM': 'AI & Data Intelligence',
    'GRT': 'AI & Data Indexing',
    'AGIX': 'AI (SingularityNET)',
    'OCEAN': 'AI & Data',
    'ASI': 'AI (Superalliance)',
    'AKT': 'AI & Cloud (Akash)',
    'AIOZ': 'AI & DePIN',
    'GLM': 'AI & Computing',
    'PRIME': 'AI & Gaming',
    'ABT': 'AI & Data',
    'NMR': 'AI & Data',

    # --- MEME COINS ---
    'SHIB': 'Meme Coin',
    'PEPE': 'Meme Coin',
    'WIF': 'Meme Coin (Solana)',
    'BONK': 'Meme Coin (Solana)',
    'FLOKI': 'Meme Coin',
    'BOME': 'Meme Coin',
    'MEME': 'Meme Coin',
    'DOGS': 'Meme Coin (Ton)',
    'NOT': 'Meme / Gaming (Ton)',
    'BRETT': 'Meme Coin (Base)',
    'POPCAT': 'Meme Coin',
    'MOG': 'Meme Coin',
    'NEIRO': 'Meme Coin',
    'TURBO': 'Meme Coin',
    'PEOPLE': 'Meme / DAO',
    '1000SATS': 'Meme / BRC-20',
    'ORDI': 'Meme / BRC-20',

    # --- LAYER-1 (Alternatives) ---
    'SUI': 'Layer-1 (Move)',
    'APT': 'Layer-1 (Move)',
    'SEI': 'Layer-1 (Trading)',
    'TON': 'Layer-1 (Telegram)',
    'KAS': 'Layer-1 (PoW)',
    'TIA': 'Layer-1 (Modular)',
    'INJ': 'Layer-1 (Finance)',
    'ATOM': 'Layer-0 (Cosmos)',
    'HBAR': 'Layer-1 (Enterprise)',
    'ALGO': 'Layer-1',
    'ICP': 'Layer-1 (Internet Computer)',
    'FTM': 'Layer-1 (Fantom/Sonic)',
    'S' : 'Layer-1 (Sonic)',
    'EGLD': 'Layer-1',
    'XTZ': 'Layer-1',
    'FLOW': 'Layer-1 (NFT)',
    'MINA': 'Layer-1 (ZK)',
    'KDA': 'Layer-1',
    'ZIL': 'Layer-1',
    'IOTA': 'Layer-1 (IoT)',
    'XLM': 'Layer-1 (Payments)',
    'EOS': 'Layer-1',
    'HYPE': 'Layer-1 (Hyperliquid)',

    # --- LAYER-2 (Scaling) ---
    'ARB': 'Layer-2 (Optimistic)',
    'OP': 'Layer-2 (Optimistic)',
    'STX': 'Layer-2 (Bitcoin)',
    'IMX': 'Layer-2 (Gaming)',
    'MNT': 'Layer-2 (Mantle)',
    'STRK': 'Layer-2 (ZK)',
    'ZK': 'Layer-2 (ZKsync)',
    'MANTA': 'Layer-2',
    'METIS': 'Layer-2',
    'SCR': 'Layer-2',
    
    # --- DEFI (Decentralized Finance) ---
    'UNI': 'DeFi (DEX)',
    'AAVE': 'DeFi (Lending)',
    'MKR': 'DeFi (DAO)',
    'LDO': 'DeFi (Liquid Staking)',
    'RPL': 'DeFi (Liquid Staking)',
    'FXS': 'DeFi (Stable/LSD)',
    'CRV': 'DeFi (Stable Swap)',
    'SNX': 'DeFi (Derivatives)',
    'DYDX': 'DeFi (Derivatives)',
    'GMX': 'DeFi (Perp DEX)',
    'JUP': 'DeFi (Solana Aggregator)',
    'RAY': 'DeFi (Solana DEX)',
    'CAKE': 'DeFi (BSC DEX)',
    '1INCH': 'DeFi (Aggregator)',
    'RUNE': 'DeFi (Cross-chain)',
    'PENDLE': 'DeFi (Yield Trading)',
    'ENA': 'DeFi (Synthetic Dollar)',
    'COMP': 'DeFi (Lending)',
    'LRC': 'DeFi (Exchange)',
    'CVX': 'DeFi (Yield)',

    # --- REAL WORLD ASSETS (RWA) ---
    'ONDO': 'RWA (Tokenized Securities)',
    'OM': 'RWA (Mantra)',
    'TRU': 'RWA (Credit)',
    'POLYX': 'RWA (Regulatory)',
    'CFG': 'RWA (Centrifuge)',
    'GFI': 'RWA (Credit)',

    # --- GAMING & METAVERSE ---
    'SAND': 'Gaming/Metaverse',
    'MANA': 'Gaming/Metaverse',
    'AXS': 'Gaming (P2E)',
    'GALA': 'Gaming',
    'ENJ': 'Gaming',
    'BEAM': 'Gaming (Infrastructure)',
    'APE': 'Metaverse / NFT',
    'PIXEL': 'Gaming',
    'ILV': 'Gaming',
    'YGG': 'Gaming Guild',
    
    # --- ORACLE & INFRASTRUCTURE ---
    'PYTH': 'Oracle',
    'TRB': 'Oracle',
    'API3': 'Oracle',
    'JASMY': 'IoT / Data',
    'ENS': 'Infrastructure (Identity)',
    'ETHFI': 'Infrastructure (Restaking)',
    'REZ': 'Infrastructure (Restaking)',
    'ALT': 'Infrastructure (Rollups)',

    # --- EXCHANGE TOKENS ---
    'OKB': 'Exchange Token',
    'KCS': 'Exchange Token',
    'CRO': 'Exchange Token',
    'BGB': 'Exchange Token',
    'GT': 'Exchange Token',
    'HT': 'Exchange Token',

    # --- PRIVACY ---
    'XMR': 'Privacy Coin',
    'ZEC': 'Privacy Coin',
    'ROSE': 'Privacy / Layer-1',

    # --- CLASSIC / OLD GEN ---
    'ETC': 'Layer-1 (Classic)',
    'LUNA': 'Layer-1 (Reborn)',
    'LUNC': 'Layer-1 (Classic/Meme)',
    'USTC': 'Stablecoin (Failed/Meme)',
    'EOS': 'Layer-1 (Classic)',
    'NEO': 'Layer-1 (Classic)',
    'QTUM': 'Layer-1 (Classic)',
    'BAT': 'Browser / Ad',
    'CHZ': 'Fan Tokens / Sports',
    'HOT': 'Infrastructure',
    'RVN': 'Layer-1 (PoW)'
}