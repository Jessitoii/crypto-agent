"""
Global Configuration Interface

This module centralizes all system configurations, including API credentials,
LLM parameters, exchange connectivity, and filtering heuristics.
Environment variables are loaded via python-dotenv for security and flexibility.
"""

import os
from pathlib import Path

# --- Project Directory Routing ---
# Dynamically resolve absolute paths for data and module directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NEXUS_DIR = BASE_DIR / "nexus"

from dotenv import load_dotenv

# --- Prompt Manifest ---
# Centralized imports for AI instruction sets
from prompts import (
    SYSTEM_PROMPT,
    ANALYZE_SPECIFIC_PROMPT,
    DETECT_SYMBOL_PROMPT,
    GENERATE_SEARCH_QUERY_PROMPT,
    GET_COIN_PROFILE_PROMPT,
    ANALYZE_GENERAL_PROMPT
)

# Initialize environment variables from .env file
load_dotenv()

# --- LLM Infrastructure Configuration ---
# Configures the primary reasoning engine for news analysis
USE_GROQCLOUD = True
GROQCLOUD_API_KEY = os.getenv('GROQCLOUD_API_KEY')
GROQCLOUD_MODEL = os.getenv('GROQCLOUD_MODEL', 'google/gemini-2.0-flash-exp:free')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
USE_GEMINI = os.getenv('USE_GEMINI', 'False').lower() == 'true'

LLM_CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "temperature": 0.0,
    "num_ctx": 4096,
    "max_tokens": 256,
}

# --- Exchange Connectivity Configuration ---
# Switches between Binance Mainnet and Testnet environments
USE_MAINNET = True
REAL_TRADING_ENABLED = False
    
if USE_MAINNET:
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET')
    IS_TESTNET = False
else:
    API_KEY = os.getenv('BINANCE_DEMO_API_KEY')
    API_SECRET = os.getenv('BINANCE_DEMO_API_SECRET')
    IS_TESTNET = True

# WebSocket endpoint for real-time market data ingestion
BASE_URL = os.getenv('BASE_URL', "wss://stream.binance.com:9443/ws")
WEBSOCKET_URL = BASE_URL

# --- Ingestion Source Configuration ---
# High-signal Telegram channels for market Alpha
TARGET_CHANNELS = ['cointelegraph', 'wublockchainenglish', 'CryptoRankNews', 'TheBlockNewsLite', 'coindesk', 'arkhamintelligence', 'glassnode'] 

# Aggregated RSS feeds for supplemental news coverage
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptopotato.com/feed/",
    "https://u.today/rss",
    "https://beincrypto.com/feed/"
]

# --- Telegram Infrastructure Configuration ---
# Telethon client parameters for secure session management
API_ID = int(os.getenv('API_ID', 0))
API_HASH = os.getenv('API_HASH')
TELETHON_SESSION_NAME = os.getenv('TELETHON_SESSION_NAME')

# --- Simulation & Risk Management Parameters ---
# Defines the economic constraints for paper trading sessions
STARTING_BALANCE = 1000
LEVERAGE = 10 
FIXED_TRADE_AMOUNT = 8

# --- Content Filtering & Heuristic Heuristics ---
# Keywords used to discard low-signal or summary-style news items
IGNORE_KEYWORDS = [
    'daily', 'digest', 'recap', 'summary', 'analysis', 'price analysis', 
    'prediction', 'overview', 'roundup', 'market wrap', 'outlook', 
    'forecast', 'top gainer', 'top loser', 'market update',
    'slides', 'declines', 'drops', 'plummet' 
]

# Tickers that frequently collide with common English words to avoid false positives
DANGEROUS_TICKERS = {
    'S', 'THE', 'A', 'I', 'IS', 'TO', 'IT', 'BY', 'ON', 'IN', 'AT', 'OF', 'M',
    'ME', 'MY', 'UP', 'DO', 'GO', 'OR', 'IF', 'BE', 'AS', 'WE', 'SO',
    'NEAR', 'ONE', 'SUN', 'GAS', 'POL', 'BOND', 'OM', 'ELF', 'MEME', 'AI', 'MOVE', 'LINK'
}

# Mapping of colliding tickers to their full project names for disambiguation
AMBIGUOUS_COINS = {
    'link': 'Chainlink',
    'one': 'Harmony',
    'pol': 'Polygon',
    'gas': 'NeoGas',
    'sun': 'Sun',
    'just': 'Just',
    'omg': 'OMG Network', 
    'meme': 'Memecoin', 
    'beta': 'Beta Finance', 
    'iot': 'Helium IOT', 
    'pump': 'Pump.fun',
}