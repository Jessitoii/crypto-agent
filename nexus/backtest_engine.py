"""
NEXUS Forensic Backtest & Simulation Engine

This module provides a comprehensive offline simulation environment for 
validating NEXUS model performance against historical Telegram signals and 
real market outcomes.

It supports both 'Local Brain' (DeBERTa-v3) and 'Remote Brain' (LLM) 
inference modes, providing detailed trade logs, missed opportunity analysis, 
and technical confluence auditing.
"""

import asyncio
import time
import os
import json
import torch
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
import re

# Simulation Configuration: Toggle between local neural net and remote LLM reasoning
LOCAL_BRAIN_MODE = True
if LOCAL_BRAIN_MODE:
    from model import NexusPredictor
    local_brain = NexusPredictor("standard_deberta_nexus")
else:
    local_brain = None

# Core Project Dependencies
from config import TARGET_CHANNELS, API_ID, API_HASH, TELETHON_SESSION_NAME, STARTING_BALANCE, DATA_DIR
from main import BotContext, SharedState
from binance_client import BinanceExecutionEngine
from services import process_news, ensure_fresh_data
from utils import find_coins, get_top_100_map
from price_buffer import PriceBuffer
from exchange import PaperExchange
from brain import AgentBrain
from config import GROQCLOUD_API_KEY, GROQCLOUD_MODEL, GOOGLE_API_KEY, GEMINI_MODEL

def clean_news_text(text):
    """
    Sanitizes raw Telegram messages for model consumption.
    
    Removes URLs, handles, markdown artifacts, and formatting noise 
    to isolate the core news headline.
    
    Args:
        text (str): Raw message content.
        
    Returns:
        str: Cleaned headline.
    """
    # 1. Forensic URL extraction/removal
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # 2. Handle neutralization
    text = re.sub(r'@\w+', '', text)
    # 3. Artifact scrubbing
    text = re.sub(r'\[News\]\(.*?\)|\[Markets\]\(.*?\)|\[YouTube\]\(.*?\)', '', text, flags=re.IGNORECASE)
    text = text.replace('**', '')
    text = text.replace('NOW:', '').replace('BREAKING:', '')
    text = text.replace("[— link]( ", "")
    # Normalize whitespace
    return " ".join(text.split()).strip()

class MockMemory:
    """Mock database interface for stateless backtesting."""
    def is_duplicate(self, text): return False, 0.0
    def add_news(self, source, content): pass
    def log_decision(self, record): return 999 
    def log_trade(self, record, decision_id=None): pass

async def get_historical_technicals(ctx, pair, msg_ts):
    """
    Reconstructs the technical market state at a specific historical point.
    
    Args:
        ctx (BotContext): Execution context.
        pair (str): Trading pair.
        msg_ts (float): Event timestamp.
        
    Returns:
        dict: RSI, Momentum, and BTC correlation metrics.
    """
    # 1. Fetch 100 minutes of historical data for RSI hydration
    klines = await ctx.real_exchange.client.futures_klines(
        symbol=pair.upper(),
        interval='1m',
        endTime=int(msg_ts * 1000),
        limit=100
    )
    
    if not klines: return None

    # Populate temporary PriceBuffer for calculation parity with live bot
    temp_buffer = PriceBuffer()
    for k in klines:
        temp_buffer.update_candle(float(k[4]), k[0]/1000, True)
    
    temp_buffer.current_price = float(klines[-1][4])
    
    # 2. Extract BTC trend for correlation analysis
    btc_klines = await ctx.real_exchange.client.futures_klines(
        symbol="BTCUSDT", interval='1m', endTime=int(msg_ts * 1000), limit=60
    )
    
    btc_trend = 0.0
    if btc_klines:
        btc_start, btc_end = float(btc_klines[0][4]), float(btc_klines[-1][4])
        btc_trend = ((btc_end - btc_start) / btc_start) * 100

    return {
        'price': temp_buffer.current_price,
        'rsi': temp_buffer.calculate_rsi(),
        'changes': temp_buffer.get_all_changes(),
        'btc_trend': btc_trend,
    }

coin_map = get_top_100_map()

async def simulate_process_news(message, ctx, f_log):
    """
    Simulates the live news processing pipeline using historical data.
    """
    msg_text = message.text
    msg_ts = message.date.timestamp()
    msg_dt = message.date.strftime("%Y-%m-%d %H:%M:%S")

    # --- 1. IDEMPOTENCY FILTER ---
    is_dup, _ = ctx.memory.is_duplicate(msg_text)
    if is_dup: return

    # --- 2. ENTITY DETECTION ---
    detected_pairs = find_coins(msg_text, coin_map=coin_map)
    
    if not detected_pairs:
        # LLM fallback for non-regex detectable tickers
        found_symbol = await ctx.brain.detect_symbol(msg_text, coin_map)
        if found_symbol:
            detected_pairs.append(f"{found_symbol.lower()}usdt")

    if not detected_pairs: return

    # --- 3. CORE ANALYTICAL LOOP ---
    for pair in detected_pairs:
        try:
            # Reconstruct market context
            klines = await ctx.real_exchange.client.futures_klines(
                symbol=pair.upper(), interval='1m', startTime=int(msg_ts * 1000), limit=61
            )
            if not klines: continue

            entry_price = float(klines[0][4]) 
            tech = await get_historical_technicals(ctx, pair, msg_ts)
            if not tech: continue

            # Classify technical state for instruction injection
            rsi, momentum = tech['rsi'], tech['changes']["1h"]
            rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            mom_label = "BULLISH_MOM" if momentum > 0.5 else "BEARISH_MOM" if momentum < -0.5 else "FLAT"

            # Execute Model Inference
            news_clean = clean_news_text(msg_text)
            formatted_input = f"[N] {news_clean} [C] {pair.upper()} [RSI] {rsi_label} [MOM] {mom_label}"
            analysis = ctx.local_brain.analyze(formatted_input)
            action, confidence = analysis["action"], analysis["confidence"]

            # --- 4. TRADE EXECUTION SIMULATION ---
            if confidence >= 75 and action in ["LONG", "SHORT"]:
                trade_amount, leverage = 100, 10
                tp_pct, sl_pct = 2.0, 1.0

                report_entry = (
                    f"\n{'='*60}\n"
                    f"[TRADE] SIMULATED ENTRY | {msg_dt}\n"
                    f"{'-'*60}\n"
                    f"NEWS: {news_clean}\n"
                    f"TARGET: {pair.upper()}\n"
                    f"TECH: RSI={rsi:.2f} ({rsi_label}) | MOM={momentum:.2f} ({mom_label})\n"
                    f"AI DECISION: {action} ({confidence:.2f}%)\n"
                    f"{'-'*60}\n"
                )
                
                # Open position in paper exchange
                ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=tp_pct, sl_pct=sl_pct,
                    amount_usdt=100, leverage=leverage, validity=30,
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                
                # Forward-track price movement to resolve trade outcome
                for k in klines:
                    minute_ts = k[0] / 1000
                    ticks = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                    for i, tick_price in enumerate(ticks):
                        current_ts = minute_ts + (i * 15)
                        res_log, _, _, pnl, peak, _ = ctx.exchange.check_positions_test(
                            pair, tick_price, now_ts=current_ts
                        )
                        
                        if res_log:
                            close_dt = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S")
                            report_exit = (
                                f"[RESULT] TRADE CLOSED ({close_dt}):\n"
                                f"   - Status: {res_log} | PnL: {pnl:.2f} USDT\n"
                                f"{'='*60}\n"
                            )
                            f_log.write(report_entry + report_exit)
                            f_log.flush()
                            return

            # --- 5. COUNTER-FACTUAL ANALYSIS (HOLD Discipline) ---
            elif action == "HOLD":
                future_candles = klines[1:21] # Check 20min window
                if future_candles:
                    max_p = max([float(k[2]) for k in future_candles])
                    min_p = min([float(k[3]) for k in future_candles])
                    move_up = ((max_p - entry_price) / entry_price) * 100
                    move_down = ((min_p - entry_price) / entry_price) * 100
                    
                    missed = "LONG" if move_up >= 1.5 else "SHORT" if move_down <= -1.5 else None
                    if missed:
                        f_log.write(f"\n[MISSED] {missed} Opportunity: {max(abs(move_up), abs(move_down)):.2f}% on {pair} at {msg_dt}\n")
                        f_log.flush()

        except Exception as e:
            print(f"[ERROR] Simulation fault on {pair}: {e}")

async def process_offline_entry(entry, ctx, f_log):
    """
    Processes a pre-cached offline event entry for rapid backtesting.
    """
    try:
        msg_text = entry["msg_text"]
        msg_dt = entry["msg_dt"]
        pair = entry["pair"]
        technicals = entry["technicals"]
        outcomes = entry["outcomes"]
        msg_ts = entry["msg_ts"]
        
        rsi, momentum = technicals["rsi"], technicals["momentum_1h"]
        news_clean = clean_news_text(msg_text)
        
        # Inference Pass
        if LOCAL_BRAIN_MODE:
            analysis = ctx.local_brain.predict(news_text=news_clean, symbol=pair.replace("USDT", ""))
            action, confidence = analysis["decision"], analysis["confidence"]
        else:
            analysis = await ctx.brain.analyze_specific_no_research(news=news_clean, symbol=pair.replace("USDT", ""))
            action, confidence = analysis.get("action", "HOLD"), analysis.get("conviction_score", 0)

        # Simulation Logic (Resolved via cached outcomes)
        if confidence >= 75 and action in ["LONG", "SHORT"]:
            entry_price = technicals["close"]
            ctx.exchange.open_position_test(
                symbol=pair, side=action, price=entry_price,
                tp_pct=2.0, sl_pct=1.0,
                amount_usdt=100, leverage=10, validity=30,
                app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
            )
            
            for k in outcomes.get("future_candles", []):
                 ticks = [k['o'], k['h'], k['l'], k['c']]
                 for i, tick_price in enumerate(ticks):
                    r_log, _, _, pnl, _, _ = ctx.exchange.check_positions_test(pair, tick_price, now_ts=k['ts'] + (i*15))
                    if r_log:
                        f_log.write(f"[TRADE] {pair} | {action} | {r_log} | PnL: {pnl:.2f}\n")
                        return

        elif action == "HOLD":
            max_g = outcomes.get("max_gain_20m", 0.0)
            if max_g >= 1.5:
                f_log.write(f"[MISSED] {pair} | LONG | +{max_g:.2f}% | {msg_dt}\n")

    except Exception as e:
        print(f"[ERROR] Offline playback failure: {e}")

async def run_simulation():
    """
    Orchestrates the backtest suite execution.
    """
    print("[SYSTEM] Hydrating Backtest Environment...")
    ctx = BotContext()
    ctx.app_state, ctx.memory = SharedState(), MockMemory()
    ctx.exchange, ctx.local_brain = PaperExchange(1000.0), local_brain
    
    # Initialize optional brain components
    ctx.brain = AgentBrain(use_groqcloud=False, api_key=GROQCLOUD_API_KEY, groqcloud_model=GROQCLOUD_MODEL)
    
    offline_file = str(DATA_DIR / "offline_test_data.json")
    results_file = str(DATA_DIR / "backtest_results_nexus.txt")
    
    if os.path.exists(offline_file):
        print(f"[SUCCESS] Offline cache identified: {offline_file}")
        with open(offline_file, "r", encoding="utf-8") as f_in:
             offline_data = json.load(f_in)
        
        with open(results_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n--- NEXUS FORENSIC RUN: {datetime.now()} ---\n")
            for i, entry in enumerate(offline_data):
                if i % 50 == 0: sys.stdout.write(f"\rProgress: {i}/{len(offline_data)}")
                await process_offline_entry(entry, ctx, f_out)
                
        print(f"\n[SUCCESS] Simulation complete. Logs serialized to {results_file}")
    else:
        print("[ERROR] No offline cache found. Aborting.")

if __name__ == "__main__":
    asyncio.run(run_simulation())