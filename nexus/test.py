import asyncio
import time
import os
import json
import torch
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
from setfit import SetFitModel
import re

LOCAL_BRAIN_MODE = True
if LOCAL_BRAIN_MODE:
    from model import NexusPredictor
    local_brain = NexusPredictor("standard_deberta_nexus")
else:
    local_brain = None

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
    # 1. Clean URLs (http, https)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # 2. Clean Telegram/Twitter handles (@cointelegraph etc.)
    text = re.sub(r'@\w+', '', text)
    # 3. Clean Markdown link remnants and button text
    text = re.sub(r'\[News\]\(.*?\)|\[Markets\]\(.*?\)|\[YouTube\]\(.*?\)', '', text, flags=re.IGNORECASE)
    # 4. Remove bold (**) markers
    text = text.replace('**', '')
    # 5. Clean unnecessary whitespace and headers
    text = text.replace('**', '').replace('NOW:', '').replace('BREAKING:', '')
    text = text.replace("[— link]( ", "")
    # Remove extra spaces
    return " ".join(text.split()).strip()

# 1. MOCK OBJECT TO DISABLE DATABASE
class MockMemory:
    def is_duplicate(self, text): return False, 0.0
    def add_news(self, source, content): pass
    def log_decision(self, record): return 999 # Fake ID
    def log_trade(self, record, decision_id=None): pass

# [NEW] SETFIT MODEL WRAPPER

async def get_historical_technicals(ctx, pair, msg_ts):
    """Calculates technical metrics at the time of the news release."""
    # 1. Fetch 100 minutes of historical data for RSI calculation
    klines = await ctx.real_exchange.client.futures_klines(
        symbol=pair.upper(),
        interval='1m',
        endTime=int(msg_ts * 1000),
        limit=100
    )
    
    if not klines:
        return None

    # Initialize and populate buffer
    temp_buffer = PriceBuffer()
    for k in klines:
        # (price, timestamp, is_closed)
        temp_buffer.update_candle(float(k[4]), k[0]/1000, True)
    
    # Set current price to last close
    temp_buffer.current_price = float(klines[-1][4])
    
    # 2. Fetch BTC trend
    btc_klines = await ctx.real_exchange.client.futures_klines(
        symbol="BTCUSDT",
        interval='1m',
        endTime=int(msg_ts * 1000),
        limit=60
    )
    
    btc_trend = 0.0
    if btc_klines:
        btc_start = float(btc_klines[0][4])
        btc_end = float(btc_klines[-1][4])
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
    Simulation version of services.py -> process_news().
    """
    msg_text = message.text
    msg_ts = message.date.timestamp()
    msg_dt = message.date.strftime("%Y-%m-%d %H:%M:%S")

    # --- 1. FILTERING (is_duplicate logic) ---
    is_dup, _ = ctx.memory.is_duplicate(msg_text)
    if is_dup: return

    # --- 2. COIN DETECTION (Regex + AI Fallback) ---
    detected_pairs = find_coins(msg_text, coin_map=coin_map)
    
    if not detected_pairs:
        # AI Fallback: If regex fails, use brain for entity extraction.
        found_symbol = await ctx.brain.detect_symbol(msg_text, coin_map)
        if found_symbol:
            pot_pair = f"{found_symbol.lower()}usdt"
            detected_pairs.append(pot_pair)

    if not detected_pairs:
        return

    # --- 3. ANALYSIS LOOP ---
    for pair in detected_pairs:
        try:
            # A) Historical Data Retrieval (60-minute window)
            klines = await ctx.real_exchange.client.futures_klines(
                symbol=pair.upper(),
                interval='1m',
                startTime=int(msg_ts * 1000),
                limit=61 # Analysis + 60min tracking
            )
            if not klines: continue

            # Entry Price
            entry_price = float(klines[0][4]) 
            
            # Technical Data
            tech = await get_historical_technicals(ctx, pair, msg_ts)
            if not tech: continue

            print(f"[TECHNICAL] Data received for {pair}: RSI: {tech['rsi']:.2f} | BTC 1h: {tech['btc_trend']:.2f}%")

            # Safe Dictionary Access and Details
            clean_symbol = pair.lower().replace("usdt", "")
            c_data = coin_map.get(clean_symbol)
            if isinstance(c_data, dict):
                coin_full_name = c_data.get("name", "Unknown").title()
                m_cap = c_data.get("cap", 0)
            else:
                coin_full_name = "Unknown"
                m_cap = 0

            # Market Cap Formatting
            if m_cap > 1_000_000_000:
                cap_str = f"${m_cap / 1_000_000_000:.2f} BILLION"
            elif m_cap > 1_000_000:
                cap_str = f"${m_cap / 1_000_000:.2f} MILLION"
            else:
                cap_str = "UNKNOWN/SMALL"

            # B) Model-based inference
            rsi = tech['rsi']
            momentum = tech['changes']["1h"]
            rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            mom_label = "BULLISH_MOM" if momentum > 0.5 else "BEARISH_MOM" if momentum < -0.5 else "FLAT"

            msg_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', msg_text)
            msg_text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', msg_text)
            formatted_input = f"[N] {msg_text} [C] {pair.upper()} [MC] {cap_str} [RSI] {rsi_label} [MOM] {mom_label} [F]0.0"
            analysis = ctx.local_brain.analyze(formatted_input)
            action = analysis["action"]
            confidence = analysis["confidence"]

            print(f"[AI] NEXUS decision for {pair}: {action} (Confidence: {confidence:.2f}%)")

            # C) Order Execution
            if confidence >= 75 and action in ["LONG", "SHORT"]:
                
                # --- SIMULATED ORDER OPEN ---
                trade_amount = 100
                leverage = 10
                
                if action == "LONG":
                    tp_pct = 2.0
                    sl_pct = 1.0
                else:
                    tp_pct = 2.0
                    sl_pct = 1.0

                report_entry = (
                    f"\n{'='*60}\n"
                    f"[TRADE] SIMULATED ENTRY | {msg_dt}\n"
                    f"{'-'*60}\n"
                    f"NEWS: {msg_text.strip()}\n"
                    f"TARGET: {pair.upper()} ({coin_full_name})\n"
                    f"MCAP: {cap_str}\n"
                    f"TECH: RSI={rsi:.2f} ({rsi_label}) | MOM={momentum:.2f} ({mom_label})\n"
                    f"AI DECISION (NEXUS v2):\n"
                    f"   - Action: {action}\n"
                    f"   - Confidence: {confidence:.2f}%\n"
                    f"   - Raw Probs: {analysis['probs']}\n"
                    f"{'-'*60}\n"
                )
                
                # Open trade
                open_log, _ = ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=tp_pct, sl_pct=sl_pct,
                    amount_usdt=100, leverage=leverage, validity=30,
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                
                print(f"[INFO] Position opened: {pair} | {action}")

                # --- 4. POSITION MONITORING ---
                for k in klines:
                    minute_ts = k[0] / 1000
                    ticks = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                    
                    for i, tick_price in enumerate(ticks):
                        current_ts = minute_ts + (i * 15)
                        res_log, _, sym, pnl, peak, _ = ctx.exchange.check_positions_test(
                            pair, tick_price, now_ts=current_ts
                        )
                        
                        if res_log:
                            close_dt = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S")
                            duration_min = (current_ts - msg_ts) / 60
                            report_exit = (
                                f"[RESULT] TRADE CLOSED ({close_dt}):\n"
                                f"   - Status: {res_log}\n"
                                f"   - Entry: {entry_price} | Exit: {tick_price}\n"
                                f"   - Leverage: {leverage}x\n"
                                f"   - Duration: {duration_min:.1f} min\n"
                                f"   - PnL: {pnl:.2f} USDT\n"
                                f"   - Peak Price: {peak}\n"
                                f"{'='*60}\n"
                            )
                            
                            f_log.write(report_entry + report_exit)
                            f_log.flush()
                            print(f"[INFO] Trade closed: {pair} | PnL: {pnl:.2f}")
                            return

            # D) Missed Opportunity (HOLD or Low Confidence)
            elif action == "HOLD":
                future_candles = klines[1:21]
                if future_candles:
                    max_price = max([float(k[2]) for k in future_candles])
                    min_price = min([float(k[3]) for k in future_candles])
                    
                    pct_change_up = ((max_price - entry_price) / entry_price) * 100
                    pct_change_down = ((min_price - entry_price) / entry_price) * 100
                    
                    missed_action = None
                    change_val = 0.0
                    
                    if pct_change_up >= 1.5:
                        missed_action = "LONG"
                        change_val = pct_change_up
                    elif pct_change_down <= -1.5:
                        missed_action = "SHORT"
                        change_val = pct_change_down 
                        
                    if missed_action:
                        lev_10_profit = abs(change_val) * 10
                        
                        missed_log = (
                            f"\n{'='*60}\n"
                            f"[MISSED] OPPORTUNITY (HOLD) | {msg_dt}\n"
                            f"{'-'*60}\n"
                            f"NEWS: {msg_text.strip()}\n"
                            f"TARGET: {pair.upper()} ({coin_full_name})\n"
                            f"AI DECISION: {action} (Confidence: {confidence:.2f}%)\n"
                            f"OUTCOME (20m): {change_val:.2f}% ({missed_action})\n"
                            f"MISSED PnL (10x): {lev_10_profit:.2f}%\n"
                            f"TECH: RSI={rsi:.2f} | MOM={momentum:.2f}\n"
                            f"{'='*60}\n"
                        )
                        f_log.write(missed_log)
                        f_log.flush()
                        print(f"[INFO] Missed opportunity logged for {pair}: {change_val:.2f}%")

        except Exception as e:
            print(f"[ERROR] Simulation error for {pair}: {e}")

async def process_offline_entry(entry, ctx, f_log):
    """
    Processes a single entry from the offline dataset.
    """
    try:
        msg_text = entry["msg_text"]
        msg_dt = entry["msg_dt"]
        pair = entry["pair"]
        coin_full_name = entry["coin_full_name"]
        cap_str = entry["cap_str"]
        technicals = entry["technicals"]
        outcomes = entry["outcomes"]
        msg_ts = entry["msg_ts"]
        
        # Identify params from technicals
        rsi = technicals["rsi"]
        momentum = technicals["momentum_1h"]
        rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        mom_label = "BULLISH_MOM" if momentum > 0.5 else "BEARISH_MOM" if momentum < -0.5 else "FLAT"

        # AI Prediction
        msg_text_clean = clean_news_text(msg_text)
        
        if LOCAL_BRAIN_MODE:
            analysis = ctx.local_brain.predict(news_text=msg_text_clean, symbol=pair.replace("USDT", ""))
            action = analysis["decision"]
            confidence = analysis["confidence"]
        else:
            analysis = await ctx.brain.analyze_specific_no_research(
                news=msg_text_clean,
                symbol=pair.replace("USDT", ""),
            )
            action = analysis.get("action", "HOLD")
            confidence = analysis.get("conviction_score", 0)

        print(f"[AI] NEXUS decision for {pair}: {action} (Confidence: {confidence:.2f}%)")

        # Decision Logic
        if confidence >= 75 and action in ["LONG", "SHORT"]:
            entry_price = technicals["close"]
            leverage = 10
            
            report_entry = (
                f"\n{'='*60}\n"
                f"[TRADE] SIMULATED ENTRY | {msg_dt}\n"
                f"{'-'*60}\n"
                f"NEWS: {msg_text_clean}\n"
                f"TARGET: {pair} ({coin_full_name})\n"
                f"MCAP: {cap_str}\n"
                f"TECH: RSI={rsi:.2f} ({rsi_label}) | MOM={momentum:.2f} ({mom_label})\n"
                f"AI DECISION (NEXUS v2):\n"
                f"   - Action: {action}\n"
                f"   - Confidence: {confidence:.2f}%\n"
                f"{'-'*60}\n"
            )

            future_candles = outcomes.get("future_candles", [])
            
            # Simulation Loop
            res_log = None
            pnl = 0.0
            peak = 0.0
            
            # First open
            if LOCAL_BRAIN_MODE:
                ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=2.0, sl_pct=1.0,
                    amount_usdt=100, leverage=leverage, validity=30,
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                print(f"[INFO] Position opened: {pair} | {action}")
            else:
                ctx.exchange.open_position_test(
                    symbol=pair, side=action, price=entry_price,
                    tp_pct=abs(analysis['tp_pct']), sl_pct=0.8,
                    amount_usdt=100, leverage=leverage, validity=analysis['validity_minutes'],
                    app_state=ctx.app_state, decision_id=999, now_ts=msg_ts
                )
                print(f"[INFO] Position opened: {pair} | {action}")
            
            for k in future_candles:
                 minute_ts = k['ts']
                 ticks = [k['o'], k['h'], k['l'], k['c']]
                 
                 for i, tick_price in enumerate(ticks):
                    current_ts = minute_ts + (i * 15)
                    r_log, _, _, val_pnl, val_peak, _ = ctx.exchange.check_positions_test(
                        pair, tick_price, now_ts=current_ts
                    )
                    
                    if r_log:
                        res_log = r_log
                        pnl = val_pnl
                        peak = val_peak
                        close_dt = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S")
                        duration_min = (current_ts - msg_ts) / 60
                        
                        report_exit = (
                            f"[RESULT] TRADE CLOSED ({close_dt}):\n"
                            f"   - Status: {res_log}\n"
                            f"   - Entry: {entry_price} | Exit: {tick_price}\n"
                            f"   - Leverage: {leverage}x\n"
                            f"   - Duration: {duration_min:.1f} min\n"
                            f"   - PnL: {pnl:.2f} USDT\n"
                            f"   - Peak Price: {peak}\n"
                            f"{'='*60}\n"
                        )
                        f_log.write(report_entry + report_exit)
                        f_log.flush()
                        print(f"[INFO] Trade closed: {pair} | PnL: {pnl:.2f}")
                        return

        # Missed Opportunity Check
        elif action == "HOLD":
            entry_price = technicals["close"]
            max_gain_20m = outcomes.get("max_gain_20m", 0.0)
            max_loss_20m = outcomes.get("max_loss_20m", 0.0)
            
            missed_action = None
            change_val = 0.0
            
            if max_gain_20m >= 1.5:
                missed_action = "LONG"
                change_val = max_gain_20m
            elif max_loss_20m <= -1.5:
                missed_action = "SHORT"
                change_val = max_loss_20m 
            
            if missed_action:
                lev_10_profit = abs(change_val) * 10
                missed_log = (
                    f"\n{'='*60}\n"
                    f"[MISSED] OPPORTUNITY (HOLD) | {msg_dt}\n"
                    f"{'-'*60}\n"
                    f"NEWS: {msg_text.strip()}\n"
                    f"TARGET: {pair} ({coin_full_name})\n"
                    f"AI DECISION: {action} (Confidence: {confidence:.2f}%)\n"
                    f"OUTCOME (20m): {change_val:.2f}% ({missed_action})\n"
                    f"MISSED PnL (10x): {lev_10_profit:.2f}%\n"
                    f"TECH: RSI={rsi:.2f} | MOM={momentum:.2f}\n"
                    f"{'='*60}\n"
                )
                f_log.write(missed_log)
                f_log.flush()
                print(f"[INFO] Missed opportunity logged for {pair}: {change_val:.2f}%")

    except Exception as e:
        print(f"[ERROR] Offline error: {e}")

async def run_simulation():
    print("[SYSTEM] Starting NEXUS backtest simulation...")
    
    # Context Preparation
    ctx = BotContext()
    ctx.app_state = SharedState()
    ctx.memory = MockMemory()
    ctx.exchange = PaperExchange(1000.0)
    
    # Load Local Model
    ctx.local_brain = local_brain
    
    # Initialize brain (retained for detect_symbol fallback)
    ctx.brain = AgentBrain(
        use_groqcloud=False,
        api_key=GROQCLOUD_API_KEY,
        groqcloud_model=GROQCLOUD_MODEL,
        use_gemini=False,
        google_api_key=GOOGLE_API_KEY,
        gemini_model=GEMINI_MODEL
    )
    
    # [OFFLINE MODE CHECK]
    offline_file = str(DATA_DIR / "offline_test_data.json")
    results_file = str(DATA_DIR / "backtest_results_nexus_phi.txt")
    if not os.path.exists(results_file):
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(offline_file):
        print(f"[INFO] Offline data file found: {offline_file}")
        print("[SYSTEM] Entering OFFLINE simulation mode...")
        
        with open(offline_file, "r", encoding="utf-8") as f_in:
             offline_data = json.load(f_in)
        
        with open(results_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n--- OFFLINE SIMULATION RUN: {datetime.now()} ---\n")
            
            print(f"[INFO] Processing {len(offline_data)} records...")
            for i, entry in enumerate(offline_data):
                if i % 10 == 0: print(f"Progress: {i}/{len(offline_data)}")
                await process_offline_entry(entry, ctx, f_out)
                
        print(f"[SUCCESS] Offline simulation complete. Results: {results_file}")
        return

    # [ONLINE MODE FALLBACK] - REMOVED
    print("[ERROR] Offline data not found. Please run test_dataset.py first.")
    return

if __name__ == "__main__":
    # Run the simulation for the SetFit model
    asyncio.run(run_simulation())