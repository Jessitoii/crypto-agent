"""
Background Services and Business Logic

This module implements the core business logic of the Crypto-Agent.
It manages asynchronous service loops for:
1. Real-time news ingestion and processing (Telegram, RSS).
2. Market data synchronization via WebSockets.
3. Automated trade execution and risk management.
4. Data collection and outcome verification for model refinement.
"""

import asyncio
import time
import json
import re
import datetime
import os
import websockets
from telethon import events

from rss_listener import RSSMonitor
from utils import get_top_100_map, perform_research, find_coins
from config import (
    TARGET_CHANNELS,
    RSS_FEEDS,
    WEBSOCKET_URL,
    REAL_TRADING_ENABLED,
    IGNORE_KEYWORDS,
    FIXED_TRADE_AMOUNT,
    LEVERAGE,
)
from price_buffer import PriceBuffer

# Static mapping for top market assets to optimize lookup performance
TARGET_PAIRS = get_top_100_map()


def log_txt(message, filename="trade_logs.txt"):
    """
    Logs trade-related events to a persistent text file for auditing.
    
    Args:
        message (str): Log message content.
        filename (str): Name of the audit log file.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n### {datetime.datetime.now()} ###\n{message}\n##################\n")


async def update_system_balance(ctx, last_pnl=0.0):
    """
    Synchronizes the local balance state with the execution engine.
    
    In live mode, it fetches the actual wallet balance from Binance.
    In paper mode, it applies simulated PnL to the virtual balance.
    
    Args:
        ctx (BotContext): Main application context.
        last_pnl (float): Realized profit or loss to apply (Paper mode).
    """
    if REAL_TRADING_ENABLED:
        # Brief sleep to allow exchange-side balance settlement
        await asyncio.sleep(1)
        total, available = await ctx.real_exchange.get_usdt_balance()
        if total > 0:
            old_balance = ctx.exchange.balance
            ctx.exchange.balance = total
            diff = total - old_balance
            ctx.log_ui(
                f"Balance Updated: {total:.2f} USDT (Diff: {diff:+.2f})",
                "info",
            )
    else:
        ctx.exchange.balance += last_pnl
        ctx.log_ui(
            f"Simulation Balance: {ctx.exchange.balance:.2f} USDT (PnL: {last_pnl:+.2f})",
            "info",
        )


async def send_telegram_alert(ctx, message):
    """
    Dispatches critical alerts to the user's private Telegram chat.
    
    Args:
        ctx (BotContext): Application context.
        message (str): The alert message content.
    """
    try:
        if not ctx.telegram_client.is_connected():
            print("[TELEGRAM] Warning: No connection, attempting to connect...")
            await ctx.telegram_client.connect()

        if not await ctx.telegram_client.is_user_authorized():
            ctx.log_ui(
                "[TELEGRAM] Warning: Unauthorized session.", "error"
            )
            print("[TELEGRAM] Warning: Unauthorized session.")
            return

        await ctx.telegram_client.send_message("me", f"**BOT ALERT**\n{message}")
        print("[TELEGRAM] Info: Message sent.")

    except Exception as e:
        print(f"[ERROR] Telegram send error: {e}")
        ctx.log_ui(f"Telegram Send Error: {e}", "error")


async def ensure_fresh_data(ctx, pair):
    """
    Validates data freshness and triggers backfills if latency is detected.
    
    Args:
        ctx (BotContext): Application context.
        pair (str): Target asset pair.
        
    Returns:
        bool: True if data is synchronized and ready for analysis.
    """
    stats = ctx.market_memory[pair]
    is_stale = False
    current_minute = int(time.time() / 60)

    # Heuristic: Data is stale if last candle is > 3 minutes old or missing
    if stats.current_price == 0:
        is_stale = True
    elif stats.candles:
        last_candle_time = stats.candles[-1][0]
        if (current_minute - last_candle_time) > 3:
            is_stale = True
    else:
        is_stale = True

    if is_stale:
        ctx.log_ui(f"{pair} data stale or missing. Fetching fresh data...", "warning")
        hist_data, chg_24h = await ctx.real_exchange.fetch_missing_data(pair)

        if hist_data:
            stats.candles.clear()
            for c, t in hist_data:
                stats.update_candle(c, t, True)
            stats.set_24h_change(chg_24h)
            stats.current_price = hist_data[-1][0]
            return True
        else:
            return False

    return True


async def execute_trade_logic(ctx, pair, dec, stats, source, msg, changes, search_res):
    """
    Orchestrates the transition from AI decision to order execution.
    
    Implements tiered risk allocation based on model confidence and 
    manages dual execution (Real + Virtual Simulation).
    
    Args:
        ctx (BotContext): Application context.
        pair (str): Target pair.
        dec (dict): AI recommendation object.
        stats (PriceBuffer): Market data buffer.
        source (str): News source label.
        msg (str): Original news text.
        changes (dict): Price delta metrics.
        search_res (str): Supplement research context.
    """
    confidence = dec.get("confidence", 0)
    balance = ctx.exchange.balance

    # --- TIERED RISK ALLOCATION STRATEGY ---
    # Risk Tier 1: Conservative (Conf 65-74%)
    trade_amount = balance * 0.40  
    leverage = 10  
    
    # Risk Tier 2: Standard (Conf 75-89%)
    if confidence >= 75:
        trade_amount = balance * 0.50  
        leverage = 15  
    
    # Risk Tier 3: Aggressive / Nuclear Mode (Conf 90%+)
    if confidence >= 90:
        trade_amount = balance * 0.60  
        leverage = 20  

        ctx.log_ui(
            f"NUCLEAR MODE ACTIVE: {pair} | 20x | 60% Balance Allocation",
            "warning",
        )

    tp_pct = dec.get("tp_pct", 2.0)
    sl_pct = dec.get("sl_pct", 1.0)
    validity = dec.get("validity_minutes", 15)

    can_open_paper_trade = False

    # --- 1. Real-World Execution Pipeline ---
    if REAL_TRADING_ENABLED:
        api_result = await ctx.real_exchange.execute_trade(
            pair, dec["action"], trade_amount, leverage, tp_pct, sl_pct
        )
        if api_result == "Execution failed":
            ctx.log_ui(
                f"Binance rejected trade: {pair.upper()}. Simulation cancelled.",
                "error",
            )
            can_open_paper_trade = False
        elif api_result == "No connection":
            ctx.log_ui("API not connected. Falling back to Paper Trading.", "warning")
            can_open_paper_trade = True
        else:
            can_open_paper_trade = True
    else:
        can_open_paper_trade = True

    # --- 2. Simulation & Performance Tracking ---
    if can_open_paper_trade:
        log, color = ctx.exchange.open_position(
            symbol=pair,
            side=dec["action"],
            price=stats.current_price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            amount_usdt=trade_amount,
            leverage=leverage,
            validity=validity,
            app_state=ctx.app_state,
            decision_id=dec.get("db_id"),
        )

        full_log = log + f'\nSrc: {source}\nReason: {dec.get("reason")}\nNews: {msg}'
        ctx.log_ui(full_log, color)
        log_txt(full_log)

        # Persistence for offline training and evaluation
        ctx.dataset_manager.log_trade_entry(
            symbol=pair,
            news=msg,
            price_data=str(changes),
            ai_decision=dec,
            search_context=search_res,
            entry_price=stats.current_price,
        )
        asyncio.create_task(send_telegram_alert(ctx, full_log))

        # Dynamic WebSocket subscription to track the specific asset high-frequency
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{pair.lower()}@kline_1m"],
            "id": int(time.time()),
        }
        await ctx.stream_command_queue.put(subscribe_msg)


async def process_news(msg, source, ctx):
    """
    Universal entry point for news processing.
    
    Handles deduplication, symbol extraction, market research, 
    AI inference, and order book safety verification.
    
    Args:
        msg (str): Raw news content.
        source (str): Source identifier (e.g., 'TELEGRAM', 'RSS').
        ctx (BotContext): Application context.
    """
    start_time = time.time()
    if not ctx.app_state.is_running:
        return

    # --- 1. Filtering & Deduplication ---
    # Prevents processing the same event multiple times across different channels
    is_dup, score = ctx.memory.is_duplicate(msg)
    if is_dup:
        ctx.log_ui(f"[DUPLICATE] News filtered (Similarity: {score:.2f})", "warning")
        return

    ctx.memory.add_news(source, msg)
    clean_msg = msg.replace("— link", "").replace("Link:", "")
    msg_lower = clean_msg.lower()

    log_txt(f"[{source}] Incoming News: {clean_msg}")

    # Heuristic filter for summary/digest news which lack immediate alpha
    for word in IGNORE_KEYWORDS:
        if word in msg_lower:
            ctx.log_ui(f"[FILTER] Stale keywords detected: '{word}'", "warning")
            return

    ctx.log_ui(f"[{source}] Processing: {msg[:40]}...", "info")

    # --- 2. Ticker Detection ---
    # Uses hierarchical detection: Regex (Performance) -> LLM (Accuracy)
    detected_pairs = find_coins(msg, coin_map=TARGET_PAIRS)

    if not detected_pairs:
        ctx.log_ui("Regex failed, consulting agent...", "warning")
        found_symbol = await ctx.brain.detect_symbol(msg, TARGET_PAIRS)
        if found_symbol:
            pot_pair = f"{found_symbol.lower()}usdt"
            if pot_pair in TARGET_PAIRS:
                ctx.log_ui(f"AGENT FOUND: {found_symbol}", "success")
                detected_pairs.append(pot_pair)

    # --- 3. Asset Analysis Loop ---
    coin_map = get_top_100_map()

    for pair in detected_pairs:
        data_ready = await ensure_fresh_data(ctx, pair)
        if not data_ready:
            ctx.log_ui(f"Failed to fetch {pair} data, analysis aborted.", "error")
            continue

        stats = ctx.market_memory[pair]

        # Dynamic Web Research to provide LLM with the latest sectoral context
        smart_query = await ctx.brain.generate_search_query(
            msg, pair.replace("usdt", "")
        )
        ctx.log_ui(f"Researching: '{smart_query}'", "info")
        search_res = await perform_research(smart_query)

        clean_symbol = pair.replace("usdt", "").lower()
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

        rsi_val = stats.calculate_rsi()
        changes = stats.get_all_changes()

        # BTC Trend Correlation Sync
        btc_pair = "btcusdt"
        btc_stats = ctx.market_memory.get(btc_pair)

        btc_is_stale = False
        if not btc_stats or not btc_stats.candles:
            btc_is_stale = True
        elif (int(time.time() / 60) - btc_stats.candles[-1][0]) > 5:
            btc_is_stale = True

        if btc_is_stale:
            btc_hist, btc_24h = await ctx.real_exchange.fetch_missing_data(btc_pair)
            if btc_hist:
                if btc_pair not in ctx.market_memory:
                    ctx.market_memory[btc_pair] = PriceBuffer()

                ctx.market_memory[btc_pair].candles.clear()
                for c, t in btc_hist:
                    ctx.market_memory[btc_pair].update_candle(c, t, True)
                ctx.market_memory[btc_pair].current_price = btc_hist[-1][0]
                btc_stats = ctx.market_memory[btc_pair]

        btc_trend = btc_stats.get_change(60) if btc_stats else 0.0

        ctx.log_ui(f"Analysis Price ({pair}): {stats.current_price}", "info")

        # --- AI INFERENCE ENGINE ---
        volume_24h, funding_rate = await ctx.real_exchange.get_extended_metrics(pair)
        
        dec = await ctx.brain.analyze_specific(
            msg,
            pair,
            stats.current_price,
            changes,
            search_res,
            coin_full_name,
            cap_str,
            rsi_val,
            btc_trend,
            volume_24h,
            funding_rate,
        )

        # Log decision for downstream data mining
        ctx.collector.log_decision(msg, pair, stats.current_price, str(changes), dec)

        # Populate Dashboard with recent decision state
        decision_record = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "symbol": pair.upper().replace("USDT", ""),
            "action": dec.get("action", "HOLD"),
            "confidence": dec.get("confidence", 0),
            "reason": dec.get("reason", "N/A"),
            "price": stats.current_price,
            "news_snippet": msg[:60] + "...",
            "validity": dec.get("validity_minutes", 0),
            "tp_pct": dec.get("tp_pct", 0.0),
            "sl_pct": dec.get("sl_pct", 0.0),
        }
        ctx.ai_decisions.append(decision_record)
        decision_id = ctx.memory.log_decision(decision_record)  
        dec["db_id"] = decision_id

        # --- 4. Order Book Integrity & Spread Verification ---
        if dec["action"] in ["LONG", "SHORT"] and REAL_TRADING_ENABLED:
            imbalance, depth_info = await ctx.real_exchange.get_order_book_imbalance(
                pair
            )
            ctx.log_ui(
                f"Depth Analysis ({pair}): Imbalance {imbalance:.2f} | {depth_info}",
                "info",
            )

            # Cancel Longs if there is extreme sell pressure in the book
            if dec["action"] == "LONG" and imbalance < -0.5:
                ctx.log_ui(
                    f"Order Wall Detected: High Sell Pressure ({imbalance:.2f}). LONG Cancelled.",
                    "warning",
                )
                dec["action"] = "HOLD"  
                dec["reason"] += " [CANCELLED: Sell Wall]"

            # Cancel Shorts if there is extreme buy pressure in the book
            elif dec["action"] == "SHORT" and imbalance > 0.5:
                ctx.log_ui(
                    f"Order Wall Detected: High Buy Pressure ({imbalance:.2f}). SHORT Cancelled.",
                    "warning",
                )
                dec["action"] = "HOLD"  
                dec["reason"] += " [CANCELLED: Buy Wall]"

            try:
                ticker = await ctx.real_exchange.client.futures_orderbook_ticker(
                    symbol=pair.upper()
                )
                bid = float(ticker["bidPrice"])
                ask = float(ticker["askPrice"])

                spread_pct = ((ask - bid) / ask) * 100
                ctx.log_ui(f"Spread Analysis ({pair}): {spread_pct:.3f}%", "info")

                # Block execution if the spread is too high (avoid slippage)
                if spread_pct > 0.3:  
                    ctx.log_ui(
                        f"High Spread Warning ({spread_pct:.2f}%). Execution skipped.",
                        "warning",
                    )
                    dec["action"] = "HOLD"  
                    dec["reason"] += f" [CANCELLED: High Spread {spread_pct:.2f}%]"

            except Exception as e:
                ctx.log_ui(f"Could not retrieve spread data: {e}", "warning")

        # Threshold check: Only execute if AI confidence meets the criteria
        if dec["confidence"] >= 65 and dec["action"] in ["LONG", "SHORT"]:
            await execute_trade_logic(
                ctx, pair, dec, stats, source, msg, changes, search_res
            )
        else:
            log = f"Pass: {pair.upper()} ({coin_full_name}) | {dec['action']} | (Conf: {dec['confidence']}%) | Reason: {dec.get('reason')}\nNews: {msg}"
            ctx.log_ui(log, "warning")
            log_txt(log)
            asyncio.create_task(send_telegram_alert(ctx, log))

    end_time = time.time()
    ctx.log_ui(
        f"[{source}] Processing Cycle Time: {end_time - start_time:.2f} s.", "info"
    )


async def websocket_loop(ctx):
    """
    Asynchronous WebSocket supervisor for real-time market data.
    
    Handles automatic reconnections and delegates message processing 
    to specific handlers.
    """
    ctx.log_ui("Connecting Websocket (Sniper Mode)...", "info")

    while ctx.app_state.is_running:
        try:
            async with websockets.connect(WEBSOCKET_URL) as ws:
                ctx.log_ui("Websocket Connected.", "success")

                async def sender():
                    """Sends subscription/unsubscription commands to Binance."""
                    while ctx.app_state.is_running:
                        try:
                            command = await ctx.stream_command_queue.get()
                            await ws.send(json.dumps(command))
                        except Exception as e:
                            ctx.log_ui(f"WS Sender Error: {e}", "error")
                            break  

                async def receiver():
                    """Handles incoming price updates and triggers position checks."""
                    async for msg in ws:
                        try:
                            raw_data = json.loads(msg)
                            data = raw_data.get("data", raw_data)

                            if isinstance(data, dict) and data.get("e") == "kline":
                                pair = data["s"].lower()
                                k = data["k"]
                                price = float(k["c"])
                                is_closed = k["x"]
                                ts = k["t"] / 1000

                                if pair not in ctx.market_memory:
                                    ctx.market_memory[pair] = PriceBuffer()

                                ctx.market_memory[pair].update_candle(
                                    price, ts, is_closed
                                )

                                # Active Position Monitor Synchronization
                                if pair in ctx.exchange.positions:
                                    log, color, closed_sym, pnl, peak_price, decision_id = (
                                        ctx.exchange.check_positions(pair, price)
                                    )

                                    if log:
                                        ctx.log_ui(log, color)
                                        log_txt(log)

                                        if closed_sym:
                                            await handle_closed_position(
                                                ctx, closed_sym, pnl, peak_price, log, decision_id
                                            )

                        except Exception as e:
                            ctx.log_ui(f"WS Msg Processing Error: {e}", "warning")
                            continue

                await asyncio.gather(sender(), receiver())

        except Exception as e:
            ctx.log_ui(
                f"Websocket Disconnected: {e}. Retrying in 5s...",
                "error",
            )
            await asyncio.sleep(5)


async def position_monitor_loop(ctx):
    """
    High-frequency watchdog for open positions.
    
    Secondary monitoring channel to ensure TP/SL/Expiry are caught 
    even if WebSocket updates for a specific ticker are delayed.
    """
    ctx.log_ui("Position Monitor Active.", "success")

    while ctx.app_state.is_running:
        try:
            await asyncio.sleep(2)  

            if not ctx.exchange.positions:
                continue

            open_symbols = list(ctx.exchange.positions.keys())

            for pair in open_symbols:
                if pair not in ctx.market_memory:
                    continue

                current_price = ctx.market_memory[pair].current_price
                if current_price == 0:
                    continue

                log, color, closed_sym, pnl, peak, decision_id = (
                    ctx.exchange.check_positions(pair, current_price)
                )

                if log:
                    ctx.log_ui(log, color)
                    log_txt(log)
                    if closed_sym:
                        await handle_closed_position(
                            ctx, closed_sym, pnl, peak, log, decision_id
                        )

        except Exception as e:
            print(f"[ERROR] Monitor Loop Warning: {e}")
            await asyncio.sleep(5)


async def handle_closed_position(ctx, symbol, pnl, peak_price, log_msg, decision_id=None): 
    """
    Post-trade cleanup and performance recording.
    
    Args:
        ctx (BotContext): Application context.
        symbol (str): Ticker that was closed.
        pnl (float): Realized profit/loss.
        peak_price (float): Maximum/Minimum price reached during the trade.
        log_msg (str): Summary log message.
        decision_id (int, optional): Reference to the original AI decision.
    """
    # Liquidation on real exchange if enabled
    if REAL_TRADING_ENABLED:
        asyncio.create_task(ctx.real_exchange.close_position_market(symbol))

    try:
        # Audit Trail Logging
        ctx.dataset_manager.log_trade_exit(symbol, pnl, "Closed", peak_price)
        asyncio.create_task(send_telegram_alert(ctx, log_msg))
        
        # SQLite Persistence
        if ctx.exchange.history:
            last_trade = ctx.exchange.history[-1]
            last_trade["peak_price"] = peak_price
            ctx.memory.log_trade(last_trade, decision_id)

        # Unsubscribe from high-frequency updates to save bandwidth/resources
        try:
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": [f"{symbol.lower()}@kline_1m"],
                "id": int(time.time()),
            }
            await ctx.stream_command_queue.put(unsubscribe_msg)
        except Exception:
            pass
        
        # Trigger balance synchronization
        asyncio.create_task(update_system_balance(ctx, last_pnl=pnl))

    except Exception as e:
        ctx.log_ui(f"CRITICAL ERROR in handle_closed_position: {e}", "error")


async def telegram_loop(ctx):
    """
    Bootstraps and maintains the Telegram event listener.
    """
    ctx.log_ui("Connecting Telegram Service...", "info")
    try:
        await ctx.telegram_client.start()

        print(f"TELEGRAM CONNECTED: {ctx.telegram_client.is_connected()}")
        print(f"TELEGRAM AUTHORIZED: {await ctx.telegram_client.is_user_authorized()}")
        await send_telegram_alert(ctx, "Telegram Ingestion Service Online")
        
        if not await ctx.telegram_client.is_user_authorized():
            ctx.log_ui("TELEGRAM SESSION AUTHORIZATION FAILED", "error")
            return

        ctx.log_ui("Telegram Listener Registered", "success")

        @ctx.telegram_client.on(events.NewMessage(chats=TARGET_CHANNELS))
        async def handler(event):
            """Handler for incoming messages from targeted Alpha channels."""
            if event.message.message:
                await process_news(event.message.message, "TELEGRAM", ctx)

    except Exception as e:
        ctx.log_ui(f"Telegram Service Error: {e}", "error")


async def collector_loop(ctx):
    """
    Periodically validates the accuracy of historical trade outcomes.
    """
    ctx.log_ui("Data Verification Service Active", "success")
    while True:
        await asyncio.sleep(60)
        curr_prices = {
            p: ctx.market_memory[p].current_price
            for p in TARGET_PAIRS
            if ctx.market_memory[p].current_price > 0
        }
        if curr_prices:
            await ctx.collector.check_outcomes(curr_prices)


async def rss_loop(ctx):
    """
    Initializes the RSS Monitoring subsystem.
    """
    ctx.log_ui("Initializing RSS Ingestion...", "info")
    rss_bot = RSSMonitor(
        callback_func=lambda msg, src: asyncio.create_task(process_news(msg, src, ctx))
    )
    await rss_bot.start_loop()

