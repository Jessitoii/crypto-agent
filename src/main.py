"""
Crypto-Agent Entry Point

This module serves as the primary execution engine for the Crypto-Agent application.
It initializes the UI (NiceGUI), sets up the trading environment (Paper/Binance),
and manages background services including news collection, market monitoring,
and AI-driven decision making.

Attributes:
    SESSION_PATH (str): Path to the Telegram session file.
"""

import asyncio
from collections import defaultdict, deque
import time
import os
from nicegui import ui, app
from telethon import TelegramClient
import threading


# --- Modules ---
from config import (
    FIXED_TRADE_AMOUNT,
    DATA_DIR,
    USE_GROQCLOUD,
    GROQCLOUD_API_KEY,
    GROQCLOUD_MODEL,
    STARTING_BALANCE,
    API_KEY,
    API_SECRET,
    IS_TESTNET,
    API_ID,
    API_HASH,
    REAL_TRADING_ENABLED
)
from exchange import PaperExchange
from brain import AgentBrain
from price_buffer import PriceBuffer
from binance_client import BinanceExecutionEngine
from data_collector import TrainingDataCollector
from dataset_manager import DatasetManager
from database import MemoryManager
from dashboard import create_dashboard
import services

# Runtime Configuration
SESSION_PATH = str(DATA_DIR / "crypto_agent_session")

class BotContext:
    """
    Centralized state container for the trading bot.
    
    Maintains runtime logs, AI decisions, and UI component references
    to ensure thread-safe access across asynchronous tasks.
    
    Attributes:
        is_running (bool): Global execution flag.
        log_container (ui.element): NiceGUI container for real-time logging.
        runtime_logs (deque): Circular buffer for system logs.
        ai_decisions (deque): Circular buffer for model inference results.
    """
    def __init__(self):
        self.is_running = True
        self.log_container = None
        self.runtime_logs = deque(maxlen=200)
        self.ai_decisions = deque(maxlen=100)

class SharedState:
    """
    Minimalistic shared state for cross-module coordination.
    
    Attributes:
        is_running (bool): Status of the main application loop.
    """
    def __init__(self):
        self.is_running = True

if __name__ == "__main__":
    # --- GLOBAL STATE CONTAINER ---
    # Initializing the singleton context to manage bot state
    ctx = BotContext()

    # --- INITIALIZATION ---
    ctx.app_state = SharedState()
    ctx.market_memory = defaultdict(PriceBuffer)
    ctx.exchange = PaperExchange(STARTING_BALANCE)
    
    # Initialize the AI Brain with configured LLM parameters
    ctx.brain = AgentBrain(
        use_groqcloud=USE_GROQCLOUD,
        api_key=GROQCLOUD_API_KEY,
        groqcloud_model=GROQCLOUD_MODEL,
    )
    
    # Setup execution engines for both simulated and real trading
    ctx.real_exchange = BinanceExecutionEngine(API_KEY, API_SECRET, testnet=IS_TESTNET)
    ctx.collector = TrainingDataCollector()
    ctx.dataset_manager = DatasetManager(str(DATA_DIR / "training_dataset.jsonl"))
    
    # Initialize Telegram client for real-time news ingestion
    ctx.telegram_client = TelegramClient(
        SESSION_PATH, API_ID, API_HASH, use_ipv6=False, timeout=10
    )
    ctx.stream_command_queue = None
    
    # Local SQLite database for persistent history and agent memory
    ctx.memory = MemoryManager(str(DATA_DIR / "nexus_db.sqlite"))


    def log_ui_wrapper(message, type="info"):
        """
        Technical logger wrapper for dual-channel output (Console + UI).
        
        Args:
            message (str): The log message content.
            type (str, optional): Log level ('info', 'success', 'warning', 'error'). 
                Defaults to "info".
        """
        timestamp = time.strftime("%H:%M:%S")
        
        log_label = f"[{type.upper()}]"
        if type == "info":
            log_label = "[LOG]"

        full_msg = f"[{timestamp}] {log_label} {message}"
        print(full_msg)

        # 1. Store in memory for UI persistence across refreshes
        ctx.runtime_logs.append(full_msg)

        # 2. Push to UI if container is initialized (reactive update)
        try:
            if ctx.log_container is not None:
                ctx.log_container.push(full_msg)
        except Exception:
            # Silent fail if UI is not yet ready for updates
            pass

    # Attach the logger to the context for global access
    ctx.log_ui = log_ui_wrapper

    async def start_tasks():
        """
        Asynchronous startup routine for initializing background services.
        
        Synchronizes balance with Binance if live trading is enabled and
        spawns worker tasks for data collection and market monitoring.
        """
        # Bootstrap agent memory from persistent storage
        ctx.memory.load_recent_history(ctx)
        ctx.stream_command_queue = asyncio.Queue()
        
        if REAL_TRADING_ENABLED:
            await ctx.real_exchange.connect()

            # Synchronize virtual balance with real-world Binance wallet
            real_total, real_available = await ctx.real_exchange.get_usdt_balance()

            if real_total > 0:
                ctx.exchange.balance = real_total
                ctx.exchange.initial_balance = real_total

                ctx.log_ui(
                    f"Balance Synced: {real_total:.2f} USDT (Available: {real_available:.2f})",
                    "success",
                )
            else:
                ctx.log_ui(
                    "Real balance could not be fetched or is zero. Using default.", "warning"
                )
        else:
            # Connect even in paper mode to ensure market data connectivity
            await ctx.real_exchange.connect()
            ctx.log_ui("Real Trading Disabled (Paper Trading Mode)", "warning")

        # Launch Background Execution Loops
        asyncio.create_task(services.websocket_loop(ctx))
        asyncio.create_task(services.collector_loop(ctx))
        asyncio.create_task(services.telegram_loop(ctx))
        asyncio.create_task(services.position_monitor_loop(ctx))


    # --- UI ENTRY POINT ---
    @ui.page("/")
    def index():
        """
        Renders the main dashboard page using NiceGUI.
        """
        async def manual_news_handler(text, source="MANUAL"):
            """
            Bridge for processing manually entered news items via UI.
            """
            await services.process_news(text, source, ctx)

        # Initialize and render the dashboard layout
        ctx.log_container = create_dashboard(
            ctx=ctx,
            on_manual_submit=manual_news_handler,
            existing_logs=ctx.runtime_logs,
        )


    # Register startup hooks and launch the web server
    app.on_startup(start_tasks)
    ui.run(title="Crypto AI", host="0.0.0.0", dark=True, port=8080, reload=False)