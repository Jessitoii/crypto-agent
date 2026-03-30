import asyncio
from collections import defaultdict, deque
import time
import os
from nicegui import ui, app
from telethon import TelegramClient
import threading


# Modules
from config import (
    USE_GROQCLOUD,
    GROQCLOUD_API_KEY,
    GROQCLOUD_MODEL,
    USE_MAINNET,
    REAL_TRADING_ENABLED,
    API_KEY,
    API_SECRET,
    IS_TESTNET,
    TARGET_CHANNELS,
    RSS_FEEDS,
    API_ID,
    API_HASH,
    TELETHON_SESSION_NAME,
    STARTING_BALANCE,
    LEVERAGE,
    FIXED_TRADE_AMOUNT,
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

# Path Configuration
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace("src", "data")
os.chdir(dir)

SESSION_PATH = os.path.join(dir, "crypto_agent_session")

class BotContext:
    def __init__(self):
        self.is_running = True
        self.log_container = None
        self.runtime_logs = deque(maxlen=200)
        self.ai_decisions = deque(maxlen=100)

class SharedState:
    def __init__(self):
        self.is_running = True

if __name__ == "__main__":
    # --- GLOBAL STATE CONTAINER ---
    ctx = BotContext()

    # --- INITIALIZATION ---
    ctx.app_state = SharedState()
    ctx.market_memory = defaultdict(PriceBuffer)
    ctx.exchange = PaperExchange(STARTING_BALANCE)
    ctx.brain = AgentBrain(
        use_groqcloud=USE_GROQCLOUD,
        api_key=GROQCLOUD_API_KEY,
        groqcloud_model=GROQCLOUD_MODEL,
    )
    ctx.real_exchange = BinanceExecutionEngine(API_KEY, API_SECRET, testnet=IS_TESTNET)
    ctx.collector = TrainingDataCollector()
    ctx.dataset_manager = DatasetManager()
    ctx.telegram_client = TelegramClient(
        SESSION_PATH, API_ID, API_HASH, use_ipv6=False, timeout=10
    )
    ctx.stream_command_queue = None
    ctx.memory = MemoryManager()


    # Technical Logger Wrapper
    def log_ui_wrapper(message, type="info"):
        timestamp = time.strftime("%H:%M:%S")
        
        log_label = f"[{type.upper()}]"
        if type == "info":
            log_label = "[LOG]"

        full_msg = f"[{timestamp}] {log_label} {message}"
        print(full_msg)

        # 1. Store in memory for UI persistence
        ctx.runtime_logs.append(full_msg)

        # 2. Push to UI if container is initialized
        try:
            if ctx.log_container is not None:
                ctx.log_container.push(full_msg)
        except Exception:
            pass

    ctx.log_ui = log_ui_wrapper

    # --- STARTUP TASKS ---
    async def start_tasks():
        ctx.memory.load_recent_history(ctx)
        ctx.stream_command_queue = asyncio.Queue()
        
        if REAL_TRADING_ENABLED:
            await ctx.real_exchange.connect()

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
            await ctx.real_exchange.connect()
            ctx.log_ui("Real Trading Disabled (Paper Trading Mode)", "warning")

        # Launch Background Loops
        asyncio.create_task(services.websocket_loop(ctx))
        asyncio.create_task(services.collector_loop(ctx))
        asyncio.create_task(services.telegram_loop(ctx))
        asyncio.create_task(services.position_monitor_loop(ctx))


    # --- UI ENTRY POINT ---
    @ui.page("/")
    def index():
        async def manual_news_handler(text, source="MANUAL"):
            await services.process_news(text, source, ctx)

        ctx.log_container = create_dashboard(
            ctx=ctx,
            on_manual_submit=manual_news_handler,
            existing_logs=ctx.runtime_logs,
        )


    app.on_startup(start_tasks)
    ui.run(title="Crypto AI", host="0.0.0.0", dark=True, port=8080, reload=False)