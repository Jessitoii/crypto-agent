"""
NEXUS AI Terminal - High-Frequency Trading Dashboard

This module implements the primary User Interface using NiceGUI.
It provides real-time visualization of:
1. System KPIs and Portfolio Health.
2. Active Asynchronous Positions with live PnL and expiry timers.
3. AI Reasoning Logs for decision transparency.
4. Strategy performance reports and historical trade audit trails.
"""

from nicegui import ui
import asyncio
import time
import config
from services import update_system_balance

def create_kpi(label, icon="attach_money"):
    """
    Constructs a stylized KPI card for the dashboard.
    
    Args:
        label (str): Metric label (e.g., 'Total PnL').
        icon (str): Material icon name.
        
    Returns:
        ui.label: The reactive label object for value updates.
    """
    with ui.card().classes(
        "bg-gray-900 border-l-4 border-primary p-3 flex-row gap-3 items-center"
    ):
        ui.icon(icon, size="md").classes("text-gray-600")
        with ui.column().classes("gap-0"):
            ui.label(label).classes("text-xs text-gray-400 uppercase tracking-widest")
            lbl = ui.label("...").classes("text-xl font-mono font-bold")
            return lbl

def create_dashboard(ctx, on_manual_submit, existing_logs=None):
    """
    Initializes the entire UI layout and interaction logic.
    
    Args:
        ctx (BotContext): Central application state.
        on_manual_submit (callable): Callback for manual news analysis.
        existing_logs (list, optional): Historical logs to display on boot.
        
    Returns:
        ui.log: The main system log container.
    """
    # Professional Sectoral Color Palette
    ui.colors(
        primary="#00B4D8",
        secondary="#0077B6",
        accent="#90E0EF",
        positive="#21BA45",
        negative="#C10015",
        dark="#0B0F19",
    )

    # --- TOP NAVIGATION BAR ---
    with ui.header().classes(
        "bg-dark/90 backdrop-blur-md border-b border-gray-800 p-4 items-center gap-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("hub", size="32px").classes("text-primary animate-pulse")
            ui.label("NEXUS AI TERMINAL").classes(
                "text-h6 font-mono font-bold tracking-wider text-white"
            )

        ui.space()

        # Lifecycle Status Indicators
        with ui.row().classes("gap-2"):
            def toggle_bot():
                """Toggles the bot's global execution state."""
                ctx.app_state.is_running = not ctx.app_state.is_running
                status_badge.set_text(
                    "SYSTEM: ONLINE" if ctx.app_state.is_running else "SYSTEM: PAUSED"
                )
                status_badge.classes(
                    replace=f"text-xs font-bold px-2 py-1 rounded {'bg-positive/20 text-positive' if ctx.app_state.is_running else 'bg-negative/20 text-negative'}"
                )

            initial_state = "SYSTEM: ONLINE" if ctx.app_state.is_running else "SYSTEM: PAUSED"
            initial_class = "bg-positive/20 text-positive" if ctx.app_state.is_running else "bg-negative/20 text-negative"
            
            status_badge = ui.label(initial_state).classes(
                f"text-xs font-bold px-2 py-1 rounded {initial_class} cursor-pointer"
            )
            status_badge.on("click", toggle_bot)
            
            ui.label("API: CONNECTED").classes(
                "text-xs font-bold px-2 py-1 rounded bg-blue-500/20 text-blue-400"
            )

    # --- EMERGENCY OPERATIONS ---
    async def panic_close_all():
        """
        Executes an immediate liquidation of all active simulated and real positions.
        """
        open_symbols = list(ctx.exchange.positions.keys())
        if not open_symbols:
            ui.notify("No active exposure detected.", type="warning")
            return
            
        n = len(open_symbols)
        ctx.log_ui(f"🚨 EMERGENCY LIQUIDATION: Terminating {n} positions...", "warning")
        
        for symbol in open_symbols:
            try:
                pos = ctx.exchange.positions.get(symbol)
                if not pos: continue
                
                pnl = pos.get("pnl", 0.0)
                reason = "OPERATOR PANIC EXIT 🚨"
                
                # Execute closure in virtual exchange
                log_msg, color = ctx.exchange.close_position(symbol, reason, pnl)
                ctx.log_ui(log_msg, color)
                
                # Mirror closure on Binance if enabled
                if config.REAL_TRADING_ENABLED:
                    await ctx.real_exchange.close_position_market(symbol)
                
                # Clean up stream subscriptions
                unsubscribe_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": [f"{symbol.lower()}@kline_1m"],
                    "id": int(time.time()),
                }
                await ctx.stream_command_queue.put(unsubscribe_msg)
                
                # Update global equity curves
                asyncio.create_task(update_system_balance(ctx, last_pnl=pnl))
                
            except Exception as e:
                ctx.log_ui(f"⚠️ Liquidation Fault ({symbol}): {e}", "error")
                
        ui.notify(f"Total Liquidation ({n} assets) Complete.", type="positive", position="center")

    # --- TAB NAVIGATION SYSTEM ---
    with ui.tabs().classes("w-full text-gray-400") as tabs:
        dash_tab = ui.tab("COCKPIT", icon="dashboard")
        ai_tab = ui.tab("AI LOGS", icon="psychology")
        report_tab = ui.tab("STRATEGY REPORT", icon="assessment")
        market_tab = ui.tab("MARKET", icon="show_chart")
        history_tab = ui.tab("HISTORY", icon="history")

    with ui.tab_panels(tabs, value=dash_tab).classes("w-full bg-transparent p-0"):

        # --- SECTION: COCKPIT (Main Operations View) ---
        with ui.tab_panel(dash_tab).classes("p-4 gap-4"):
            # High-Level Metrics
            with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
                bal_label = create_kpi("Wallet Equity")
                pnl_label = create_kpi("Cumulative PnL", icon="trending_up")
                win_label = create_kpi("Win Rate (Total)", icon="pie_chart")
                pos_count_label = create_kpi("Open Exposure", icon="layers")

            # Main Operational Layout
            with ui.grid(columns=3).classes("w-full h-[70vh] gap-4"):
                # Active Trades Column
                with ui.column().classes(
                    "col-span-2 h-full bg-gray-900/50 rounded-lg border border-gray-800 p-4"
                ):
                    with ui.row().classes("w-full justify-between items-center mb-2"):
                        ui.label("⚡ LIVE RISK EXPOSURE").classes(
                            "text-sm font-bold text-primary"
                        )
                        ui.button(
                            "PANIC EXIT",
                            icon="close",
                            color="negative",
                            on_click=panic_close_all,
                        ).props("outline size=xs")
                    positions_container = ui.column().classes(
                        "w-full gap-2 overflow-y-auto pr-2"
                    )

                # Real-time System Logs
                with ui.column().classes(
                    "col-span-1 h-full bg-black rounded-lg border border-gray-800 p-0 flex flex-col"
                ):
                    ui.label(">_ SYSTEM RUNTIME LOGS").classes(
                        "text-xs font-mono text-gray-500 p-2 border-b border-gray-800 bg-gray-900"
                    )
                    log_container = ui.log(max_lines=300).classes(
                        "w-full h-full p-2 font-mono text-xs text-green-400 leading-tight bg-transparent"
                    )
                    if existing_logs:
                        for l in existing_logs:
                            log_container.push(l)

            # Manual Analysis Interface
            with ui.row().classes(
                "w-full mt-4 bg-gray-900 p-2 rounded-lg items-center gap-2 border border-gray-800"
            ):
                ui.icon("edit_note", size="24px").classes("text-blue-400 ml-2")
                news_input = (
                    ui.input(placeholder="Manual Alpha Analysis: 'SEC v. Binance lawsuit update...'")
                    .classes("w-full flex-1")
                    .props("dark dense borderless")
                )

                async def submit():
                    """Submits manual news for AI heuristic analysis."""
                    if news_input.value:
                        await on_manual_submit(news_input.value, "MANUAL")
                        news_input.value = ""

                ui.button(icon="send", on_click=submit).props("flat dense color=primary")

        # --- SECTION: AI REASONING (Transparency View) ---
        with ui.tab_panel(ai_tab).classes("p-4"):
            ui.label("🧠 AI COGNITION LOG (Decision Transparency)").classes(
                "text-lg font-bold mb-4 text-white"
            )
            # Table Header for better alignment
            with ui.row().classes("w-full grid grid-cols-12 text-[10px] font-bold text-gray-500 border-b border-gray-700 pb-2 mb-2 items-center"):
                ui.label("TIME").classes("col-span-1")
                ui.label("COIN").classes("col-span-1")
                ui.label("ACTION").classes("col-span-1")
                ui.label("CONF/VAL").classes("col-span-1")
                ui.label("PRICE").classes("col-span-1")
                ui.label("TP/SL").classes("col-span-1")
                ui.label("REASON").classes("col-span-3")
                ui.label("NEWS").classes("col-span-3")
            ai_decisions_container = ui.column().classes("w-full gap-1 overflow-y-auto h-[75vh]")

        # --- SECTION: STRATEGY REPORT (Quantitative Analysis) ---
        with ui.tab_panel(report_tab).classes("p-4"):
            with ui.row().classes("items-center justify-between w-full mb-4"):
                ui.label("📊 QUANTITATIVE PERFORMANCE AUDIT").classes(
                    "text-lg font-bold text-white"
                )

                async def refresh_report():
                    """Aggregates historical trades and predicts AI efficacy."""
                    full_story = ctx.memory.get_full_trade_story()
                    for row in full_story:
                        entry, exit, peak = row.get('entry_price'), row.get('exit_price'), row.get('peak_price')
                        if entry and exit:
                            row['roi'] = f"%{((exit - entry) / entry * 100 if row['action'] == 'LONG' else (entry - exit) / entry * 100):.2f}"
                        else:
                            row['roi'] = "-"
                        row['entry_price'] = f"{entry:.4f}" if entry else "-"
                        row['exit_price'] = f"{exit:.4f}" if exit else "-"
                        row['peak_price'] = f"{peak:.4f}" if peak else "-"
                    
                    report_table.rows = full_story
                    report_table.update()
                    ui.notify("Performance Data Hydrated.", type="info")

                ui.button("REBUILD REPORT", icon="refresh", on_click=refresh_report).props("outline size=sm")

            columns = [
                {"name": "time", "label": "Timestamp", "field": "time", "sortable": True, "align": "left"},
                {"name": "symbol", "label": "Asset", "field": "symbol", "sortable": True, "align": "left"},
                {"name": "action", "label": "Side", "field": "action", "align": "center"},
                {"name": "entry_price", "label": "Entry", "field": "entry_price", "align": "right"},
                {"name": "exit_price", "label": "Exit", "field": "exit_price", "align": "right"},
                {"name": "peak_price", "label": "Extreme MFM", "field": "peak_price", "align": "right"},
                {"name": "roi", "label": "ROI (%)", "field": "roi", "align": "right"},
                {"name": "pnl", "label": "PnL ($)", "field": "pnl", "sortable": True, "align": "right"},
                {"name": "close_reason", "label": "Exit Type", "field": "close_reason", "align": "left"},
                {"name": "ai_reason", "label": "Predictive Logic", "field": "ai_reason", "align": "left"}
            ]
            report_table = ui.table(columns=columns, rows=[], row_key="time").classes("w-full bg-gray-900 text-gray-300")

        # --- SECTION: MARKET (Price Discovery) ---
        with ui.tab_panel(market_tab).classes("p-4"):
            ui.label("📡 REAL-TIME PRICE DISCOVERY (WebSocket Ingestion)").classes(
                "text-lg font-bold mb-4 text-white"
            )
            market_grid = ui.grid(columns=5).classes("w-full gap-3")

        # --- SECTION: HISTORY (Audit Trail) ---
        with ui.tab_panel(history_tab).classes("p-4"):
            ui.label("📜 HISTORICAL EXECUTION LOGS").classes(
                "text-lg font-bold mb-4 text-white"
            )
            history_container = ui.column().classes("w-full gap-2")

    # --- UI SYNCHRONIZATION LOOP ---
    def refresh_ui():
        """
        Periodically updates all UI components with the latest application state.
        
        This loop ensures the dashboard remains a high-fidelity representation
        of the underlying asynchronous trading engine.
        """
        try:
            exchange = ctx.exchange

            # Update Metrics
            bal_label.set_text(f"${exchange.balance:.2f}")
            pnl_label.set_text(f"${exchange.total_pnl:.2f}")
            pnl_label.classes(replace=f"text-xl font-mono font-bold {'text-positive' if exchange.total_pnl >= 0 else 'text-negative'}")

            hist = exchange.history
            total_closed = len(hist)
            wins = len([t for t in hist if t["pnl"] > 0])
            wr = (wins / total_closed * 100) if total_closed > 0 else 0
            win_label.set_text(f"%{wr:.1f} ({wins}/{total_closed})")
            pos_count_label.set_text(str(len(exchange.positions)))

            # Refresh Positions View
            positions_container.clear()
            if not exchange.positions:
                with positions_container:
                    ui.label("Engine Status: Scanning for Alpha Signals...").classes(
                        "text-gray-600 italic text-sm w-full text-center mt-10"
                    )

            for sym, pos in exchange.positions.items():
                pnl = pos["pnl"]
                pnl_color, border_color = ("text-positive", "border-positive") if pnl >= 0 else ("text-negative", "border-negative")

                with positions_container:
                    with ui.card().classes(f"w-full bg-gray-800 border-l-4 {border_color} p-3 flex flex-row justify-between items-center"):
                        with ui.column().classes("gap-0"):
                            with ui.row().classes("gap-2 items-center"):
                                ui.label(sym.upper()).classes("font-bold text-lg text-white")
                                ui.label(f"{pos['side']} {pos['lev']}x").classes(
                                    f"text-xs px-1 rounded {'bg-green-900 text-green-300' if pos['side']=='LONG' else 'bg-red-900 text-red-300'}"
                                )
                            ui.label(f"Entry: {pos['entry']}").classes("text-[10px] text-gray-500")

                        with ui.column().classes("items-center"):
                            ui.label(f"{pos['current_price']}").classes("font-mono font-bold text-md text-white")
                            ui.label("MARK PRICE").classes("text-[10px] text-gray-500")

                        with ui.column().classes("items-end"):
                            ui.label(f"${pnl:.2f}").classes(f"font-bold text-xl {pnl_color}")
                            with ui.row().classes("gap-2 text-[10px] text-gray-400"):
                                ui.label(f"TP: {pos['tp']:.2f}")
                                ui.label(f"SL: {pos['sl']:.2f}")

                                # Position Expiry Countdown
                                exp_time = pos.get("expiry_time", 0)
                                if exp_time > 0:
                                    remaining_sec = max(0, exp_time - time.time())
                                    mins, secs = int(remaining_sec // 60), int(remaining_sec % 60)
                                    time_color = "text-red-400" if remaining_sec < 60 else "text-gray-400"
                                    ui.label(f"⏳ {mins}m {secs}s").classes(f"{time_color} font-mono")

            # Refresh AI Decisions View
            ai_decisions_container.clear()
            with ai_decisions_container:
                if not ctx.ai_decisions:
                    ui.label("Inference Engine: Idle.").classes("text-gray-600 italic p-4")
                for d in reversed(ctx.ai_decisions):
                    action_col = "text-green-400 font-bold" if d["action"] == "LONG" else ("text-red-400 font-bold" if d["action"] == "SHORT" else "text-gray-500")
                    with ui.row().classes('w-full grid grid-cols-12 text-[11px] py-1 border-b border-gray-800 items-center hover:bg-gray-800/50'):
                        ui.label(d['time']).classes('col-span-1 text-gray-400 font-mono')
                        ui.label(d['symbol']).classes('col-span-1 font-bold text-blue-300')
                        ui.label(d['action']).classes(f'col-span-1 {action_col} font-bold')
                        ui.label(f"%{d.get('confidence',0)} / {d.get('validity_minutes',0)}m").classes('col-span-1 text-yellow-500 font-mono')
                        ui.label(f"{d.get('price',0)}").classes('col-span-1 text-gray-400 font-mono')
                        ui.label(f"{d.get('tp_pct',0)} / {d.get('sl_pct',0)}").classes('col-span-1 text-blue-200 font-mono')
                        ui.label(d.get('reason', 'N/A')).classes('col-span-3 text-gray-300 truncate').tooltip(d.get('reason'))
                        ui.label(d.get('news_snippet', 'N/A')).classes('col-span-3 text-gray-500 truncate italic').tooltip(d.get('news_snippet'))

            # Refresh Live Market Cards
            market_grid.clear()
            with market_grid:
                active_coins = {k: v for k, v in ctx.market_memory.items() if v.current_price > 0}
                if not active_coins:
                    ui.label("Ingesting Market Streams...").classes("col-span-5 text-center text-gray-600 mt-10")
                for pair, buffer in active_coins.items():
                    change_1h = buffer.get_change(60)
                    bg_col, txt_col = ("bg-green-900/30", "text-green-400") if change_1h >= 0 else ("bg-red-900/30", "text-red-400")
                    with ui.card().classes(f"{bg_col} border border-gray-700 p-2 gap-1"):
                        ui.label(pair.upper().replace("USDT", "")).classes("font-bold text-xs text-gray-300")
                        ui.label(f"{buffer.current_price:.4f}").classes("font-mono text-sm text-white")
                        ui.label(f"%{change_1h:.2f}").classes(f"text-xs {txt_col}")

            # Refresh History Table
            history_container.clear()
            with history_container:
                if not exchange.history:
                    ui.label("No historical outcomes recorded.").classes('text-gray-600 italic p-4')
                else:
                    with ui.row().classes('w-full grid grid-cols-6 text-xs font-bold text-gray-500 border-b border-gray-700 pb-1'):
                        ui.label('TIME'); ui.label('ASSET'); ui.label('REALIZED PNL'); ui.label('MAX MFM'); ui.label('EXIT TYPE'); ui.label('SIDE')
                    
                    for trade in reversed(exchange.history[-20:]):
                        col = "text-green-400" if trade['pnl'] > 0 else "text-red-400"
                        with ui.row().classes('w-full grid grid-cols-6 text-xs py-1 border-b border-gray-800 items-center hover:bg-gray-800/50'):
                            ui.label(trade['time']).classes('text-gray-400')
                            ui.label(trade['symbol']).classes('font-bold text-gray-300')
                            ui.label(f"${trade['pnl']:.2f}").classes(f"font-bold {col}")
                            ui.label(f"{trade.get('peak', 0):.4f}").classes('text-yellow-500 font-mono')
                            ui.label(trade['reason']).classes('text-gray-500 truncate')
                            ui.label(trade['side']).classes(f"{'text-green-300' if trade['side']=='LONG' else 'text-red-300'}")

        except Exception as e:
            print(f"Dashboard Refresh Fault: {e}")

    # Synchronous UI heart-beat (1Hz)
    ui.timer(1.0, refresh_ui)
    return log_container

