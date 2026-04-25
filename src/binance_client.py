"""
Binance Futures Execution Engine

This module provides a high-level interface for interacting with the Binance 
Futures API. It handles order execution, TP/SL placement using algorithmic orders,
precision rounding, and market data retrieval.

The engine supports both Mainnet and Testnet environments and implements 
robust error handling for exchange connectivity and order validation.
"""

from binance import AsyncClient
from binance.enums import *
import math

class BinanceExecutionEngine:
    """
    Main execution bridge between the bot and Binance Futures API.
    
    Handles connectivity, order management, and real-time balance/metric 
    synchronization.
    
    Attributes:
        api_key (str): Binance API credential.
        api_secret (str): Binance Secret credential.
        testnet (bool): If True, connects to the Binance Testnet environment.
        client (AsyncClient): Underlying asynchronous Binance client.
        symbol_info (dict): Metadata cache for symbols (step size, tick size, etc.).
    """
    def __init__(self, api_key, api_secret, testnet=False):
        """
        Initializes the execution engine.
        
        Args:
            api_key (str): Binance API Key.
            api_secret (str): Binance API Secret.
            testnet (bool): Whether to use the Testnet environment.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.symbol_info = {} 

    async def connect(self):
        """
        Establishes an asynchronous connection to the Binance API.
        
        Fetches exchange information to populate the symbol metadata cache
        required for precision rounding and minimal order validation.
        """
        try:
            self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
            info = await self.client.futures_exchange_info()
            
            # Map filters for easy access during trade execution
            for s in info['symbols']:
                filters = {f['filterType']: f for f in s['filters']}
                try:
                    # Fetch MIN_NOTIONAL filter; default is 5.0 for altcoins
                    min_notional = 5.0
                    if 'MIN_NOTIONAL' in filters:
                        min_notional = float(filters['MIN_NOTIONAL']['notional'])
                    
                    self.symbol_info[s['symbol'].lower()] = {
                        'stepSize': float(filters['LOT_SIZE']['stepSize']),
                        'tickSize': float(filters['PRICE_FILTER']['tickSize']),
                        'minQty': float(filters['LOT_SIZE']['minQty']),
                        'minNotional': min_notional
                    }
                except Exception:
                    # Skip symbols with unconventional filter structures
                    continue
        except Exception as e:
            print(f"[EXCHANGE ERROR] Failed to initialize Binance connection: {e}")

    def _get_precision(self, size):
        """
        Determines decimal precision based on filter size.
        
        Args:
            size (float): The step or tick size from exchange info.
            
        Returns:
            int: Number of decimal places.
        """
        if size == 0: return 0
        return int(round(-math.log(size, 10), 0))

    def _round_step(self, quantity, step_size):
        """
        Rounds quantity down to the nearest valid step increment.
        
        Args:
            quantity (float): Input quantity.
            step_size (float): Minimum quantity increment.
            
        Returns:
            float: Rounded quantity.
        """
        if step_size == 0: return quantity
        precision = self._get_precision(step_size)
        qty = int(quantity / step_size) * step_size
        return float(f"{qty:.{precision}f}")

    def _ceil_step(self, quantity, step_size):
        """
        Rounds quantity up to the nearest valid step increment.
        
        Args:
            quantity (float): Input quantity.
            step_size (float): Minimum quantity increment.
            
        Returns:
            float: Ceiled quantity.
        """
        if step_size == 0: return quantity
        precision = self._get_precision(step_size)
        qty = math.ceil(quantity / step_size) * step_size
        return float(f"{qty:.{precision}f}")

    def _round_price(self, price, tick_size):
        """
        Rounds price to the nearest valid price tick.
        
        Args:
            price (float): Input price.
            tick_size (float): Minimum price increment.
            
        Returns:
            float: Rounded price.
        """
        if tick_size == 0: return price
        precision = self._get_precision(tick_size)
        price = round(price / tick_size) * tick_size
        return float(f"{price:.{precision}f}")

    async def execute_trade(self, symbol, side, amount_usdt, leverage, tp_pct, sl_pct):
        """
        Executes a full trade sequence: Leverage change -> Market Order -> TP/SL placement.
        
        Args:
            symbol (str): Asset pair ticker.
            side (str): Trade direction ('LONG' or 'SHORT').
            amount_usdt (float): Margin amount in USDT.
            leverage (int): Target leverage for the position.
            tp_pct (float): Take-profit percentage from entry.
            sl_pct (float): Stop-loss percentage from entry.
            
        Returns:
            str: Status message indicating success or failure reason.
        """
        if not self.client: return "API not connected"
        sym = symbol.upper()
        sym_lower = symbol.lower()
        
        try:
            # 1. Update Account Leverage and Fetch Market Ticker
            await self.client.futures_change_leverage(symbol=sym, leverage=leverage)
            ticker = await self.client.futures_symbol_ticker(symbol=sym)
            current_market_price = float(ticker['price'])
            
            # 2. Calculated Quantity based on USDT Margin and Leverage
            raw_qty = (amount_usdt * leverage) / current_market_price
            
            step_size = self.symbol_info[sym_lower]['stepSize']
            min_qty = self.symbol_info[sym_lower]['minQty']
            min_notional = self.symbol_info[sym_lower]['minNotional']
            
            qty = self._round_step(raw_qty, step_size)
            
            # --- CHECK 1: QUANTITY LIMIT VALIDATION ---
            if qty < min_qty:
                print(f"[WARNING] Qty ({qty}) below min_qty ({min_qty}). Adjusting.")
                qty = min_qty
            
            # --- CHECK 2: NOTIONAL LIMIT VALIDATION ---
            # Ensure the total position value meets Binance's minimum requirement
            current_notional_value = qty * current_market_price
            
            if current_notional_value < min_notional:
                print(f"[WARNING] Notional ({current_notional_value:.2f}) below min_notional ({min_notional}). Forcing...")
                
                required_qty = min_notional / current_market_price
                
                # Apply 1% safety margin when ceiling to avoid floating point rejections
                qty = self._ceil_step(required_qty * 1.01, step_size)
                
                print(f"[INFO] Adjusted Qty for Notional: {qty} (Est: {qty * current_market_price:.2f})")

            # 3. Execution of Entry Market Order
            side_enum = SIDE_BUY if side == 'LONG' else SIDE_SELL
            order = await self.client.futures_create_order(
                symbol=sym, side=side_enum, type=ORDER_TYPE_MARKET, quantity=qty
            )
            
            # Retrieve realized entry price for accurate TP/SL calculation
            filled_price = float(order.get('avgPrice', 0.0))
            entry_price = filled_price if filled_price > 0 else current_market_price
            
            # 4. Placement of Protective Algorithmic Orders (TP/SL)
            try:
                await self._place_tp_sl(sym, side, entry_price, tp_pct, sl_pct)
                print(f"[API] {sym} {side} execution successful at {entry_price}")
            except Exception as e:
                return "TP/SL Placement Error"
            
            return "Position opened"        
        except Exception as e: 
            print(f"[API ERROR] Execution failed for {symbol}: {e}")
            return "Position Opening Error"


    async def _place_tp_sl(self, symbol, side, entry, tp_pct, sl_pct):
        """
        Generates and sends algorithmic TP/SL orders to the exchange.
        
        Args:
            symbol (str): Asset pair.
            side (str): Position side ('LONG'/'SHORT').
            entry (float): The actual entry price.
            tp_pct (float): Target profit percentage.
            sl_pct (float): Maximum loss percentage.
        """
        try:
            tick = self.symbol_info[symbol.lower()]['tickSize']
            
            # Calculate absolute price targets based on trade direction
            if side == 'LONG':
                tp_raw = entry * (1 + tp_pct/100)
                sl_raw = entry * (1 - sl_pct/100)
                close_side = 'SELL' 
            else: # SHORT
                tp_raw = entry * (1 - tp_pct/100)
                sl_raw = entry * (1 + sl_pct/100)
                close_side = 'BUY' 

            # Safeguard against calculation errors leading to zero/negative prices
            if tp_raw <= tick: tp_raw = entry + (tick * 10) if side=='LONG' else entry - (tick * 10)
            if sl_raw <= tick: sl_raw = entry - (tick * 10) if side=='LONG' else entry + (tick * 10)

            # Round prices to match exchange tick specifications
            tp = self._round_price(tp_raw, tick)
            sl = self._round_price(sl_raw, tick)
            
            print(f"[INFO] Configured TP: {tp} | SL: {sl}")

            # --- STOP LOSS CONFIGURATION ---
            # Using 'MARK_PRICE' to avoid stop-outs during temporary order book volatility
            await self.client.futures_create_algo_order(
                symbol=symbol, 
                side=close_side, 
                type='STOP_MARKET', 
                triggerPrice=sl,
                closePosition=True, 
                workingType='MARK_PRICE',
                algoType='CONDITIONAL'
            )
            
            # --- TAKE PROFIT CONFIGURATION ---
            await self.client.futures_create_algo_order(
                symbol=symbol, 
                side=close_side, 
                type='TAKE_PROFIT_MARKET', 
                triggerPrice=tp,
                closePosition=True, 
                workingType='MARK_PRICE',
                algoType='CONDITIONAL'
            )
            
            print(f"[API] Algo TP/SL synchronization complete for {symbol}")

        except Exception as e: 
            print(f"[TP/SL ERROR] Failed to place algo orders: {e}")

    async def close(self):
        """Closes the underlying asynchronous API connection."""
        if self.client: await self.client.close_connection()
    
    async def close_position_market(self, symbol):
        """
        Liquidates any open position for a symbol using a market order.
        
        Args:
            symbol (str): Ticker to close.
        """
        if not self.client: return
        sym = symbol.upper()
        try:
            # Cancel all pending orders first to avoid accidental fills
            await self.client.futures_cancel_all_open_orders(symbol=sym)
            positions = await self.client.futures_position_information(symbol=sym)
            
            for p in positions:
                amt = float(p['positionAmt'])
                if amt != 0:
                    side = SIDE_SELL if amt > 0 else SIDE_BUY
                    await self.client.futures_create_order(symbol=sym, side=side, type=ORDER_TYPE_MARKET, quantity=abs(amt))
                    print(f"[API] {sym} liquidated at market price.")
        except Exception as e: print(f"[CLOSE ERROR] Liquidation failed for {symbol}: {e}")

    async def fetch_missing_data(self, symbol):
        """
        Backfills historical market data for a symbol.
        
        Args:
            symbol (str): Target ticker.
            
        Returns:
            tuple: (List of price/time tuples, 24h Price Change Percent)
        """
        if not self.client: return None, 0.0
        try:
            klines = await self.client.futures_klines(symbol=symbol.upper(), interval=KLINE_INTERVAL_1MINUTE, limit=60)
            data = [(float(k[4]), int(k[0])/1000) for k in klines]
            ticker = await self.client.futures_ticker(symbol=symbol.upper())
            return data, float(ticker['priceChangePercent'])
        except Exception: 
            return None, 0.0
    
    async def get_usdt_balance(self):
        """
        Retrieves real-time USDT liquidity from the Futures wallet.
        
        Returns:
            tuple: (Total Wallet Balance, Available/Free Balance)
        """
        if not self.client:
            print("[BALANCE] Connection not established.")
            return 0.0, 0.0
            
        try:
            balances = await self.client.futures_account_balance()
            
            for asset in balances:
                if asset['asset'] == 'USDT':
                    total_balance = float(asset['balance'])
                    available_balance = float(asset.get('availableBalance', 0.0))                    
                    print(f"[WALLET] Liquidity Synced: {total_balance:.2f} USDT")
                    return total_balance, available_balance
            
            return 0.0, 0.0
            
        except Exception as e:
            print(f"[BALANCE ERROR] Failed to fetch account state: {e}")
            return 0.0, 0.0

    async def get_extended_metrics(self, symbol):
        """
        Collects sectoral market metrics: 24h Volume and Perpetual Funding Rate.
        
        Args:
            symbol (str): Target ticker.
            
        Returns:
            tuple: (Formatted Volume String, Funding Rate Percentage)
        """
        if not self.client: return "Unknown", 0.0

        try:
            # 1. 24h Quote Volume retrieval
            ticker_stats = await self.client.futures_ticker(symbol=symbol.upper())
            volume_usdt = float(ticker_stats.get('quoteVolume', 0))
            
            # Professional string formatting for volume metrics
            if volume_usdt > 1_000_000_000:
                vol_str = f"${volume_usdt / 1_000_000_000:.2f}B"
            else:
                vol_str = f"${volume_usdt / 1_000_000:.2f}M"

            # 2. Real-time Funding Rate retrieval
            premium_index = await self.client.futures_mark_price(symbol=symbol.upper())
            funding_rate = float(premium_index.get('lastFundingRate', 0)) * 100 
            
            return vol_str, funding_rate

        except Exception as e:
            print(f"[METRIC ERROR] Failed to fetch extended data for {symbol}: {e}")
            return "Unknown", 0.0
        
    async def get_order_book_imbalance(self, symbol, limit=100):
        """
        Quantifies buy/sell pressure imbalance from the top of the order book.
        
        Args:
            symbol (str): Target ticker.
            limit (int): Depth of book to analyze.
            
        Returns:
            tuple: (Imbalance Coefficient [-1, 1], Formatted depth summary string)
        """
        if not self.client: return 0.0, "No Connection"
        
        try:
            depth = await self.client.futures_order_book(symbol=symbol.upper(), limit=limit)
            
            total_bids = sum([float(x[1]) for x in depth['bids']])
            total_asks = sum([float(x[1]) for x in depth['asks']])
            
            if total_bids + total_asks == 0: return 0.0, "Illiquid Book"

            # Coefficient calculation: >0 indicates buying pressure, <0 indicates selling pressure
            imbalance = (total_bids - total_asks) / (total_bids + total_asks)
            return imbalance, f"Bids: {total_bids:.2f} | Asks: {total_asks:.2f}"
            
        except Exception as e:
            print(f"[DEPTH ERROR] Market depth analysis failed for {symbol}: {e}")
            return 0.0, "Data Retrieval Error"