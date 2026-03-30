from binance import AsyncClient
from binance.enums import *
import math

class BinanceExecutionEngine:
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.symbol_info = {} 

    async def connect(self):
        try:
            self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
            info = await self.client.futures_exchange_info()
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
                    continue
            env = "TESTNET" if self.testnet else "MAINNET"
        except Exception as e:
            print(f"[EXCHANGE ERROR] {e}")

    def _get_precision(self, size):
        if size == 0: return 0
        return int(round(-math.log(size, 10), 0))

    def _round_step(self, quantity, step_size):
        """
        Rounds quantity down to the nearest multiple of step_size.
        """
        if step_size == 0: return quantity
        precision = self._get_precision(step_size)
        qty = int(quantity / step_size) * step_size
        return float(f"{qty:.{precision}f}")

    def _ceil_step(self, quantity, step_size):
        """
        Rounds quantity up to the nearest multiple of step_size.
        """
        if step_size == 0: return quantity
        precision = self._get_precision(step_size)
        qty = math.ceil(quantity / step_size) * step_size
        return float(f"{qty:.{precision}f}")

    def _round_price(self, price, tick_size):
        """
        Rounds price to the nearest multiple of tick_size.
        """
        if tick_size == 0: return price
        precision = self._get_precision(tick_size)
        price = round(price / tick_size) * tick_size
        return float(f"{price:.{precision}f}")

    async def execute_trade(self, symbol, side, amount_usdt, leverage, tp_pct, sl_pct):
        if not self.client: return
        sym = symbol.upper()
        sym_lower = symbol.lower()
        
        try:
            # 1. Leverage and Price
            await self.client.futures_change_leverage(symbol=sym, leverage=leverage)
            ticker = await self.client.futures_symbol_ticker(symbol=sym)
            current_market_price = float(ticker['price'])
            
            # 2. Base Quantity Calculation
            raw_qty = (amount_usdt * leverage) / current_market_price
            
            step_size = self.symbol_info[sym_lower]['stepSize']
            min_qty = self.symbol_info[sym_lower]['minQty']
            min_notional = self.symbol_info[sym_lower]['minNotional']
            
            qty = self._round_step(raw_qty, step_size)
            
            # --- CHECK 1: QUANTITY LIMIT ---
            if qty < min_qty:
                print(f"[WARNING] Qty ({qty}) below min_qty ({min_qty}). Adjusting.")
                qty = min_qty
            
            # --- CHECK 2: NOTIONAL LIMIT ---
            current_notional_value = qty * current_market_price
            
            if current_notional_value < min_notional:
                print(f"[WARNING] Notional ({current_notional_value:.2f}) below min_notional ({min_notional}). Forcing...")
                
                required_qty = min_notional / current_market_price
                
                # Round up with 1% safety margin to ensure notional threshold is met
                qty = self._ceil_step(required_qty * 1.01, step_size)
                
                print(f"[INFO] New Qty: {qty} (Est. Notional: {qty * current_market_price:.2f})")

            # 3. Open Position
            side_enum = SIDE_BUY if side == 'LONG' else SIDE_SELL
            order = await self.client.futures_create_order(
                symbol=sym, side=side_enum, type=ORDER_TYPE_MARKET, quantity=qty
            )
            
            # Get actual execution price
            filled_price = float(order.get('avgPrice', 0.0))
            entry_price = filled_price if filled_price > 0 else current_market_price
            
            # 4. Place TP/SL
            try:
                await self._place_tp_sl(sym, side, entry_price, tp_pct, sl_pct)
                print(f"[API] {sym} {side} @ {entry_price} (Qty: {qty})")
            except Exception as e:
                return "TP/SL Placement Error"
            
            return "Position opened"        
        except Exception as e: 
            print(f"[API ERROR] {e}")
            return "Position Opening Error"


    async def _place_tp_sl(self, symbol, side, entry, tp_pct, sl_pct):
        try:
            tick = self.symbol_info[symbol.lower()]['tickSize']
            
            # Direction setting
            if side == 'LONG':
                tp_raw = entry * (1 + tp_pct/100)
                sl_raw = entry * (1 - sl_pct/100)
                close_side = 'SELL' 
            else: # SHORT
                tp_raw = entry * (1 - tp_pct/100)
                sl_raw = entry * (1 + sl_pct/100)
                close_side = 'BUY' 

            # Negative price protection
            if tp_raw <= tick: tp_raw = entry + (tick * 10) if side=='LONG' else entry - (tick * 10)
            if sl_raw <= tick: sl_raw = entry - (tick * 10) if side=='LONG' else entry + (tick * 10)

            # Rounding
            tp = self._round_price(tp_raw, tick)
            sl = self._round_price(sl_raw, tick)
            
            print(f"[INFO] Calculating TP/SL: TP={tp} | SL={sl}")

            # --- STOP LOSS ORDER ---
            # workingType='MARK_PRICE' protects against volatility spikes
            await self.client.futures_create_algo_order(
                symbol=symbol, 
                side=close_side, 
                type='STOP_MARKET', 
                triggerPrice=sl,
                closePosition=True, 
                workingType='MARK_PRICE',
                algoType='CONDITIONAL'
            )
            
            # --- TAKE PROFIT ORDER ---
            await self.client.futures_create_algo_order(
                symbol=symbol, 
                side=close_side, 
                type='TAKE_PROFIT_MARKET', 
                triggerPrice=tp,
                closePosition=True, 
                workingType='MARK_PRICE',
                algoType='CONDITIONAL'
            )
            
            print(f"[API] TP/SL placed successfully for {symbol}")

        except Exception as e: 
            print(f"[TP/SL ERROR] {e}")

    async def close(self):
        if self.client: await self.client.close_connection()
    
    async def close_position_market(self, symbol):
        if not self.client: return
        sym = symbol.upper()
        try:
            await self.client.futures_cancel_all_open_orders(symbol=sym)
            positions = await self.client.futures_position_information(symbol=sym)
            for p in positions:
                amt = float(p['positionAmt'])
                if amt != 0:
                    side = SIDE_SELL if amt > 0 else SIDE_BUY
                    await self.client.futures_create_order(symbol=sym, side=side, type=ORDER_TYPE_MARKET, quantity=abs(amt))
                    print(f"[API] {sym} position closed market.")
        except Exception as e: print(f"[CLOSE ERROR] {e}")

    async def fetch_missing_data(self, symbol):
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
        Fetches current USDT balance from Binance Futures account.
        Returns: (Total Balance, Available Balance)
        """
        if not self.client:
            print("[BALANCE] API not connected.")
            return 0.0, 0.0
            
        try:
            balances = await self.client.futures_account_balance()
            
            for asset in balances:
                if asset['asset'] == 'USDT':
                    total_balance = float(asset['balance'])
                    available_balance = float(asset.get('availableBalance', 0.0))                    
                    print(f"[WALLET] Total: {total_balance:.2f} USDT | Available: {available_balance:.2f} USDT")
                    return total_balance, available_balance
            
            print("[BALANCE] USDT asset not found.")
            return 0.0, 0.0
            
        except Exception as e:
            print(f"[BALANCE ERROR] {e}")
            return 0.0, 0.0

    async def get_extended_metrics(self, symbol):
        """
        Fetches 24h Volume and Funding Rate for advanced analysis.
        """
        if not self.client: return "Unknown", 0.0

        try:
            # 1. 24h Ticker Stats (for Volume)
            ticker_stats = await self.client.futures_ticker(symbol=symbol.upper())
            volume_usdt = float(ticker_stats.get('quoteVolume', 0))
            
            # Format Volume (Billion/Million)
            if volume_usdt > 1_000_000_000:
                vol_str = f"${volume_usdt / 1_000_000_000:.2f}B"
            else:
                vol_str = f"${volume_usdt / 1_000_000:.2f}M"

            # 2. Funding Rate
            premium_index = await self.client.futures_mark_price(symbol=symbol.upper())
            funding_rate = float(premium_index.get('lastFundingRate', 0)) * 100 
            
            return vol_str, funding_rate

        except Exception as e:
            print(f"[METRIC ERROR] {symbol}: {e}")
            return "Unknown", 0.0
        
    async def get_order_book_imbalance(self, symbol, limit=100):
        """
        Measures Buyer/Seller imbalance in the order book.
        """
        if not self.client: return 0.0, "No Connection"
        
        try:
            depth = await self.client.futures_order_book(symbol=symbol.upper(), limit=limit)
            
            total_bids = sum([float(x[1]) for x in depth['bids']])
            total_asks = sum([float(x[1]) for x in depth['asks']])
            
            if total_bids + total_asks == 0: return 0.0, "No Volume"

            imbalance = (total_bids - total_asks) / (total_bids + total_asks)
            return imbalance, f"Bids: {total_bids:.2f} | Asks: {total_asks:.2f}"
            
        except Exception as e:
            print(f"[DEPTH ERROR] {e}")
            return 0.0, "Error"