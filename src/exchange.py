import time 

class PaperExchange:
    """
    A simulated exchange for paper trading.
    """
    def __init__(self, balance):
        self.balance = balance
        self.positions = {}
        self.total_pnl = 0.0
        self.history = []

    def open_position(self, symbol, side, price, tp_pct, sl_pct, amount_usdt, leverage, validity, app_state, decision_id):
        """Opens a simulated position."""
        if not app_state.is_running:
            return "Bot paused.", "warning"

        symbol = symbol.lower()

        if symbol in self.positions:
            return f"{symbol.upper()} already open!", "warning"

        if self.balance < amount_usdt:
            return "Insufficient Balance!", "error"

        margin = float(amount_usdt)
        qty = (amount_usdt * leverage) / price
        
        if side == 'LONG':
            tp = price * (1 + tp_pct/100)
            sl = price * (1 - sl_pct/100)
        else:
            tp = price * (1 - tp_pct/100)
            sl = price * (1 + sl_pct/100)

        self.positions[symbol] = {
            'entry': price,
            'current_price': price,
            'highest_price': price,
            'lowest_price': price,
            'side': side,
            'margin': margin,
            'qty': qty,
            'lev': leverage,
            'tp': tp,
            'sl': sl,
            'pnl': 0.0,
            'start_time': time.time(),
            'validity': validity,
            'expiry_time': time.time() + validity * 60,
            'decision_id': decision_id
        }
        
        self.balance -= margin
        return f"POSITION OPENED: {symbol.upper()} {side} | Entry: {price} | TP: {tp_pct}% | SL: {sl_pct}% | VM: {validity}m", "info"
    
    def check_positions(self, symbol, current_price):
        """Updates positions with current price and checks for exit conditions (TP/SL/Expiry)."""
        if symbol not in self.positions:
            return None, None, None, 0.0, 0.0, None

        pos = self.positions[symbol]
        side = pos['side']
        entry = pos['entry']
        
        peak_price = entry
        if side == 'LONG':
            current_high = pos.get('highest_price', entry)
            if current_price > current_high:
                pos['highest_price'] = current_price
            peak_price = pos['highest_price']
        else:
            current_low = pos.get('lowest_price', entry)
            if pos.get('lowest_price', 0) == 0 or current_price < current_low:
                pos['lowest_price'] = current_price
            peak_price = pos['lowest_price']

        if side == 'LONG':
            pnl = (current_price - entry) * pos['qty']
        else:
            pnl = (entry - current_price) * pos['qty']
            
        pos['pnl'] = pnl

        roi = 0.0
        if side == 'LONG':
            roi = (current_price - entry) / entry * 100
            if roi > 0.8 and pos['sl'] < entry: pos['sl'] = entry * 1.0015
            if roi > 1.5:
                new_sl = entry * 1.01 
                if pos['sl'] < new_sl: pos['sl'] = new_sl
        elif side == 'SHORT':
            roi = (entry - current_price) / entry * 100
            if roi > 0.8 and pos['sl'] > entry: pos['sl'] = entry * 0.9985
            if roi > 1.5:
                new_sl = entry * 0.99
                if pos['sl'] > new_sl: pos['sl'] = new_sl

        close_reason = None
        
        if side == 'LONG':
            if current_price >= pos['tp']: close_reason = "TAKE PROFIT"
            elif current_price <= pos['sl']: close_reason = "STOP LOSS"
        else:
            if current_price <= pos['tp']: close_reason = "TAKE PROFIT"
            elif current_price >= pos['sl']: close_reason = "STOP LOSS"

        if time.time() > pos.get('expiry_time', float('inf')):
            close_reason = "EXPIRED"

        if close_reason:
            decision_id = pos.get('decision_id')
            log_msg = f"CLOSED: {symbol.upper()} ({close_reason}) | PnL: {pnl:.2f} USDT | Entry: {entry} | Exit: {current_price}"
            color = "success" if pnl > 0 else "error"
            
            self.close_position(symbol, close_reason, pnl)
            
            return log_msg, color, symbol, pnl, peak_price, decision_id

        return None, None, None, 0.0, 0.0, None
    
    def close_position(self, symbol, reason, pnl):
        """Finalizes position closure and updates balance/history."""
        symbol = symbol.lower()

        if symbol not in self.positions: 
            return "Error: Position not found", "error"
        
        pos = self.positions[symbol]
        
        self.balance += pos['margin'] + pnl
        self.total_pnl += pnl
        
        peak_price = 0.0
        if pos['side'] == 'LONG': peak_price = pos.get('highest_price', pos['entry'])
        elif pos['side'] == 'SHORT': peak_price = pos.get('lowest_price', pos['entry'])
        
        record = {
            'time': time.strftime("%H:%M:%S"),
            'symbol': symbol.upper(),
            'side': pos['side'],
            'entry': pos['entry'],
            'exit': pos.get('current_price', 0),
            'pnl': pnl,
            'peak': peak_price,
            'reason': reason
        }
        self.history.append(record)
        
        del self.positions[symbol]
        
        return f"CLOSED: {symbol.upper()} ({reason}) | PnL: {pnl:.2f} USDT", "success" if pnl > 0 else "error"
    
    def open_position_test(self, symbol, side, price, tp_pct, sl_pct, amount_usdt, leverage, validity, app_state, decision_id, now_ts):
        """Opens a simulated position for backtesting/replay purposes."""
        if not app_state.is_running:
            return "Bot paused.", "warning"

        symbol = symbol.lower()
        if symbol in self.positions:
            return f"{symbol.upper()} already open!", "warning"

        margin = amount_usdt
        qty = (amount_usdt * leverage) / price
        
        if side == 'LONG':
            tp = price * (1 + tp_pct/100)
            sl = price * (1 - sl_pct/100)
        else:
            tp = price * (1 - tp_pct/100)
            sl = price * (1 + sl_pct/100)

        self.positions[symbol] = {
            'entry': price,
            'current_price': price,
            'highest_price': price,
            'lowest_price': price,
            'side': side,
            'margin': margin,
            'qty': qty,
            'lev': leverage,
            'tp': tp,
            'sl': sl,
            'pnl': 0.0,
            'start_time': now_ts,
            'validity': validity,
            'expiry_time': now_ts + (validity * 60),
            'decision_id': decision_id
        }
        
        self.balance -= margin
        return f"[TEST] POSITION OPENED: {symbol.upper()} {side} | Entry: {price}", "info"

    def check_positions_test(self, symbol, current_price, now_ts):
        """Updates positions with historical price for backtesting."""
        symbol = symbol.lower()
        if symbol not in self.positions:
            return None, None, None, 0.0, 0.0, None

        pos = self.positions[symbol]
        side = pos['side']
        entry = pos['entry']
        
        if side == 'LONG':
            if current_price > pos.get('highest_price', entry):
                pos['highest_price'] = current_price
            peak_price = pos['highest_price']
            pnl = (current_price - entry) * pos['qty']
        else:
            if current_price < pos.get('lowest_price', entry):
                pos['lowest_price'] = current_price
            peak_price = pos['lowest_price']
            pnl = (entry - current_price) * pos['qty']
                
        pos['pnl'] = pnl

        roi = 0.0
        if side == 'LONG':
            roi = (current_price - entry) / entry * 100
            if roi > 0.8 and pos['sl'] < entry: pos['sl'] = entry * 1.0015 
            if roi > 1.5:
                new_sl = entry * 1.01 
                if pos['sl'] < new_sl: pos['sl'] = new_sl
        elif side == 'SHORT':
            roi = (entry - current_price) / entry * 100
            if roi > 0.8 and pos['sl'] > entry: pos['sl'] = entry * 0.9985
            if roi > 1.5:
                new_sl = entry * 0.99
                if pos['sl'] > new_sl: pos['sl'] = new_sl

        close_reason = None
        if side == 'LONG':
            if current_price >= pos['tp']: close_reason = "TAKE PROFIT"
            elif current_price <= pos['sl']: close_reason = "STOP LOSS"
        else:
            if current_price <= pos['tp']: close_reason = "TAKE PROFIT"
            elif current_price >= pos['sl']: close_reason = "STOP LOSS"

        if now_ts > pos.get('expiry_time', float('inf')):
            close_reason = "EXPIRED"

        if close_reason:
            decision_id = pos.get('decision_id')
            log_msg, color = self.close_position_test(symbol, close_reason, pnl, now_ts)
            return log_msg, color, symbol, pnl, peak_price, decision_id

        return None, None, None, 0.0, 0.0, None

    def close_position_test(self, symbol, reason, pnl, now_ts):
        """Closes a position and records it for backtesting history."""
        symbol = symbol.lower()
        if symbol not in self.positions: 
            return "Error: Position not found", "error"
        
        pos = self.positions[symbol]
        self.balance += pos['margin'] + pnl
        self.total_pnl += pnl
        
        readable_time = time.strftime("%H:%M:%S", time.gmtime(now_ts))
        
        record = {
            'time': readable_time,
            'symbol': symbol.upper(),
            'side': pos['side'],
            'entry': pos['entry'],
            'exit': pos.get('current_price', 0),
            'pnl': pnl,
            'peak': pos.get('highest_price') if pos['side'] == 'LONG' else pos.get('lowest_price'),
            'reason': reason
        }
        self.history.append(record)
        del self.positions[symbol]
        
        return f"[TEST] CLOSED: {symbol.upper()} | PnL: {pnl:.2f} USDT", "success"