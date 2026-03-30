import json
import os

class DatasetManager:
    def __init__(self, filename="training_dataset.jsonl"):
        self.filename = filename
        self.open_trades = {}

    def log_trade_entry(self, symbol, news, price_data, ai_decision, search_context="", entry_price=0.0):
        """
        Buffers trade data in memory when a position is opened.
        """
        self.open_trades[symbol] = {
            "news": news,
            "price_data": price_data, 
            "search_context": search_context,
            "original_decision": ai_decision,
            "entry_price": entry_price
        }

    # TODO: consider splitting
    def log_trade_exit(self, symbol, pnl, exit_reason, peak_price=0.0):
        """
        Analyzes trade outcome on close and generates training data (Hindsight Labeling).
        """
        if symbol not in self.open_trades:
            return

        trade_data = self.open_trades.pop(symbol)
        
        entry_price = trade_data.get('entry_price', 0.0)
        original_decision = trade_data['original_decision']
        original_action = original_decision.get('action')
        
        # --- TRAINING LOGIC (HINDSIGHT LABELING) ---
        ideal_response = {}
        
        if pnl > 0:
            ideal_response = original_decision
            ideal_response['reason'] += f" [VALIDATED: Trade made profit: {pnl:.2f} USDT]"
        
        else:
            max_favorable_move_pct = 0.0
            
            if entry_price > 0 and peak_price > 0:
                if original_action == 'LONG':
                    max_favorable_move_pct = (peak_price - entry_price) / entry_price * 100
                elif original_action == 'SHORT':
                    max_favorable_move_pct = (entry_price - peak_price) / entry_price * 100
            
            # --- CORRECTION LOGIC ---
            
            if max_favorable_move_pct > 0.5:
                ideal_response = original_decision.copy()
                
                new_tp = round(max_favorable_move_pct * 0.8, 2)
                if new_tp < 0.2: new_tp = 0.5 # Minimum protection
                
                ideal_response['tp_pct'] = new_tp
                ideal_response['reason'] = f"Correction: Direction was correct (Moved {max_favorable_move_pct:.2f}%), but TP was too high. Lower TP to {new_tp}%."
                
            else:
                ideal_response = {
                    "action": "HOLD",
                    "confidence": 100,
                    "reason": f"Correction: The original trade ({original_action}) resulted in a loss of {pnl:.2f} USDT. Safer to wait."
                }

        system_prompt = "You are a crypto trading AI. Analyze the news and market data to decide direction."
        
        user_input = f"""DETECTED COIN: {symbol}
MARKET DATA: {trade_data['price_data']}
NEWS: "{trade_data['news']}"
RESEARCH: "{trade_data['search_context']}"
"""
        
        entry = {
            "instruction": system_prompt,
            "input": user_input.strip(),
            "output": json.dumps(ideal_response)
        }

        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        print(f"[DATASET] Entry saved: {symbol} (Peak: {peak_price} | PnL: {pnl:.2f})")