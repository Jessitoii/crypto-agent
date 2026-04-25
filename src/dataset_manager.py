"""
Autonomous Hindsight Dataset Manager

This module implements a 'Hindsight Labeling' system. It monitors open 
simulated positions and, upon closure, evaluates whether the AI's 
original decision was optimal.

If a loss occurred but the direction was correct, it suggests a tighter 
Take-Profit (TP). If the direction was wrong, it corrects the ideal 
action to 'HOLD'. This creates a self-correcting feedback loop for 
model fine-tuning.
"""

import json
import os

class DatasetManager:
    """
    Manages the lifecycle of trade-to-dataset conversion.
    
    Attributes:
        filename (str): The JSONL file where training data is persisted.
        open_trades (dict): In-memory buffer for active trades.
    """
    def __init__(self, filename="training_dataset.jsonl"):
        """
        Initializes the DatasetManager.
        
        Args:
            filename (str): Path to the output dataset file.
        """
        self.filename = filename
        self.open_trades = {}

    def log_trade_entry(self, symbol, news, price_data, ai_decision, search_context="", entry_price=0.0):
        """
        Captures the initial state of a trade for later evaluation.
        
        Args:
            symbol (str): Asset ticker.
            news (str): News that triggered the trade.
            price_data (str): Snapshot of price changes at entry.
            ai_decision (dict): The original AI recommendation.
            search_context (str): Web research data used for the decision.
            entry_price (float): Execution price.
        """
        self.open_trades[symbol] = {
            "news": news,
            "price_data": price_data, 
            "search_context": search_context,
            "original_decision": ai_decision,
            "entry_price": entry_price
        }

    def log_trade_exit(self, symbol, pnl, exit_reason, peak_price=0.0):
        """
        Performs hindsight analysis on a closed trade and generates a training record.
        
        Evaluates the maximum favorable movement (MFM) to determine if the 
        entry logic was sound even if the exit (TP/SL) was suboptimal.
        
        Args:
            symbol (str): Asset ticker.
            pnl (float): Realized profit/loss.
            exit_reason (str): Label for the exit trigger.
            peak_price (float): The extreme price reached during the trade.
        """
        if symbol not in self.open_trades:
            return

        trade_data = self.open_trades.pop(symbol)
        
        entry_price = trade_data.get('entry_price', 0.0)
        original_decision = trade_data['original_decision']
        original_action = original_decision.get('action')
        
        # --- HINDSIGHT LABELING LOGIC ---
        ideal_response = {}
        
        # Case 1: Profitable Trade - Validate the original decision
        if pnl > 0:
            ideal_response = original_decision
            ideal_response['reason'] += f" [VALIDATED: Realized {pnl:.2f} USDT profit]"
        
        # Case 2: Loss-making Trade - Analyze for missed opportunities or errors
        else:
            max_favorable_move_pct = 0.0
            
            # Calculate how much the price moved in the predicted direction before hitting SL/Expiry
            if entry_price > 0 and peak_price > 0:
                if original_action == 'LONG':
                    max_favorable_move_pct = (peak_price - entry_price) / entry_price * 100
                elif original_action == 'SHORT':
                    max_favorable_move_pct = (entry_price - peak_price) / entry_price * 100
            
            # --- SUB-CASE: THE 'NEAR MISS' CORRECTION ---
            # If price moved > 0.5% in favor but hit SL, the direction was right but TP was too greedy.
            if max_favorable_move_pct > 0.5:
                ideal_response = original_decision.copy()
                
                # Suggest a 20% haircut on the actual peak as a new conservative TP
                new_tp = round(max_favorable_move_pct * 0.8, 2)
                if new_tp < 0.2: new_tp = 0.5 
                
                ideal_response['tp_pct'] = new_tp
                ideal_response['reason'] = (f"Correction: Correct direction ({max_favorable_move_pct:.2f}% move), "
                                          f"but TP was unrealistic. Suggested TP: {new_tp}%.")
                
            # --- SUB-CASE: THE 'FALSE POSITIVE' CORRECTION ---
            # If the trade never went in favor, the original action was likely a mistake.
            else:
                ideal_response = {
                    "action": "HOLD",
                    "confidence": 100,
                    "reason": f"Correction: The original {original_action} signal failed (PnL: {pnl:.2f}). In hindsight, HOLD was safer."
                }

        # Structure data in ChatML/Instruction format for fine-tuning
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

        # Synchronous write to ensure data integrity during shutdown/crashes
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        print(f"[DATASET] Synthetic Ground-Truth Saved: {symbol} (PnL: {pnl:.2f})")