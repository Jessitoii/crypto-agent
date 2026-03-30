import os
import aiofiles
import time 
import json

class TrainingDataCollector:
    """
    Collects trade decisions and market outcomes to generate training datasets.
    """
    def __init__(self, filename="data_collection.jsonl"):
        self.filename = filename
        self.pending_events = [] # Decisions pending outcome verification

    def log_decision(self, news, pair, initial_price, stats_1m, model_output):
        """
        Logs a bot decision to the pending list for later verification.
        """
        event = {
            "timestamp": time.time(),
            "news": news,
            "pair": pair,
            "entry_price": initial_price,
            "stats_1m": stats_1m,
            "model_output": model_output,
            "check_time": time.time() + 900 # Verify after 15 minutes
        }
        self.pending_events.append(event)
        return f"Decision logged: Outcome will be checked in 15m.", "info"

    async def check_outcomes(self, current_prices):
        """
        Checks pending events and generates ground truth data based on price movement.
        """
        completed = []
        now = time.time()

        for event in self.pending_events:
            # Skip if check time has not arrived
            if now < event['check_time']:
                continue

            pair = event['pair']
            if pair not in current_prices: continue

            exit_price = current_prices[pair]
            entry_price = event['entry_price']
            
            # Actual percentage change
            actual_change = ((exit_price - entry_price) / entry_price) * 100
            
            # --- LABELING LOGIC ---
            ideal_action = "HOLD"
            reason = "Price remained stable."
            
            if actual_change > 1.0: # If price pumped > 1% -> LONG was the ideal action
                ideal_action = "LONG"
                reason = f"Price pumped {actual_change:.2f}% in 15m."
            elif actual_change < -1.0: # If price dumped > 1% -> SHORT was the ideal action
                ideal_action = "SHORT"
                reason = f"Price dumped {actual_change:.2f}% in 15m."
            
            # Training data entry in Alpaca/Chat format
            training_entry = {
                "instruction": f"Analyze this crypto news for {pair}. Price is {entry_price}, 1m change is {event['stats_1m']}%. Return JSON.",
                "input": event['news'],
                "output": json.dumps({
                    "action": ideal_action,
                    "confidence": 100,
                    "reason": reason
                })
            }
            
            # Save only meaningful movements to avoid dataset bloat from HOLD cases
            if ideal_action != "HOLD":
                async with aiofiles.open(self.filename, mode='a', encoding='utf-8') as f:
                    await f.write(json.dumps(training_entry) + "\n")
                return f"TRAINING DATA SAVED: {pair.upper()} -> {ideal_action}", "success"
            
            completed.append(event)

        # Remove processed events
        for c in completed:
            self.pending_events.remove(c)