"""
Model Refinement Data Collector

This module implements an automated ground-truth labeling pipeline.
It captures AI decisions in real-time and verifies their outcome against 
actual price movements after a predefined lookback period (15 minutes).

Verified outcomes are saved in Alpaca/Chat JSONL format, creating a 
high-quality dataset for fine-tuning LLMs on specific crypto news alpha.
"""

import os
import aiofiles
import time 
import json

class TrainingDataCollector:
    """
    Orchestrates the collection and automated labeling of trading data.
    
    Attributes:
        filename (str): Target path for the JSONL dataset.
        pending_events (list): Queue of decisions awaiting outcome verification.
    """
    def __init__(self, filename="data_collection.jsonl"):
        """
        Initializes the collector.
        
        Args:
            filename (str): Name of the training dataset file.
        """
        self.filename = filename
        self.pending_events = [] 

    def log_decision(self, news, pair, initial_price, stats_1m, model_output):
        """
        Registers an AI decision for future outcome verification.
        
        Args:
            news (str): The news text that triggered the decision.
            pair (str): The asset ticker.
            initial_price (float): Price at the time of news arrival.
            stats_1m (float): 1-minute price change at decision time.
            model_output (dict): The raw JSON output from the AI model.
            
        Returns:
            tuple: (Status message, UI color)
        """
        event = {
            "timestamp": time.time(),
            "news": news,
            "pair": pair,
            "entry_price": initial_price,
            "stats_1m": stats_1m,
            "model_output": model_output,
            "check_time": time.time() + 900 # Scheduled verification (15m offset)
        }
        self.pending_events.append(event)
        return f"Decision captured: Ground-truth verification scheduled for +15m.", "info"

    async def check_outcomes(self, current_prices):
        """
        Evaluates pending decisions against current market prices.
        
        Iterates through the queue, calculates 15m price deltas, and assigns 
        ideal action labels (LONG/SHORT/HOLD) based on realized alpha.
        
        Args:
            current_prices (dict): Map of current ticker prices.
        """
        completed = []
        now = time.time()

        for event in self.pending_events:
            # Throttle processing until the evaluation window is reached
            if now < event['check_time']:
                continue

            pair = event['pair']
            if pair not in current_prices: continue

            exit_price = current_prices[pair]
            entry_price = event['entry_price']
            
            # Realized alpha percentage
            actual_change = ((exit_price - entry_price) / entry_price) * 100
            
            # --- AUTOMATED LABELING LOGIC ---
            ideal_action = "HOLD"
            reason = "Price remained stable within the lookback window."
            
            # Label as LONG if news resulted in a >1% rally
            if actual_change > 1.0: 
                ideal_action = "LONG"
                reason = f"Ideal capture: Price appreciated {actual_change:.2f}% in 15m."
            # Label as SHORT if news resulted in a >1% crash
            elif actual_change < -1.0: 
                ideal_action = "SHORT"
                reason = f"Ideal capture: Price depreciated {actual_change:.2f}% in 15m."
            
            # Construct Alpaca-style training entry for LLM supervised fine-tuning
            training_entry = {
                "instruction": f"Analyze this crypto news for {pair}. Price is {entry_price}, 1m change is {event['stats_1m']}%. Return JSON.",
                "input": event['news'],
                "output": json.dumps({
                    "action": ideal_action,
                    "confidence": 100,
                    "reason": reason
                })
            }
            
            # Optimization: Filter out 'HOLD' events to maintain high signal density in the dataset
            if ideal_action != "HOLD":
                async with aiofiles.open(self.filename, mode='a', encoding='utf-8') as f:
                    await f.write(json.dumps(training_entry) + "\n")
                print(f"[COLLECTOR] Dataset updated: {pair.upper()} ({ideal_action})")
            
            completed.append(event)

        # Batch cleanup of processed events to prevent memory leaks
        for c in completed:
            if c in self.pending_events:
                self.pending_events.remove(c)