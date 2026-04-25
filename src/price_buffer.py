"""
Real-time Price Buffer and Technical Indicators

This module implements a sliding-window data structure for managing 
high-frequency market data. It handles candle formation, price change
calculations across multiple timeframes, and technical indicator (RSI) 
computation.
"""

from collections import deque

class PriceBuffer:
    """
    Maintains a rolling window of minute-level price candles and calculates indicators.
    
    Attributes:
        candles (deque): Rolling buffer of (timestamp, price) tuples.
        current_price (float): Latest real-time price from WebSocket.
        change_24h (float): External 24-hour percentage change.
    """
    def __init__(self):
        """Initializes the price buffer with a 60-minute lookback capacity."""
        self.candles = deque(maxlen=60) 
        self.current_price = 0.0
        self.change_24h = 0.0

    def update_candle(self, price, timestamp, is_closed):
        """
        Updates the buffer with new candle data from a live stream.
        
        If a candle is marked as closed, it is appended to the historical 
        historical lookback buffer for technical analysis.
        
        Args:
            price (float): Current asset price.
            timestamp (float): Unix timestamp in seconds.
            is_closed (bool): True if the 1-minute candle interval has concluded.
        """
        self.current_price = price
        
        if is_closed:
            minute_ts = int(timestamp / 60)
            
            # Ensure only unique minute-aligned candles are stored
            if not self.candles or self.candles[-1][0] != minute_ts:
                self.candles.append((minute_ts, price))

    def set_24h_change(self, percent):
        """
        Sets the 24-hour percentage change provided by external market data sources.
        
        Args:
            percent (float): 24h percentage change.
        """
        self.change_24h = percent

    def get_change(self, minutes):
        """
        Calculates the percentage change over a specified number of minutes.
        
        Args:
            minutes (int): Lookback period in minutes.
            
        Returns:
            float: Percentage price change.
        """
        if not self.candles or self.current_price == 0:
            return 0.0
            
        # Fallback to earliest available data if buffer isn't full
        if len(self.candles) < minutes:
            old_price = self.candles[0][1]
        else:
            old_price = self.candles[-minutes][1]
            
        if old_price == 0: return 0.0
        
        return ((self.current_price - old_price) / old_price) * 100
    
    def get_all_changes(self):
        """
        Aggregates price delta metrics for multi-timeframe analysis.
        
        Returns:
            dict: Map of timeframes (1m, 10m, 1h, 24h) to percentage changes.
        """
        return {
            "1m": self.get_change(1),
            "10m": self.get_change(10),
            "1h": self.get_change(60),
            "24h": self.change_24h
        }

    def calculate_rsi(self, period=14):
        """
        Calculates the Relative Strength Index (RSI) from buffered data.
        
        Args:
            period (int): Period used for gain/loss averaging.
            
        Returns:
            float: RSI value [0, 100]. Returns 50.0 if insufficient data.
        """
        if len(self.candles) < period + 1: return 50.0
        
        closes = [c[1] for c in self.candles]
        
        # Calculate price deltas and segregate gains/losses
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        # Standard SMA-based RSI calculation
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
