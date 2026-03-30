from collections import deque

class PriceBuffer:
    """
    Maintains a rolling window of minute-level price candles and calculates indicators.
    """
    def __init__(self):
        self.candles = deque(maxlen=60) 
        self.current_price = 0.0
        self.change_24h = 0.0

    def update_candle(self, price, timestamp, is_closed):
        """
        Updates the buffer with new candle data from websocket.
        If is_closed is True, the price is appended to the historical buffer.
        """
        self.current_price = price
        
        if is_closed:
            minute_ts = int(timestamp / 60)
            
            if not self.candles or self.candles[-1][0] != minute_ts:
                self.candles.append((minute_ts, price))

    def set_24h_change(self, percent):
        """Sets the 24-hour percentage change provided by external source."""
        self.change_24h = percent

    def get_change(self, minutes):
        """
        Calculates the percentage change over a specified number of minutes.
        """
        if not self.candles or self.current_price == 0:
            return 0.0
            
        if len(self.candles) < minutes:
            old_price = self.candles[0][1]
        else:
            old_price = self.candles[-minutes][1]
            
        if old_price == 0: return 0.0
        
        return ((self.current_price - old_price) / old_price) * 100
    
    def get_all_changes(self):
        """Returns percentage changes for multiple timeframes."""
        return {
            "1m": self.get_change(1),
            "10m": self.get_change(10),
            "1h": self.get_change(60),
            "24h": self.change_24h
        }

    def calculate_rsi(self, period=14):
        """Calculates RSI (Relative Strength Index) based on buffered candle data."""
        if len(self.candles) < period + 1: return 50.0
        
        closes = [c[1] for c in self.candles]
        
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))