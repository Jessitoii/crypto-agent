"""
NEXUS Quantitative Scoring Gate (Technical Filter)

This module provides a robust technical validation layer that produces 
a 0-1 confidence score based on market microstructure metrics.

It is used to 'gate' LLM-based signals against technical realities 
(RSI extremes, Funding pressure, Momentum divergence) to prevent 
trading into exhausted trends.
"""

import numpy as np
import matplotlib.pyplot as plt

class NexusTechScoreGate:
    """
    Technical validation engine for signal verification.
    """
    def __init__(
        self,
        k_rsi=0.15,
        k_funding=80.0,
        k_trend=6.0,
        k_vol=4.0,
        rsi_neutral=50.0,
        funding_limit=0.08,
        vol_target=1.0
    ):
        self.k_rsi = k_rsi
        self.k_funding = k_funding
        self.k_trend = k_trend
        self.k_vol = k_vol

        self.rsi_neutral = rsi_neutral
        self.funding_limit = funding_limit
        self.vol_target = vol_target

    def _sigmoid(self, x):
        """Standard sigmoid activation for probability mapping."""
        return 1.0 / (1.0 + np.exp(-x))

    def _clamp(self, x):
        """Clamps values to the 0-1 probability range."""
        return float(np.clip(x, 0.0, 1.0))

    def rsi_gate(self, rsi, side):
        """
        Gaussian-weighted scoring centered around optimal entry zones.
        
        LONG Entry Target: 55-60 RSI.
        SHORT Entry Target: 40-45 RSI.
        
        Args:
            rsi (float): Current RSI (14) value.
            side (str): 'LONG' or 'SHORT'.
            
        Returns:
            float: 0-1 technical score for RSI.
        """
        center = 55 if side == "LONG" else 45
        width = 15 # Statistical variance of the score
        
        # Calculate distance-based activation
        diff = abs(rsi - center)
        score = np.exp(-(diff**2) / (2 * width**2))
        
        # Sectoral Veto: Penalize entries into extreme overbought/oversold exhaustion
        if side == "LONG" and rsi > 75: 
            score *= 0.1
        if side == "SHORT" and rsi < 25: 
            score *= 0.1
        
        return score

    def funding_gate(self, funding, side):
        """
        Scoring logic for funding-rate-based crowded trade detection.
        """
        direction = 1 if side == "LONG" else -1
        pressure = direction * funding
        return self._sigmoid(
            -self.k_funding * (pressure - self.funding_limit)
        )

    def trend_gate(self, trend_strength):
        """
        Maps normalized trend strength [-1, 1] to a technical confidence score.
        """
        return self._sigmoid(self.k_trend * trend_strength)

    def volatility_gate(self, vol_z):
        """
        Analyzes volatility Z-score to penalize abnormal ATR spikes.
        """
        return self._sigmoid(
            -self.k_vol * abs(vol_z - self.vol_target)
        )

    def technical_score(
        self,
        side,
        rsi,
        funding,
        trend_strength,
        vol_z,
        weights=None
    ):
        """
        Aggregates multi-factor technical gates into a unified logit-space score.
        
        Args:
            side (str): Trade direction.
            rsi (float): Asset RSI.
            funding (float): Futures funding rate.
            trend_strength (float): Directional momentum strength.
            vol_z (float): Volatility Z-score.
            weights (dict, optional): Custom weighting for feature aggregation.
            
        Returns:
            float: Final 0-1 technical confidence multiplier.
        """
        if weights is None:
            # Default weighting prioritizing RSI and Trend parity
            weights = {
                "rsi": 0.30,
                "funding": 0.25,
                "trend": 0.30,
                "vol": 0.15
            }

        gates = {
            "rsi": self.rsi_gate(rsi, side),
            "funding": self.funding_gate(funding, side),
            "trend": self.trend_gate(trend_strength),
            "vol": self.volatility_gate(vol_z)
        }

        # Logit-space aggregation ensures that a single 0-score gate 
        # heavily suppresses the total score (Veto effect).
        logit = 0.0
        for k, w in weights.items():
            g = np.clip(gates[k], 1e-6, 1 - 1e-6)
            logit += w * np.log(g / (1 - g))

        return self._clamp(self._sigmoid(logit))

def run_standalone_test():
    """Visualizes the technical scoring curve for diagnostic validation."""
    gate = NexusTechScoreGate()
    
    # Generate test range for RSI sensitivity analysis
    rsi_values = np.linspace(20, 90, 100)
    scores = [gate.technical_score("LONG", rsi=r, funding=0.01, trend_strength=0.5, vol_z=1.0) for r in rsi_values]
    
    plt.figure(figsize=(10, 5))
    plt.plot(rsi_values, scores, label="LONG Technical Confidence")
    plt.axvline(x=55, color='g', linestyle='--', label="Optimal LONG Zone")
    plt.xlabel("RSI")
    plt.ylabel("Technical Confidence Score")
    plt.title("NEXUS Technical Gate Logic Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_standalone_test()