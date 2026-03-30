import numpy as np
import matplotlib.pyplot as plt

class NexusTechScoreGate:
    """
    Produces a 0-1 technical confidence score.
    This is NOT a signal generator.
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

    # -------------------------
    # Utility
    # -------------------------
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _clamp(self, x):
        return float(np.clip(x, 0.0, 1.0))

    # -------------------------
    # Factor Gates
    # -------------------------
    def rsi_gate(self, rsi, side):
    # Gaussian-like scoring: En yüksek puanı 55-60 civarında ver
        center = 55 if side == "LONG" else 45
        width = 15 # Bu genişlik puanın ne kadar yayılacağını belirler
        
        # Eşik değerine göre uzaklık
        diff = abs(rsi - center)
        score = np.exp(-(diff**2) / (2 * width**2))
        
        # Ekstra Veto: Sert Kesim
        if side == "LONG" and rsi > 75: score *= 0.1
        if side == "SHORT" and rsi < 25: score *= 0.1
        
        return score

    def funding_gate(self, funding, side):
        """
        Penalize crowded side only.
        """
        direction = 1 if side == "LONG" else -1
        pressure = direction * funding
        return self._sigmoid(
            -self.k_funding * (pressure - self.funding_limit)
        )

    def trend_gate(self, trend_strength):
        """
        trend_strength ∈ [-1, 1]
        """
        return self._sigmoid(self.k_trend * trend_strength)

    def volatility_gate(self, vol_z):
        """
        Penalize extreme volatility regimes.
        vol_z = ATR / rolling_ATR
        """
        return self._sigmoid(
            -self.k_vol * abs(vol_z - self.vol_target)
        )

    # -------------------------
    # Aggregation
    # -------------------------
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
        Returns 0-1 technical confidence score
        """

        if weights is None:
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

        # Logit-space aggregation (critical)
        logit = 0.0
        for k, w in weights.items():
            g = np.clip(gates[k], 1e-6, 1 - 1e-6)
            logit += w * np.log(g / (1 - g))

        score = self._sigmoid(logit)
        return self._clamp(score)

def run_standalone_test():
    gate = NexusTechScoreGate()
    

# Senaryo: RSI 20'den 80'e giderken, Trend sabitken LONG skoru nasıl değişiyor?
    rsi_values = np.linspace(20, 90, 100)
    scores = [gate.technical_score("LONG", rsi=r, funding=0.01, trend_strength=0.5, vol_z=1.0) for r in rsi_values]
    print(scores)
    plt.plot(rsi_values, scores)
    plt.xlabel("RSI")
    plt.ylabel("Technical Score")
    plt.title("Technical Score vs RSI")
    plt.show()

if __name__ == "__main__":
    run_standalone_test()