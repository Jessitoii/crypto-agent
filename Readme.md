# Crypto Agent

LLM-based cryptocurrency trading agent that listens to Telegram news channels,
analyzes market sentiment, and generates LONG / SHORT / HOLD signals using
real-time Binance data.

This started as a simple news-to-signal pipeline and evolved into a full
research project — including a custom dual-head transformer model, multiple
fine-tuning attempts, and backtests across 10+ model configurations.

The labeled dataset collected during development is publicly available:
[Crypto News to Action — Kaggle](https://www.kaggle.com/datasets/alpercanzer/crypto-news-to-action)
(15,672 samples · LONG / SHORT / HOLD · Binance + Telegram)

---

## How it works

1. A message arrives from a Telegram channel
2. Regex checks if the mentioned coin is in the target list
3. If price data is missing in RAM, it's fetched from Binance API
4. The agent generates a search query and verifies the news via DuckDuckGo
5. LLM evaluates news + price momentum + research context → LONG / SHORT / HOLD
6. If confidence > 75%, the trade is executed on the paper exchange
7. The outcome is logged to `data/training_dataset.jsonl` for future fine-tuning

---

## Stack

- **LLM inference:** GroqCloud (Llama 3, Mixtral), Google Gemini, or Ollama (local)
- **Market data:** Binance WebSocket + REST API
- **News source:** Telegram via Telethon
- **Web research:** DuckDuckGo Search
- **Dashboard:** NiceGUI

---

## Project structure
```
crypto-agent/
├── src/                    # Trading bot
│   ├── main.py             # Orchestrator
│   ├── brain.py            # LLM logic: prompts, research, decision making
│   ├── services.py         # WebSocket, Telegram listeners
│   ├── dashboard.py        # NiceGUI dashboard
│   ├── exchange.py         # Paper trading engine
│   ├── binance_client.py   # Binance Futures API adapter
│   ├── price_buffer.py     # Recent candle buffer
│   ├── prompts.py          # Prompt templates
│   └── config.py           # Configuration
├── nexus/                  # Model training experiments
│   ├── train.py            # DeBERTa fine-tuning
│   ├── model.py            # Custom dual-head architecture (NexusV2)
│   ├── train_local_trader.py  # Unsloth + LoRA fine-tuning (Ministral-3B)
│   └── NexusTrain_Refactored.ipynb
├── data/
│   ├── nexus_elite_dataset_v5.json
│   └── backtest_results/   # Per-model backtest logs
├── system_prompt.txt
├── requirements.txt
└── .env.example
```

---

## Model experiments

The `nexus/` folder documents the full research arc — from standard fine-tuning
to a custom architecture. Short summary of what was tried and what happened:

| Model | Approach | Result |
|---|---|---|
| Gemma 27B | LLM agent, chain-of-thought | Best overall reasoning, TP/SL generation |
| Ministral 14B | LLM agent | Solid, comparable to Gemma |
| DeBERTa v3 (fine-tuned) | Standard classification | Worked, SHORT bias issue |
| NexusV2 (custom) | Dual-head DeBERTa, gate + direction | Collapsed to HOLD in inference |
| Nexus Phi / Qwen3 | Fine-tuned local models | Same HOLD collapse |
| SFT via Unsloth/LoRA | Ministral-3B instruction tuning | Incomplete |

The HOLD collapse in fine-tuned models was the main unresolved problem —
likely caused by token-level SFT loss being misaligned with sequence-level
trading objectives. The dataset was released as-is for further research.

---

## Setup

**Prerequisites:** Python 3.10+, Binance Futures account, Telegram API credentials
```bash
git clone https://github.com/Jessitoii/crypto-agent.git
cd crypto-agent
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:
```env
BINANCE_API_KEY_TESTNET=
BINANCE_API_SECRET_TESTNET=
BINANCE_API_KEY=
BINANCE_API_SECRET=

API_ID=
API_HASH=
TELETHON_SESSION_NAME=crypto_agent_session

GROQCLOUD_API_KEY=
GROQCLOUD_MODEL=llama3-70b-8192
```

Run:
```bash
python src/main.py
```

Dashboard: `http://localhost:8080`

---

## Disclaimer

This project is for research and educational purposes only.
Cryptocurrency trading involves significant financial risk.
Test on Binance Testnet before using real funds.