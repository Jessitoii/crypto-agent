# Crypto Agent

LLM-based cryptocurrency trading agent that listens to Telegram news channels, 
analyzes market sentiment, and generates LONG / SHORT / HOLD signals using 
real-time Binance data.

Built as a research project. Accompanied by the 
[Crypto News to Action dataset](https://www.kaggle.com/datasets/alpercanzer/crypto-news-to-action) 
— 15,672 labeled samples collected during development.

---

## How it works

1. A message arrives from a Telegram channel (e.g. "Mugafi partners with AVAX")
2. Regex checks if the coin is in the target list
3. If price data is missing in RAM, it's fetched from Binance API
4. The agent generates a search query and verifies the news via DuckDuckGo
5. LLM evaluates news + price momentum + research context → LONG / SHORT / HOLD
6. If confidence > 75%, the trade is executed on the paper exchange
7. The outcome is logged to `data/training_dataset.jsonl` for future fine-tuning

---

## Stack

- **LLM inference:** GroqCloud (Llama 3, Mixtral) or Ollama (local)
- **Market data:** Binance WebSocket + REST API
- **News source:** Telegram via Telethon
- **Web research:** DuckDuckGo Search
- **Dashboard:** NiceGUI

---

## Project structure