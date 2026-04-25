# Crypto Agent — Nexus AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI_Compatible_API-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kaggle_Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>LLM-powered crypto trading agent with fine-tuned sentiment analysis for pump & dump detection and LONG / SHORT / HOLD signal generation.</b>
</p>

---

## Overview

Crypto Agent (internally "Nexus AI") is a research project exploring whether language model-based reasoning can generate actionable trading signals from crypto news and social media streams. The system combines:

1. **Fine-tuned sentiment classification** — DeBERTa v3 trained on `nexus_elite_dataset_v5.json` (15,672 labeled crypto news samples) with LONG / SHORT / HOLD labels.
2. **LLM-based signal reasoning** — multiple large language models (Gemma 27B, Ministral 14B, and others) queried via OpenAI-compatible API (Ollama) to contextualize raw signals with market narrative.
3. **Pump & dump detection** — heuristic + model-based pattern recognition for anomalous volume/sentiment spikes.

The core thesis: **news-derived sentiment, when properly labeled and paired with LLM chain-of-thought reasoning, can approximate the qualitative judgment of an experienced trader** — without relying on price data alone.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                           │
│   Crypto News Feed  │  Twitter/Reddit Stream  │  OHLCV Data  │
└──────────┬───────────────────┬────────────────────┬──────────┘
           │                   │                    │
           ▼                   ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    SENTIMENT CLASSIFIER                       │
│         DeBERTa v3-large  (fine-tuned, SFT)                  │
│         Dataset: nexus_elite_dataset_v5.json                 │
│         Labels: LONG │ SHORT │ HOLD                          │
│         15,672 samples — published on Kaggle                 │
└──────────────────────┬───────────────────────────────────────┘
                       │  raw signal + confidence score
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     PUMP & DUMP DETECTOR                      │
│   Heuristic rules (volume spike, sentiment velocity)         │
│   + LLM anomaly flag via structured prompt                   │
└──────────────────────┬───────────────────────────────────────┘
                       │  enriched context bundle
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      LLM REASONING LAYER                      │
│   Primary:   Gemma 27B  ┐                                    │
│   Secondary: Ministral 14B ├─ via Ollama (OpenAI-compat API) │
│   Fallback:  Custom Nexus models (DeBERTa/DistilBERT)        │
└──────────────────────┬───────────────────────────────────────┘
                       │  final signal + rationale
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                            │
│         LONG │ SHORT │ HOLD  +  Confidence  +  Reasoning     │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tool / Library | Role |
|---|---|---|
| **Core Language** | Python 3.10+ | Entire pipeline |
| **Fine-tuning** | PyTorch + HuggingFace Transformers | DeBERTa v3 SFT |
| **Base Model (classifier)** | `microsoft/deberta-v3-large` | Sentiment classification |
| **LLM Inference** | Ollama (OpenAI-compatible API) | Local LLM serving |
| **LLM Models** | Gemma 27B, Ministral 14B | Signal reasoning |
| **Dataset** | `nexus_elite_dataset_v5.json` | 15,672 labeled samples |
| **Data Source** | CryptoPanic API, custom scrapers | News ingestion |
| **Experiment Tracking** | Weights & Biases (optional) | Loss/metric logging |
| **Dataset Hosting** | Kaggle | Public dataset release |

---

## Key Findings

| Model | Signal Collapse | Notes |
|---|---|---|
| Gemma 27B (via Ollama) | ❌ None | Best overall reasoning quality |
| Ministral 14B (via Ollama) | ❌ None | Strong, faster inference |
| Custom Nexus (DeBERTa fine-tune) | ✅ Collapsed to HOLD | SFT loss misaligned with sequence-level trading objective |
| Custom Nexus (DistilBERT fine-tune) | ✅ Collapsed to HOLD | Same root cause |

**Root cause of signal collapse:** Standard cross-entropy SFT loss optimizes token-level prediction, not the downstream trading objective. The custom models learned to hedge by outputting HOLD for all inputs, minimizing loss without learning discriminative signal. Fix direction: sequence-level reward modeling or direct preference optimization (DPO).

> The dataset (`nexus_elite_dataset_v5.json`) is publicly available on Kaggle: **[Crypto News to Action](https://www.kaggle.com/datasets/alpercanzer/crypto-news-to-action)**

---

## Project Structure

```
Crypto-Agent/
├── data/                         # Persistent storage, datasets & forensic logs
│   ├── backtest_results/         # Performance audit reports (.txt)
│   ├── nexus_elite_dataset_v5.json # Production-grade training dataset
│   └── crypto_agent_session      # Telegram session persistence
├── nexus/                        # Research & Machine Learning Infrastructure
│   ├── model.py                  # Dual-Core Transformer Architecture (NexusPredictor)
│   ├── technical_gate.py         # Logit-space Technical Veto Logic
│   ├── backtest_engine.py        # Forensic Simulation & Playback Engine
│   ├── forensic_miner.py         # High-speed RAM-based signal extraction
│   ├── train_backbone.py         # DeBERTa-v3 SFT pipeline
│   ├── train_llm_adapter.py      # LoRA fine-tuning for NEXUS-7 (Ministral/Gemma)
│   ├── distillation_engine.py    # Synthetic reasoning & data distillation
│   └── ...                       # Dataset audits & NLP utilities
├── src/                          # Real-time Orchestration & Execution
│   ├── main.py                   # Global system entry point
│   ├── brain.py                  # Neural & LLM reasoning orchestrator
│   ├── services.py               # Core business logic & signal routing
│   ├── binance_client.py         # Low-latency exchange execution
│   ├── exchange.py               # Paper-trading & PnL simulation
│   ├── dashboard.py              # Real-time monitoring interface
│   └── ...                       # Infrastructure & utilities
├── backtest_analysis.py          # Results aggregator & diagnostic tool
├── requirements.txt              # Dependency manifest
└── README.md                     # Project documentation
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/Jessitoii/crypto-agent.git
cd crypto-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file with your API_ID, API_HASH (Telegram) and Exchange keys
```

---

## Usage

### 1. Initialize Infrastructure
Before running the bot, establish a secure Telegram session:
```bash
python src/setup.py
```

### 2. Launch Production Agent
Starts the real-time news scraping and neural decision loop:
```bash
python src/main.py
```

### 3. Execute Forensic Backtest
Simulate the agent's performance against historical data caches:
```bash
python nexus/backtest_engine.py
```

### 4. Technical Performance Audit
Analyze the statistical outcomes of previous backtest runs:
```bash
python backtest_analysis.py
```

---

## Dataset

**nexus_elite_dataset_v5.json** — 15,672 manually curated and LLM-assisted labeled samples.

| Split | Size | Labels |
|---|---|---|
| Train | ~13,321 | LONG / SHORT / HOLD |
| Validation | ~2,351 | LONG / SHORT / HOLD |

📦 **Kaggle:** [Crypto News to Action](https://www.kaggle.com/datasets/alpercanzer/crypto-news-to-action)

---

## Capabilities & Limitations

- **Forensic Replay**: Includes a high-fidelity backtesting suite to validate signals before live deployment.
- **Neural Veto**: Technical gates (RSI/Funding) can override AI signals to prevent "top-buying".
- **Signals, not Advice**: The system generates quantitative signals; final risk remains with the operator.
- **Model Drift**: Fine-tuned models require periodic recalibration against changing market regimes.

---

## Roadmap

- [ ] Replace SFT with DPO for signal-level alignment
- [x] High-fidelity Forensic Backtesting suite
- [ ] Real-time CryptoPanic/NewsAPI integration
- [ ] Evaluate DeepSeek-R1 for chain-of-thought trading reasoning
- [ ] Multi-coin portfolio risk aggregation engine
- [x] RAM-cached market data mining for micro-latency research

---

## License

MIT License — see [LICENSE](LICENSE) for details.