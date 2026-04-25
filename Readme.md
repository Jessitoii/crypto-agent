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

> The dataset (`nexus_elite_dataset_v5.json`) is publicly available on Kaggle: **[https://www.kaggle.com/datasets/alpercanzer/crypto-news-to-action]**

---

## Project Structure

```
crypto-agent/
├── data/
│   ├── nexus_elite_dataset_v5.json   # 15,672 labeled samples (LONG/SHORT/HOLD)
│   └── raw/                          # Raw scraped news
├── models/
│   ├── sentiment/                    # Fine-tuned DeBERTa checkpoints
│   └── prompts/                      # LLM system prompts & templates
├── src/
│   ├── agent.py                      # Main agent loop (signal generation)
│   ├── classifier.py                 # DeBERTa inference wrapper
│   ├── llm_client.py                 # Ollama / OpenAI-compat API client
│   ├── pump_dump_detector.py         # Anomaly detection logic
│   ├── data_pipeline.py              # News ingestion & preprocessing
│   └── labeler.py                    # Dataset labeling utilities
├── training/
│   ├── train_deberta.py              # SFT training script
│   ├── config.yaml                   # Training hyperparameters
│   └── evaluate.py                   # Eval metrics
├── notebooks/
│   └── analysis.ipynb                # Signal analysis & model comparison
├── requirements.txt
└── README.md
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

# 4. Set up Ollama and pull models
# Install Ollama: https://ollama.com
ollama pull gemma:27b
ollama pull mistral:14b-instruct

# 5. (Optional) Download the dataset
# Place nexus_elite_dataset_v5.json into data/
```

---

## Usage

### Run the agent (LLM reasoning mode)

```bash
# Uses Ollama locally — no API key required
python src/agent.py --model gemma:27b --input "Bitcoin ETF rejected by SEC, volume spike detected"
```

### Fine-tune the sentiment classifier

```bash
python training/train_deberta.py \
  --dataset data/nexus_elite_dataset_v5.json \
  --model microsoft/deberta-v3-large \
  --epochs 5 \
  --lr 2e-5
```

### Run evaluation

```bash
python training/evaluate.py --checkpoint models/sentiment/checkpoint-best
```

---

## Dataset

**nexus_elite_dataset_v5.json** — 15,672 manually curated and LLM-assisted labeled samples.

| Split | Size | Labels |
|---|---|---|
| Train | ~12,500 | LONG / SHORT / HOLD |
| Validation | ~1,600 | LONG / SHORT / HOLD |
| Test | ~1,572 | LONG / SHORT / HOLD |

Samples include crypto news headlines, social media snippets, and brief market commentary — each labeled with the appropriate trading signal given the context.

📦 **Kaggle:** [nexus-elite-dataset-v5](https://www.kaggle.com) ← update with actual link

---

## Limitations

- The system generates **signals, not financial advice**. No backtesting framework is included in this version.
- Custom fine-tuned models currently **collapse to HOLD** — the SFT objective does not align with trading signal quality. DPO or RLHF is the intended next step.
- LLM reasoning quality is **model-dependent**; smaller quantized models produce noisier outputs.

---

## Roadmap

- [ ] Replace SFT with DPO for signal-level alignment
- [ ] Add backtesting module (vectorbt / backtrader)
- [ ] Stream live CryptoPanic feed in real-time
- [ ] Evaluate Qwen2.5-14B and DeepSeek-R1 as reasoning layers
- [ ] Multi-coin portfolio signal aggregation

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://github.com/Jessitoii">Jessitoii</a> · Dataset on <a href="https://www.kaggle.com">Kaggle</a>
</p>