# 📈 Volatility Forecasting Research

This repository contains a research-grade pipeline to evaluate and compare volatility forecasting models, particularly on SPY and VIX market indices.

---

## 🧠 Introduction

Forecasting financial market volatility is a fundamental task in quantitative finance. Accurately estimating future volatility has direct applications in risk management, portfolio allocation, option pricing, and algorithmic trading. While traditional models like GARCH have been widely used for decades, recent advances in deep learning have opened the door to non-linear and data-driven forecasting approaches such as LSTMs and Temporal Fusion Transformers (TFT).

---

## 🎯 Research Objective

The main goal of this research is to:

- Benchmark the performance of GARCH(1,1) models against modern deep learning approaches (LSTM and TFT).
- Assess volatility prediction capability using only SPY and VIX OHLCV data.
- Explore model robustness through rolling-window forecasting and walk-forward validation.
- Study the effectiveness of volatility-based signals in real-time derivative strategy simulations (e.g., Vega arbitrage).

---

## 📂 Project Structure

```bash
forecasting_model/
├── data/
│   ├── raw/                # Unprocessed market data (SPY, VIX)
│   ├── processed/          # Cleaned and feature-engineered datasets
├── eda/
│   └── eda_analysis.ipynb  # Exploratory data analysis, volatility visualization
├── garch_model/
│   └── garch_forecast.ipynb # GARCH model training and evaluation
├── lstm_model/
│   └── lstm_forecast.ipynb # LSTM forecasting pipeline
├── tft_model/
│   └── tft_forecast.ipynb  # TFT forecasting pipeline
├── strategy/
│   ├── strategy_backtest.py # Vega arbitrage strategy logic
│   └── quantconnect_config/ # Configuration to test in QuantConnect
├── outputs/
│   ├── predictions/        # Forecast CSVs from each model
│   └── figures/            # Plots and evaluation visuals
├── wandb/                  # Experiment tracking logs
└── README.md
```

---

## 📌 Next Steps

- Finalize GARCH benchmark results ✅
- Implement LSTM walk-forward forecast 🔄
- Train TFT with covariates and encoder-decoder attention 🔄
- Benchmark strategy using QuantConnect framework 🔄

Stay tuned as we build out and validate a robust volatility forecasting framework.
