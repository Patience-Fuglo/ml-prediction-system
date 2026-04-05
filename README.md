# 📈 Machine Learning Stock Prediction & Backtesting System

## Overview

This project implements an end-to-end **machine learning pipeline for financial markets**, designed to predict stock returns and evaluate trading strategies using real market data.

The system covers the full quantitative research workflow:

- Data collection
- Feature engineering
- Time-series safe model training
- Model evaluation with trading metrics
- Strategy backtesting
- Feature selection
- End-to-end pipeline automation

This project is designed to reflect **real-world quantitative research and trading system architecture**.

---

## Key Features

- 📊 **Real Market Data** via Yahoo Finance
- 🧠 **Feature Engineering**
  - Moving averages (SMA)
  - RSI (momentum)
  - MACD (trend/momentum)
  - Bollinger Bands (volatility)
  - Volume-based signals
  - Rolling volatility & momentum
  - ATR & return z-scores
- ⚠️ **Leakage-Free Time-Series Splitting**
- 🤖 **Machine Learning Models**
  - Linear Regression
  - Random Forest
  - XGBoost
- 📈 **Trading-Focused Evaluation**
  - RMSE / MAE
  - Directional Accuracy
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - CAGR
- 💼 **Realistic Backtesting Engine**
  - Strategy vs Buy-and-Hold comparison
  - Trade simulation with transaction costs
  - Slippage & bid-ask spread modeling
  - Signal thresholds (no-trade zone)
- 🔄 **Walk-Forward Validation**
  - Rolling window retraining
  - Out-of-sample testing
- 🌐 **Multi-Stock Analysis**
  - Per-stock models
  - Pooled cross-asset models
- 🧪 **Feature Selection**
  - Correlation filtering
  - Redundancy removal
  - Recursive Feature Elimination (RFE)
- 📉 **Equity Curve Visualization**

---

## Project Structure

```
ml-prediction-system/
│
├── data/                     # Raw & processed data
├── models_saved/             # Saved trained models
├── reports/                  # Evaluation & backtest outputs
├── plots/                    # Equity curves, charts
├── src/                      # Core logic modules
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_collector.py
│   ├── features.py
│   ├── features_v2.py
│   ├── splitter.py
│   ├── models.py
│   ├── evaluation.py
│   ├── ml_backtester.py
│   ├── ml_backtester_v2.py
│   ├── feature_selection.py
│   ├── walk_forward.py
│   ├── multi_stock.py
│   ├── pipeline.py
│   └── pipeline_v2.py
├── run_pipeline.py           # Main entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How It Works

### 1. Data Collection

Historical stock data is downloaded using Yahoo Finance.

### 2. Feature Engineering

Technical indicators are computed to capture:

- Trend (moving averages)
- Momentum (RSI, MACD)
- Volatility (Bollinger Bands, ATR)
- Volume dynamics

### 3. Train/Test Split

Time-series safe splitting ensures no data leakage.

### 4. Model Training

Models are trained to predict **future returns** over a defined horizon (default: 5 days).

### 5. Evaluation

Performance is measured using:

- Prediction error (RMSE, MAE)
- Directional accuracy
- Simulated Sharpe ratio
- Risk-adjusted metrics (Sortino, Calmar)

### 6. Backtesting

A trading strategy is simulated with realistic costs:

- Buy when prediction > threshold
- Sell when prediction < threshold
- Neutral zone for weak signals
- Includes transaction costs, slippage, spread

### 7. Walk-Forward Validation

Rolling window retraining for realistic out-of-sample testing.

### 8. Pipeline Execution

All steps are integrated into a reusable pipeline.

---

## Example Results

| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Total Return | +3.45% | +9.80% |
| Sharpe Ratio | **1.20** | 0.98 |
| Max Drawdown | **-3.07%** | -15.05% |
| Trading Costs | $1,556 | $0 |

- ✅ Higher risk-adjusted returns (Sharpe)
- ✅ Significantly lower drawdown
- Walk-forward directional accuracy: 54.55%

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Full Pipeline

```bash
python run_pipeline.py
```

### Options

```bash
# Single stock analysis
python run_pipeline.py --symbol AAPL

# Multi-stock analysis
python run_pipeline.py --multi

# Walk-forward validation
python run_pipeline.py --walk-forward

# Different model
python run_pipeline.py --model xgboost
```

---

## Example Usage

```python
from src.pipeline_v2 import PredictionPipelineV2

pipeline = PredictionPipelineV2(
    symbols=["AAPL"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    model_type="rf"
)

pipeline.setup()
pipeline.train()
pipeline.backtest()
pipeline.walk_forward_train()
pipeline.predict_latest()
```

---

## Key Insights

- Financial markets are difficult to predict
- Directional accuracy > 50% is meaningful
- Risk-adjusted returns (Sharpe) matter more than raw returns
- Lower drawdown = better capital preservation
- Walk-forward validation gives realistic performance estimates
- Per-stock models often outperform pooled models

---

## Future Improvements

- Portfolio optimization (mean-variance, risk parity)
- Alternative data sources
- Deep learning models (LSTM, Transformer)
- Live paper trading integration
- Options/derivatives modeling

---

## Author

**Patience Fuglo**  
Quantitative Finance & Machine Learning

---

## Disclaimer

This project is for educational and research purposes only.  
It does not constitute financial advice or a production trading system.
