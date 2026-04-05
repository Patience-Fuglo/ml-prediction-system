# Machine Learning Stock Prediction & Backtesting System

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

An end-to-end **machine learning pipeline for financial markets** that predicts stock returns and evaluates trading strategies using real market data.

---

## 🎯 Overview

This project implements core quantitative machine learning techniques for financial prediction:

- **Data Collection** — Historical stock data via Yahoo Finance
- **Feature Engineering** — Technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR)
- **Time-Series Safe Splitting** — No data leakage, walk-forward validation
- **ML Models** — Linear Regression, Random Forest, XGBoost
- **Trading Evaluation** — Sharpe, Sortino, Calmar, Directional Accuracy
- **Realistic Backtesting** — Transaction costs, slippage, spread modeling
- **Pipeline Automation** — End-to-end reproducible workflow

---

## 📁 Project Structure

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
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_data_collector.py
│   ├── test_features.py
│   ├── test_splitter.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_backtester.py
│   └── test_pipeline.py
├── run_pipeline.py           # Main entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Patience-Fuglo/ml-predictor.git
cd ml-predictor

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python run_pipeline.py
```

This will:
1. Download historical stock data for AAPL
2. Engineer technical features (SMA, RSI, MACD, etc.)
3. Train a Random Forest model
4. Evaluate with trading metrics
5. Run backtesting simulation
6. Save results to `reports/` and `plots/`

---

## 📊 Key Features

| Feature | Description |
|---------|-------------|
| **Real Market Data** | Historical OHLCV data via Yahoo Finance |
| **Feature Engineering** | SMA, RSI, MACD, Bollinger Bands, ATR, Volume ratios |
| **ML Models** | Linear Regression, Random Forest, XGBoost |
| **Time-Series Splitting** | Leakage-free train/val/test splits |
| **Walk-Forward Validation** | Rolling window retraining |
| **Backtesting Engine** | Transaction costs, slippage, spread |
| **Multi-Stock Analysis** | Per-stock and pooled cross-asset models |
| **Feature Selection** | Correlation filtering, RFE |

### Models

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE / MAE | Prediction error |
| Directional Accuracy | % of correct up/down predictions |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | Return / Max Drawdown |
| CAGR | Compound Annual Growth Rate |

---

## ⚙️ How It Works

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

## 📈 Sample Results

### Backtest Performance

| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Total Return | +3.45% | +9.80% |
| Sharpe Ratio | **1.20** | 0.98 |
| Max Drawdown | **-3.07%** | -15.05% |
| Trading Costs | $1,556 | $0 |

### Model Comparison

| Model | RMSE | Directional Accuracy | Sharpe |
|-------|------|---------------------|--------|
| Linear Regression | 0.0312 | 51.2% | 0.85 |
| Random Forest | 0.0298 | 53.8% | **1.20** |
| XGBoost | 0.0305 | 52.5% | 1.05 |

### Key Findings

- Higher risk-adjusted returns (Sharpe)
- Significantly lower drawdown
- Walk-forward directional accuracy: 54.55%

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests with pytest
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src --cov-report=term-missing
```

### Run Specific Test Modules

```bash
# Test data collection
pytest tests/test_data_collector.py -v

# Test feature engineering
pytest tests/test_features.py -v

# Test models
pytest tests/test_models.py -v

# Test backtester
pytest tests/test_backtester.py -v

# Test pipeline
pytest tests/test_pipeline.py -v
```

### Quick Validation (Integration Tests)

```bash
# Quick pipeline sanity check
python run_pipeline.py --symbol AAPL --model rf

# Validate walk-forward logic
python run_pipeline.py --symbol AAPL --walk-forward

# Test multi-stock mode
python run_pipeline.py --multi
```

---

## 🖥️ Run the Full Pipeline

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

## 💻 Example Usage

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

## 🔧 Configuration

### Pipeline Parameters

```python
pipeline = PredictionPipelineV2(
    symbols=["AAPL", "MSFT"],   # Stock symbols
    start_date="2020-01-01",    # Data start date
    end_date="2024-01-01",      # Data end date
    model_type="rf",            # "linear", "rf", "xgboost"
)
```

### Backtester Parameters

```python
backtester = MLBacktester(
    model=model,
    starting_cash=100000,      # Initial capital
    fee_rate=0.001,            # 0.1% per trade
    slippage=0.0005,           # 0.05% slippage
    spread=0.0001,             # 0.01% bid-ask spread
)
```

### Walk-Forward Parameters

```python
# Walk-forward settings in config.py
WF_TRAIN_SIZE = 252            # ~1 year training window
WF_TEST_SIZE = 21              # ~1 month test window
```

---

## 📚 Technical Details

### Prediction Target

The model predicts **future returns** over a horizon of N days:

```
target = (Close[t+N] / Close[t]) - 1
```

### Feature Engineering

| Feature | Formula | Description |
|---------|---------|-------------|
| SMA | `mean(Close, window)` | Simple Moving Average |
| RSI | `100 - 100/(1 + RS)` | Relative Strength Index |
| MACD | `EMA_12 - EMA_26` | Moving Average Convergence Divergence |
| BB Position | `(Close - BB_lower) / (BB_upper - BB_lower)` | Bollinger Band position |
| ATR | `mean(TR, 14)` | Average True Range |

### Trading Strategy

```
if prediction > long_threshold:
    BUY (go long)
elif prediction < short_threshold:
    SELL (close position)
else:
    HOLD (no action - avoid weak signals)
```

### Sharpe Ratio

```
Sharpe = (mean(returns) / std(returns)) × √252
```

Annualized risk-adjusted return measure.

---

## 🛠️ Technologies Used

| Category | Technologies/Concepts |
|----------|----------------------|
| Machine Learning | Regression, Random Forest, XGBoost, Feature Engineering |
| Quantitative Finance | Technical Analysis, Backtesting, Walk-Forward Validation |
| Data Analysis | Time Series, Statistical Modeling, Performance Metrics |
| Software Engineering | Modular Design, Type Hints, Testing, Documentation |
| Python Libraries | NumPy, Pandas, Scikit-learn, XGBoost, yfinance, Matplotlib |

---

## 🧠 Key Insights

- Financial markets are difficult to predict
- Directional accuracy > 50% is meaningful
- Risk-adjusted returns (Sharpe) matter more than raw returns
- Lower drawdown = better capital preservation
- Walk-forward validation gives realistic performance estimates
- Per-stock models often outperform pooled models

---

## 📝 Future Enhancements

- [ ] Portfolio optimization (mean-variance, risk parity)
- [ ] Alternative data sources (sentiment, fundamentals)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Live paper trading integration
- [ ] Options/derivatives modeling
- [ ] Interactive dashboard (Streamlit/Dash)
- [ ] Feature importance visualization
- [ ] Hyperparameter optimization (Optuna)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Patience Fuglo**  
Quantitative Finance & Machine Learning

- GitHub: [@Patience-Fuglo](https://github.com/Patience-Fuglo)
- LinkedIn: [Patience Fuglo](https://linkedin.com/in/patience-fuglo)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.  
It does not constitute financial advice or a production trading system.

---

⭐ **Star this repo if you find it useful!**