"""
Pytest fixtures for ML Predictor test suite.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
    
    # Generate realistic stock prices using random walk
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = initial_price * np.cumprod(1 + returns)
    
    # Generate OHLCV data
    df = pd.DataFrame(index=dates)
    df["Close"] = prices
    df["Open"] = df["Close"].shift(1).fillna(initial_price) * (1 + np.random.uniform(-0.01, 0.01, len(dates)))
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(dates)))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(dates)))
    df["Volume"] = np.random.randint(1000000, 10000000, len(dates))
    df["Adj Close"] = df["Close"]
    
    return df


@pytest.fixture
def sample_feature_data(sample_stock_data):
    """Create sample data with features and target."""
    df = sample_stock_data.copy()
    
    # Add basic returns
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Add simple moving averages
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    
    # Add RSI
    delta = df["Close"].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.rolling(window=14).mean()
    avg_loss = losses.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Add MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Add Bollinger Bands position
    middle = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    bb_upper = middle + 2 * rolling_std
    bb_lower = middle - 2 * rolling_std
    band_width = bb_upper - bb_lower
    df["BB_position"] = (df["Close"] - bb_lower) / band_width
    
    # Add volume ratio
    df["volume_sma"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma"]
    
    # Add target
    df["target"] = df["Close"].shift(-5) / df["Close"] - 1
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


@pytest.fixture
def feature_columns():
    """Return the list of feature columns used for ML."""
    return [
        "daily_return",
        "log_return",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_signal",
        "BB_position",
        "volume_ratio",
    ]


@pytest.fixture
def sample_predictions():
    """Create sample predictions and actuals for evaluation testing."""
    np.random.seed(42)
    n = 100
    actual = np.random.normal(0, 0.02, n)
    # Predictions with some correlation to actual
    predicted = actual * 0.5 + np.random.normal(0, 0.015, n)
    return actual, predicted
