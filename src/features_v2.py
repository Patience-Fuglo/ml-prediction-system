import numpy as np
import pandas as pd

from .features import (
    add_moving_averages,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_volume_features,
    add_target,
)


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling volatility (standard deviation of returns)."""
    df = df.copy()
    df[f"vol_{window}"] = df["daily_return"].rolling(window=window).std()
    return df


def add_momentum(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
    """Add momentum over multiple horizons."""
    df = df.copy()
    for p in periods:
        df[f"mom_{p}"] = df["Close"] / df["Close"].shift(p) - 1
    return df


def add_price_to_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add price relative to SMA (deviation from mean)."""
    df = df.copy()
    sma_col = f"SMA_{window}"
    if sma_col not in df.columns:
        df[sma_col] = df["Close"].rolling(window=window).mean()
    df[f"price_to_sma{window}"] = df["Close"] / df[sma_col] - 1
    return df


def add_macd_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD histogram (MACD - Signal)."""
    df = df.copy()
    if "MACD" not in df.columns or "MACD_signal" not in df.columns:
        df = add_macd(df)
    df["macd_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (volatility indicator)."""
    df = df.copy()
    
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=period).mean()
    
    return df


def add_return_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add z-score of daily returns (standardized returns)."""
    df = df.copy()
    
    rolling_mean = df["daily_return"].rolling(window=window).mean()
    rolling_std = df["daily_return"].rolling(window=window).std()
    
    df["return_zscore"] = (df["daily_return"] - rolling_mean) / rolling_std
    
    return df


def add_rolling_max_drawdown(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling maximum drawdown over window."""
    df = df.copy()
    
    rolling_max = df["Close"].rolling(window=window).max()
    drawdown = (df["Close"] - rolling_max) / rolling_max
    df["rolling_max_dd"] = drawdown.rolling(window=window).min()
    
    return df


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add day of week as categorical feature."""
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    return df


def add_classification_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add classification target (up/down).
    
    target_cls: 1 if future return > 0, else 0
    target_cls_3: 0 (down), 1 (neutral), 2 (up)
    """
    df = df.copy()
    
    if "target" not in df.columns:
        df = add_target(df, horizon=horizon)
    
    # Binary classification
    df["target_cls"] = (df["target"] > 0).astype(int)
    
    # 3-class classification (avoid NaN conversion issues)
    df["target_cls_3"] = 1  # Default neutral
    df.loc[df["target"] > 0.01, "target_cls_3"] = 2  # Up
    df.loc[df["target"] < -0.01, "target_cls_3"] = 0  # Down
    
    return df


def prepare_all_features_v2(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Enhanced feature engineering (V2) with all advanced features.
    """
    df = df.copy()
    
    # Original features
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    
    # New V2 features
    df = add_volatility(df, window=20)
    df = add_momentum(df, periods=[5, 10, 20])
    df = add_price_to_sma(df, window=20)
    df = add_macd_histogram(df)
    df = add_atr(df, period=14)
    df = add_return_zscore(df, window=20)
    df = add_rolling_max_drawdown(df, window=20)
    
    # Target
    df = add_target(df, horizon=horizon)
    df = add_classification_target(df, horizon=horizon)
    
    # Drop NaN rows
    df = df.dropna().copy()
    
    print("\nV2 Final columns:")
    print(df.columns.tolist())
    print(f"\nV2 Final shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    from data_collector import load_data
    
    df = load_data("data/AAPL_data.csv")
    df = prepare_all_features_v2(df, horizon=5)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nNew V2 features:")
    v2_features = ["vol_20", "mom_5", "mom_10", "mom_20", "price_to_sma20", 
                   "macd_hist", "ATR", "return_zscore", "rolling_max_dd",
                   "target_cls", "target_cls_3"]
    print(df[v2_features].head())
    
    # Save enhanced features
    df.to_csv("data/AAPL_features_v2.csv")
    print("\nSaved to data/AAPL_features_v2.csv")
