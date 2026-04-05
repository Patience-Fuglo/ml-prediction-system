import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, windows=[5, 10, 20, 50]) -> pd.DataFrame:
    """
    Add simple moving averages for the Close price.
    """
    df = df.copy()

    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()

    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI).
    RSI above 70 = overbought
    RSI below 30 = oversold
    """
    df = df.copy()

    delta = df["Close"].diff()

    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD and MACD signal line.
    """
    df = df.copy()

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Bollinger Bands and band position.
    """
    df = df.copy()

    middle = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()

    df["BB_middle"] = middle
    df["BB_upper"] = middle + 2 * rolling_std
    df["BB_lower"] = middle - 2 * rolling_std

    band_width = df["BB_upper"] - df["BB_lower"]
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / band_width

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features.
    """
    df = df.copy()

    df["volume_sma"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma"]

    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add prediction target:
    future return over the next 'horizon' days.
    """
    df = df.copy()

    df["target"] = df["Close"].shift(-horizon) / df["Close"] - 1

    return df


def prepare_all_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Run all feature engineering steps and remove NaN rows.
    """
    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_target(df, horizon=horizon)

    df = df.dropna().copy()

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nFinal shape:")
    print(df.shape)

    return df


if __name__ == "__main__":
    from data_collector import load_data

    filepath = "data/AAPL_data.csv"

    df = load_data(filepath)
    df = prepare_all_features(df, horizon=5)

    # Focused feature list for ML
    feature_cols = [
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
        "volume_ratio"
    ]

    print("\nFeature columns for ML:")
    print(feature_cols)

    print("\nNaN values per column:")
    print(df.isnull().sum())

    print("\nFirst 5 rows:")
    print(df.head())

    # Save engineered features
    df.to_csv("data/AAPL_features.csv")
    print("\nSaved features to data/AAPL_features.csv")
