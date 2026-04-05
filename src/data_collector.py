import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    """
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}. Check ticker or date range.")

    # Flatten multi-level columns if present (yfinance returns MultiIndex for single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the stock data:
    - drop missing values
    - remove duplicate dates
    - warn if any price columns contain non-positive values
    """
    df = df.copy()

    # Drop missing rows
    df = df.dropna()

    # Remove duplicate index values (dates)
    df = df[~df.index.duplicated(keep="first")]

    # Check price columns
    price_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close"] if col in df.columns]
    for col in price_cols:
        if (df[col] <= 0).any():
            warnings.warn(f"Warning: Non-positive values found in column '{col}'")

    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily percentage return and log return columns.
    """
    df = df.copy()

    if "Close" not in df.columns:
        raise KeyError("Column 'Close' not found in DataFrame.")

    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def plot_price_and_returns(df: pd.DataFrame, symbol: str) -> None:
    """
    Plot closing price and daily returns.
    """
    if "Close" not in df.columns or "daily_return" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' and 'daily_return' columns.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df["Close"])
    axes[0].set_title(f"{symbol} Closing Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)

    axes[1].plot(df.index, df["daily_return"])
    axes[1].set_title(f"{symbol} Daily Returns")
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    save_path = "ml_predictor/data/AAPL_data.csv"

    # 1. Download
    df = download_stock_data(symbol, start_date, end_date)

    # 2. Clean
    df = clean_data(df)

    # 3. Add returns
    df = add_returns(df)

    # 4. Save
    save_data(df, save_path)

    # 5. Load back
    loaded_df = load_data(save_path)

    # 6. Print checks
    print("Shape:", loaded_df.shape)
    print("\nFirst 5 rows:")
    print(loaded_df.head())

    print("\nMissing values per column:")
    print(loaded_df.isnull().sum())

    # 7. Plot
    plot_price_and_returns(loaded_df, symbol)
