import os
import numpy as np
import pandas as pd
from datetime import datetime


def ensure_directories():
    """Create output directories if they don't exist."""
    dirs = ["data", "models_saved", "reports", "plots"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def calculate_cagr(total_return, n_days):
    """Calculate Compound Annual Growth Rate."""
    if n_days <= 0:
        return 0.0
    years = n_days / 252
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Sortino Ratio: like Sharpe but only penalizes downside volatility.
    """
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate / 252
    
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_std * np.sqrt(252)


def calculate_calmar_ratio(total_return, max_drawdown, n_days):
    """
    Calmar Ratio: CAGR / |Max Drawdown|
    """
    cagr = calculate_cagr(total_return, n_days)
    if max_drawdown == 0:
        return np.inf
    return cagr / abs(max_drawdown)


def calculate_hit_ratio(returns):
    """Percentage of positive returns."""
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    return np.mean(returns > 0) * 100


def calculate_avg_win_loss(returns):
    """Calculate average win and average loss."""
    returns = np.asarray(returns)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    return avg_win, avg_loss


def calculate_profit_factor(returns):
    """Gross profit / Gross loss."""
    returns = np.asarray(returns)
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return np.inf
    return gross_profit / gross_loss


def calculate_exposure(positions):
    """Percentage of time in market."""
    positions = np.asarray(positions)
    return np.mean(positions != 0) * 100


def calculate_turnover(trades, n_days):
    """Annualized turnover rate."""
    if n_days <= 0:
        return 0.0
    return trades / n_days * 252


def save_dataframe(df, filepath, description="data"):
    """Save DataFrame to CSV with logging."""
    df.to_csv(filepath)
    log(f"Saved {description} to {filepath}")


def generate_signal(pred, long_threshold=0.005, short_threshold=-0.005):
    """
    Generate trading signal with thresholds.
    
    Returns:
        1: Long
        -1: Short
        0: No position (neutral zone)
    """
    if pred > long_threshold:
        return 1
    elif pred < short_threshold:
        return -1
    else:
        return 0


def apply_slippage_and_spread(price, direction, slippage=0.0005, spread=0.0001):
    """
    Adjust price for slippage and bid-ask spread.
    
    direction: 1 for buy, -1 for sell
    """
    total_cost = slippage + spread / 2
    if direction == 1:  # Buying
        return price * (1 + total_cost)
    else:  # Selling
        return price * (1 - total_cost)


def print_advanced_metrics(results_dict):
    """Print comprehensive trading metrics."""
    print("\n=== ADVANCED METRICS ===")
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    
    for key, value in results_dict.items():
        if isinstance(value, float):
            if "ratio" in key.lower() or "sharpe" in key.lower():
                print(f"{key:<25} {value:>15.4f}")
            elif "pct" in key.lower() or "rate" in key.lower():
                print(f"{key:<25} {value:>14.2f}%")
            else:
                print(f"{key:<25} {value:>15.2f}")
        else:
            print(f"{key:<25} {value:>15}")
