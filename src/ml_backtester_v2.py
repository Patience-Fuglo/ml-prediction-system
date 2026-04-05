import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import train_random_forest, train_xgboost, predict
from .splitter import simple_split, separate_features_target
from .features_v2 import prepare_all_features_v2
from .data_collector import load_data
from .evaluation import rmse, directional_accuracy, simulated_sharpe
from .utils import (
    log, generate_signal, apply_slippage_and_spread,
    calculate_cagr, calculate_sortino_ratio, calculate_calmar_ratio,
    calculate_hit_ratio, calculate_avg_win_loss, calculate_profit_factor,
    calculate_exposure, calculate_turnover, save_dataframe,
)
from . import config


class MLBacktesterV2:
    """
    Enhanced backtester with:
    - Slippage and spread
    - Signal thresholds
    - Position sizing
    - Advanced metrics
    - Trade logging
    """
    
    def __init__(
        self,
        model,
        starting_cash=100000,
        fee_rate=0.001,
        slippage=0.0005,
        spread=0.0001,
        long_threshold=0.005,
        short_threshold=-0.005,
    ):
        self.model = model
        self.starting_cash = starting_cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.spread = spread
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        
        self.reset()
    
    def reset(self):
        self.cash = self.starting_cash
        self.shares = 0.0
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0
        self.total_trades = 0
        
        self.portfolio_history = []
        self.benchmark_history = []
        self.position_history = []
        self.trade_log = []
        self.daily_returns = []
    
    def run(self, X_test, y_test, prices):
        """Run backtest with enhanced logic."""
        self.reset()
        
        if not X_test.index.equals(prices.index):
            raise ValueError("X_test and prices must have matching indices.")
        
        predictions = predict(self.model, X_test)
        benchmark_start_price = float(prices.iloc[0])
        
        prev_portfolio_value = self.starting_cash
        entry_price = None
        
        for i, date in enumerate(X_test.index):
            pred = float(predictions[i])
            current_price = float(prices.loc[date])
            
            # Generate signal with thresholds
            signal = generate_signal(pred, self.long_threshold, self.short_threshold)
            
            # ENTRY: Go long
            if signal == 1 and self.position != 1:
                # Close short if any
                if self.position == -1:
                    self._close_position(date, current_price, "CLOSE SHORT")
                
                # Open long
                self._open_long(date, current_price, pred)
                entry_price = current_price
            
            # ENTRY: Go short (if implementing short selling)
            elif signal == -1 and self.position != -1:
                # Close long if any
                if self.position == 1:
                    self._close_position(date, current_price, "CLOSE LONG")
                
                # For now, just go flat (no short selling)
                self.position = 0
            
            # EXIT: Go flat
            elif signal == 0 and self.position != 0:
                self._close_position(date, current_price, "EXIT TO FLAT")
            
            # Calculate portfolio value
            if self.position == 1:
                portfolio_value = self.shares * current_price
            else:
                portfolio_value = self.cash
            
            # Track daily return
            if prev_portfolio_value > 0:
                daily_ret = portfolio_value / prev_portfolio_value - 1
                self.daily_returns.append(daily_ret)
            
            prev_portfolio_value = portfolio_value
            
            self.portfolio_history.append((date, portfolio_value))
            self.position_history.append((date, self.position))
            
            # Benchmark (buy and hold)
            benchmark_value = self.starting_cash * (current_price / benchmark_start_price)
            self.benchmark_history.append((date, benchmark_value))
        
        # Close any open position at end
        if self.position != 0:
            final_date = X_test.index[-1]
            final_price = float(prices.iloc[-1])
            self._close_position(final_date, final_price, "FINAL CLOSE")
            self.portfolio_history[-1] = (final_date, self.cash)
        
        self.results = self._calculate_metrics()
        return self.results
    
    def _open_long(self, date, price, pred):
        """Open long position with slippage."""
        adj_price = apply_slippage_and_spread(price, 1, self.slippage, self.spread)
        slippage_cost = self.cash * (self.slippage + self.spread / 2)
        
        fee = self.cash * self.fee_rate
        investable = self.cash - fee
        
        self.shares = investable / adj_price
        self.cash = 0.0
        self.position = 1
        
        self.total_fees_paid += fee
        self.total_slippage_cost += slippage_cost
        self.total_trades += 1
        
        self.trade_log.append({
            "date": date,
            "action": "BUY",
            "price": price,
            "adj_price": adj_price,
            "shares": self.shares,
            "fee": fee,
            "prediction": pred,
        })
    
    def _close_position(self, date, price, action_type):
        """Close position with slippage."""
        if self.position == 0:
            return
        
        adj_price = apply_slippage_and_spread(price, -1, self.slippage, self.spread)
        
        gross_value = self.shares * adj_price
        fee = gross_value * self.fee_rate
        slippage_cost = self.shares * price * (self.slippage + self.spread / 2)
        
        self.cash = gross_value - fee
        self.shares = 0.0
        self.position = 0
        
        self.total_fees_paid += fee
        self.total_slippage_cost += slippage_cost
        self.total_trades += 1
        
        self.trade_log.append({
            "date": date,
            "action": action_type,
            "price": price,
            "adj_price": adj_price,
            "shares": 0,
            "fee": fee,
            "prediction": None,
        })
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics."""
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        benchmark_df = pd.DataFrame(self.benchmark_history, columns=["Date", "Value"]).set_index("Date")
        position_df = pd.DataFrame(self.position_history, columns=["Date", "Position"]).set_index("Date")
        
        strategy_values = portfolio_df["Value"]
        benchmark_values = benchmark_df["Value"]
        
        n_days = len(strategy_values)
        
        # Basic returns
        total_return = strategy_values.iloc[-1] / strategy_values.iloc[0] - 1
        benchmark_return = benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1
        
        # CAGR
        cagr = calculate_cagr(total_return, n_days)
        benchmark_cagr = calculate_cagr(benchmark_return, n_days)
        
        # Daily returns
        daily_returns = np.array(self.daily_returns)
        benchmark_daily = benchmark_values.pct_change().dropna().values
        
        # Sharpe
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        benchmark_sharpe = np.mean(benchmark_daily) / np.std(benchmark_daily) * np.sqrt(252) if np.std(benchmark_daily) > 0 else 0
        
        # Sortino
        sortino = calculate_sortino_ratio(daily_returns)
        
        # Max drawdown
        running_max = strategy_values.cummax()
        drawdown = (strategy_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        benchmark_running_max = benchmark_values.cummax()
        benchmark_dd = (benchmark_values - benchmark_running_max) / benchmark_running_max
        benchmark_max_dd = benchmark_dd.min()
        
        # Calmar
        calmar = calculate_calmar_ratio(total_return, max_drawdown, n_days)
        
        # Hit ratio
        hit_ratio = calculate_hit_ratio(daily_returns)
        
        # Avg win/loss
        avg_win, avg_loss = calculate_avg_win_loss(daily_returns)
        
        # Profit factor
        profit_factor = calculate_profit_factor(daily_returns)
        
        # Exposure
        positions = position_df["Position"].values
        exposure = calculate_exposure(positions)
        
        # Turnover
        turnover = calculate_turnover(self.total_trades, n_days)
        
        # Round-trip trades
        round_trips = self.total_trades // 2
        
        return {
            "final_value": strategy_values.iloc[-1],
            "benchmark_final": benchmark_values.iloc[-1],
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "cagr": cagr,
            "benchmark_cagr": benchmark_cagr,
            "sharpe": sharpe,
            "benchmark_sharpe": benchmark_sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "benchmark_max_dd": benchmark_max_dd,
            "calmar": calmar,
            "hit_ratio_pct": hit_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "exposure_pct": exposure,
            "turnover": turnover,
            "total_trades": self.total_trades,
            "round_trips": round_trips,
            "total_fees": self.total_fees_paid,
            "total_slippage": self.total_slippage_cost,
            "n_days": n_days,
        }
    
    def print_report(self):
        """Print comprehensive backtest report."""
        r = self.results
        
        print("\n" + "=" * 70)
        print("ML BACKTEST REPORT V2 (Enhanced)")
        print("=" * 70)
        
        print(f"\n{'Metric':<30} {'Strategy':>15} {'Benchmark':>15}")
        print("-" * 62)
        print(f"{'Final Value':<30} ${r['final_value']:>14,.2f} ${r['benchmark_final']:>14,.2f}")
        print(f"{'Total Return':<30} {r['total_return']*100:>14.2f}% {r['benchmark_return']*100:>14.2f}%")
        print(f"{'CAGR':<30} {r['cagr']*100:>14.2f}% {r['benchmark_cagr']*100:>14.2f}%")
        print(f"{'Sharpe Ratio':<30} {r['sharpe']:>15.4f} {r['benchmark_sharpe']:>15.4f}")
        print(f"{'Sortino Ratio':<30} {r['sortino']:>15.4f} {'N/A':>15}")
        print(f"{'Max Drawdown':<30} {r['max_drawdown']*100:>14.2f}% {r['benchmark_max_dd']*100:>14.2f}%")
        print(f"{'Calmar Ratio':<30} {r['calmar']:>15.4f} {'N/A':>15}")
        
        print("\n" + "-" * 62)
        print("Trading Statistics")
        print("-" * 62)
        print(f"{'Hit Ratio':<30} {r['hit_ratio_pct']:>14.2f}%")
        print(f"{'Avg Win':<30} {r['avg_win']*100:>14.4f}%")
        print(f"{'Avg Loss':<30} {r['avg_loss']*100:>14.4f}%")
        print(f"{'Profit Factor':<30} {r['profit_factor']:>15.4f}")
        print(f"{'Market Exposure':<30} {r['exposure_pct']:>14.2f}%")
        print(f"{'Annualized Turnover':<30} {r['turnover']:>15.2f}")
        
        print("\n" + "-" * 62)
        print("Costs")
        print("-" * 62)
        print(f"{'Total Trades':<30} {r['total_trades']:>15}")
        print(f"{'Round-Trip Trades':<30} {r['round_trips']:>15}")
        print(f"{'Total Fees':<30} ${r['total_fees']:>14,.2f}")
        print(f"{'Total Slippage Cost':<30} ${r['total_slippage']:>14,.2f}")
        print(f"{'Total Trading Costs':<30} ${r['total_fees'] + r['total_slippage']:>14,.2f}")
        
        # Verdict
        print("\n" + "=" * 62)
        if r['sharpe'] > r['benchmark_sharpe']:
            print("✅ Strategy BEATS benchmark on risk-adjusted returns (Sharpe)")
        else:
            print("❌ Strategy UNDERPERFORMS benchmark on Sharpe")
        
        if r['max_drawdown'] > r['benchmark_max_dd']:
            print("✅ Strategy has LOWER drawdown than benchmark")
        else:
            print("⚠️  Strategy has HIGHER drawdown than benchmark")
    
    def plot_equity_curves(self, save_path=None):
        """Plot equity curves with enhanced visualization."""
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        benchmark_df = pd.DataFrame(self.benchmark_history, columns=["Date", "Value"]).set_index("Date")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curves
        axes[0].plot(portfolio_df.index, portfolio_df["Value"], label="ML Strategy", linewidth=2)
        axes[0].plot(benchmark_df.index, benchmark_df["Value"], label="Buy & Hold", linewidth=2, alpha=0.7)
        axes[0].set_title("Equity Curves: ML Strategy vs Buy & Hold", fontsize=14)
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        running_max = portfolio_df["Value"].cummax()
        drawdown = (portfolio_df["Value"] - running_max) / running_max * 100
        
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[1].set_title("Strategy Drawdown", fontsize=14)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log(f"Saved plot to {save_path}")
        
        plt.show()
    
    def save_trades(self, filepath):
        """Save trade log to CSV."""
        trade_df = pd.DataFrame(self.trade_log)
        save_dataframe(trade_df, filepath, "trades")
        return trade_df


if __name__ == "__main__":
    from models import train_random_forest
    
    # Load data
    df = load_data("data/AAPL_data.csv")
    df = prepare_all_features_v2(df, horizon=5)
    
    # Feature columns
    feature_cols = config.CORE_FEATURES
    
    model_df = df[feature_cols + ["target", "Close"]].copy()
    
    # Split
    train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    test_prices = test_df["Close"]
    
    # Train
    model = train_random_forest(X_train, y_train)
    
    # Backtest V2
    backtester = MLBacktesterV2(
        model=model,
        starting_cash=config.STARTING_CASH,
        fee_rate=config.FEE_RATE,
        slippage=config.SLIPPAGE,
        spread=config.SPREAD,
        long_threshold=config.LONG_THRESHOLD,
        short_threshold=config.SHORT_THRESHOLD,
    )
    
    backtester.run(X_test, y_test, test_prices)
    backtester.print_report()
    backtester.plot_equity_curves()
    
    # Save trades
    backtester.save_trades("reports/trades.csv")
