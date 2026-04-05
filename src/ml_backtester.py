import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import train_linear, train_random_forest, train_xgboost, predict
from .splitter import simple_split, separate_features_target
from .features import prepare_all_features
from .data_collector import load_data


class MLBacktester:
    def __init__(self, model, starting_cash=100000, fee_rate=0.001):
        self.model = model
        self.starting_cash = starting_cash
        self.fee_rate = fee_rate

        self.reset()

    def reset(self):
        self.cash = self.starting_cash
        self.shares = 0.0
        self.in_position = False

        self.total_fees_paid = 0.0
        self.total_trades = 0

        self.portfolio_history = []
        self.benchmark_history = []

        self.trade_log = []
        self.closed_trade_returns = []

    def run(self, X_test, y_test, prices):
        """
        Backtest logic:
        - if predicted > 0 and in cash: buy full position
        - if predicted <= 0 and holding: sell full position
        - otherwise hold
        """
        self.reset()

        if not X_test.index.equals(prices.index):
            raise ValueError("X_test and prices must have matching indices.")

        predictions = predict(self.model, X_test)

        benchmark_start_price = float(prices.iloc[0])

        entry_value_after_fee = None

        for i, date in enumerate(X_test.index):
            pred = float(predictions[i])
            current_price = float(prices.loc[date])

            # BUY
            if pred > 0 and not self.in_position:
                buy_fee = self.cash * self.fee_rate
                investable_cash = self.cash - buy_fee

                if investable_cash > 0:
                    self.shares = investable_cash / current_price
                    self.cash = 0.0
                    self.in_position = True

                    self.total_fees_paid += buy_fee
                    self.total_trades += 1
                    entry_value_after_fee = investable_cash

                    self.trade_log.append({
                        "date": date,
                        "action": "BUY",
                        "price": current_price,
                        "fee": buy_fee
                    })

            # SELL
            elif pred <= 0 and self.in_position:
                gross_sale_value = self.shares * current_price
                sell_fee = gross_sale_value * self.fee_rate
                net_sale_value = gross_sale_value - sell_fee

                self.cash = net_sale_value
                self.shares = 0.0
                self.in_position = False

                self.total_fees_paid += sell_fee
                self.total_trades += 1

                self.trade_log.append({
                    "date": date,
                    "action": "SELL",
                    "price": current_price,
                    "fee": sell_fee
                })

                if entry_value_after_fee is not None and entry_value_after_fee > 0:
                    trade_return = (net_sale_value / entry_value_after_fee) - 1
                    self.closed_trade_returns.append(trade_return)

                entry_value_after_fee = None

            # Portfolio value
            if self.in_position:
                portfolio_value = self.shares * current_price
            else:
                portfolio_value = self.cash

            self.portfolio_history.append((date, portfolio_value))

            # Buy-and-hold benchmark
            benchmark_value = self.starting_cash * (current_price / benchmark_start_price)
            self.benchmark_history.append((date, benchmark_value))

        # Liquidate at end if still holding
        if self.in_position:
            final_date = X_test.index[-1]
            final_price = float(prices.iloc[-1])

            gross_sale_value = self.shares * final_price
            sell_fee = gross_sale_value * self.fee_rate
            net_sale_value = gross_sale_value - sell_fee

            self.cash = net_sale_value
            self.shares = 0.0
            self.in_position = False

            self.total_fees_paid += sell_fee
            self.total_trades += 1

            self.trade_log.append({
                "date": final_date,
                "action": "FINAL SELL",
                "price": final_price,
                "fee": sell_fee
            })

            if entry_value_after_fee is not None and entry_value_after_fee > 0:
                trade_return = (net_sale_value / entry_value_after_fee) - 1
                self.closed_trade_returns.append(trade_return)

            self.portfolio_history[-1] = (final_date, self.cash)

        self.results = self.calculate_metrics()
        return self.results

    def calculate_metrics(self):
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        benchmark_df = pd.DataFrame(self.benchmark_history, columns=["Date", "Value"]).set_index("Date")

        strategy_values = portfolio_df["Value"]
        benchmark_values = benchmark_df["Value"]

        strategy_returns = strategy_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()

        total_return = strategy_values.iloc[-1] / strategy_values.iloc[0] - 1
        benchmark_total_return = benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1

        n_days = len(strategy_values)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
        benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0

        strategy_std = strategy_returns.std()
        benchmark_std = benchmark_returns.std()

        sharpe = (strategy_returns.mean() / strategy_std * np.sqrt(252)) if strategy_std != 0 else 0.0
        benchmark_sharpe = (benchmark_returns.mean() / benchmark_std * np.sqrt(252)) if benchmark_std != 0 else 0.0

        max_drawdown = self.compute_max_drawdown(strategy_values)
        benchmark_max_drawdown = self.compute_max_drawdown(benchmark_values)

        num_round_trip_trades = len(self.closed_trade_returns)
        win_rate = (
            np.mean(np.array(self.closed_trade_returns) > 0) * 100
            if num_round_trip_trades > 0 else 0.0
        )

        return {
            "strategy_total_return": total_return,
            "strategy_annualized_return": annualized_return,
            "strategy_sharpe": sharpe,
            "strategy_max_drawdown": max_drawdown,
            "benchmark_total_return": benchmark_total_return,
            "benchmark_annualized_return": benchmark_annualized_return,
            "benchmark_sharpe": benchmark_sharpe,
            "benchmark_max_drawdown": benchmark_max_drawdown,
            "total_trades": self.total_trades,
            "round_trip_trades": num_round_trip_trades,
            "win_rate": win_rate,
            "total_fees_paid": self.total_fees_paid,
            "final_portfolio_value": strategy_values.iloc[-1],
            "final_benchmark_value": benchmark_values.iloc[-1],
        }

    @staticmethod
    def compute_max_drawdown(values):
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max
        return drawdown.min()

    def print_report(self):
        r = self.results

        print("\n=== ML BACKTEST REPORT ===")
        print(f"{'Metric':<28} {'ML Strategy':>15} {'Buy & Hold':>15}")
        print("-" * 62)
        print(f"{'Final Value':<28} {r['final_portfolio_value']:>15.2f} {r['final_benchmark_value']:>15.2f}")
        print(f"{'Total Return':<28} {r['strategy_total_return']*100:>14.2f}% {r['benchmark_total_return']*100:>14.2f}%")
        print(f"{'Annualized Return':<28} {r['strategy_annualized_return']*100:>14.2f}% {r['benchmark_annualized_return']*100:>14.2f}%")
        print(f"{'Sharpe Ratio':<28} {r['strategy_sharpe']:>15.4f} {r['benchmark_sharpe']:>15.4f}")
        print(f"{'Max Drawdown':<28} {r['strategy_max_drawdown']*100:>14.2f}% {r['benchmark_max_drawdown']*100:>14.2f}%")
        print("-" * 62)
        print(f"{'Total Trades':<28} {r['total_trades']:>15}")
        print(f"{'Round-Trip Trades':<28} {r['round_trip_trades']:>15}")
        print(f"{'Win Rate':<28} {r['win_rate']:>14.2f}%")
        print(f"{'Total Fees Paid':<28} {r['total_fees_paid']:>15.2f}")

    def plot_equity_curves(self):
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        benchmark_df = pd.DataFrame(self.benchmark_history, columns=["Date", "Value"]).set_index("Date")

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df["Value"], label="ML Strategy")
        plt.plot(benchmark_df.index, benchmark_df["Value"], label="Buy & Hold")
        plt.title("Equity Curve: ML Strategy vs Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load base data
    df = load_data("data/AAPL_data.csv")

    # Build features
    df = prepare_all_features(df, horizon=5)

    selected_features = [
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

    model_df = df[selected_features + ["target", "Close"]].copy()

    # Split into train / val / test
    train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)

    # Train on train only, evaluate on test
    train_model_df = train_df.copy()
    test_model_df = test_df.copy()

    X_train = train_model_df.drop(columns=["target", "Close"])
    y_train = train_model_df["target"]

    X_test = test_model_df.drop(columns=["target", "Close"])
    y_test = test_model_df["target"]
    test_prices = test_model_df["Close"]

    # Best model from ML-5
    best_model = train_random_forest(X_train, y_train)

    # Backtest
    backtester = MLBacktester(
        model=best_model,
        starting_cash=100000,
        fee_rate=0.001
    )

    backtester.run(X_test, y_test, test_prices)
    backtester.print_report()
    backtester.plot_equity_curves()
