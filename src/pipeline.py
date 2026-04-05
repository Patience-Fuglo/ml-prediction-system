import os
from datetime import datetime

import joblib
import pandas as pd

from .data_collector import download_stock_data, clean_data, add_returns
from .features import prepare_all_features
from .splitter import simple_split, separate_features_target
from .models import (
    train_linear,
    train_random_forest,
    train_xgboost,
    predict,
    get_feature_importance,
    print_top_features,
)
from .evaluation import evaluate_model
from .ml_backtester import MLBacktester


class PredictionPipeline:
    def __init__(self, symbol, model_type="xgboost"):
        self.symbol = symbol.upper()
        self.model_type = model_type.lower()

        self.raw_data = None
        self.feature_data = None

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.feature_columns = None
        self.model = None
        self.horizon = None
        self.start_date = None
        self.end_date = None

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def setup(self, start_date, end_date, horizon=5):
        self.log(f"Setting up pipeline for {self.symbol} from {start_date} to {end_date}...")

        self.start_date = start_date
        self.end_date = end_date
        self.horizon = horizon

        df = download_stock_data(self.symbol, start_date, end_date)
        df = clean_data(df)
        df = add_returns(df)

        self.raw_data = df.copy()

        feature_df = prepare_all_features(df, horizon=horizon)

        # Best practical feature set from your earlier work
        self.feature_columns = [
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

        required_columns = self.feature_columns + ["target", "Close"]
        feature_df = feature_df[required_columns].copy()

        self.feature_data = feature_df

        self.train_df, self.val_df, self.test_df = simple_split(
            self.feature_data, train_pct=0.7, val_pct=0.15
        )

        self.X_train = self.train_df[self.feature_columns].copy()
        self.y_train = self.train_df["target"].copy()

        self.X_val = self.val_df[self.feature_columns].copy()
        self.y_val = self.val_df["target"].copy()

        self.X_test = self.test_df[self.feature_columns].copy()
        self.y_test = self.test_df["target"].copy()

        self.log("Setup complete.")
        print("\n=== PIPELINE SETUP SUMMARY ===")
        print(f"Symbol: {self.symbol}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Horizon: {horizon} days")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Train samples: {len(self.train_df)}")
        print(f"Validation samples: {len(self.val_df)}")
        print(f"Test samples: {len(self.test_df)}")

    def train(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run setup() before train().")

        self.log(f"Training {self.model_type} model on {len(self.X_train)} samples...")

        if self.model_type == "linear":
            self.model = train_linear(self.X_train, self.y_train)
        elif self.model_type == "rf":
            self.model = train_random_forest(self.X_train, self.y_train)
        elif self.model_type == "xgboost":
            self.model = train_xgboost(self.X_train, self.y_train)
        else:
            raise ValueError("model_type must be 'linear', 'rf', or 'xgboost'")

        print("\n=== FEATURE IMPORTANCE ===")
        importances = get_feature_importance(self.model, self.feature_columns)
        print_top_features(importances, n=min(10, len(importances)))

        self.log("Evaluating model on validation set...")
        val_preds = predict(self.model, self.X_val)
        evaluate_model(self.y_val, val_preds, f"{self.symbol} {self.model_type.upper()} Validation")

    def backtest(self):
        if self.model is None:
            raise ValueError("Train or load a model before backtest().")

        self.log(f"Running backtest for {self.symbol} on test set...")

        test_prices = self.test_df["Close"].copy()

        backtester = MLBacktester(
            model=self.model,
            starting_cash=100000,
            fee_rate=0.001
        )

        backtester.run(self.X_test, self.y_test, test_prices)
        backtester.print_report()
        backtester.plot_equity_curves()

    def predict_latest(self):
        if self.model is None:
            raise ValueError("Train or load a model before predict_latest().")
        if self.feature_data is None:
            raise ValueError("Run setup() before predict_latest().")

        self.log(f"Making latest prediction for {self.symbol}...")

        latest_X = self.feature_data[self.feature_columns].iloc[[-1]]
        latest_date = latest_X.index[0]

        pred = float(predict(self.model, latest_X)[0])

        recommendation = "BUY" if pred > 0 else "SELL"

        print("\n=== LATEST PREDICTION ===")
        print(f"Symbol: {self.symbol}")
        print(f"Date: {latest_date}")
        print(f"Predicted {self.horizon}-day return: {pred:.6f}")
        print(f"Recommendation: {recommendation}")

        return pred, recommendation

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        self.log(f"Saving model to {filepath}...")

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "symbol": self.symbol,
            "model_type": self.model_type,
            "horizon": self.horizon,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

        joblib.dump(payload, filepath)
        self.log("Model saved successfully.")

    def load_model(self, filepath):
        self.log(f"Loading model from {filepath}...")

        payload = joblib.load(filepath)

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.symbol = payload["symbol"]
        self.model_type = payload["model_type"]
        self.horizon = payload.get("horizon")
        self.start_date = payload.get("start_date")
        self.end_date = payload.get("end_date")

        self.log("Model loaded successfully.")


if __name__ == "__main__":
    # ------------------------------
    # Full end-to-end workflow
    # ------------------------------
    pipeline = PredictionPipeline(symbol="AAPL", model_type="xgboost")

    pipeline.setup(start_date="2022-01-01", end_date="2024-01-01", horizon=5)
    pipeline.train()
    pipeline.backtest()
    pipeline.save_model("aapl_model.pkl")

    # ------------------------------
    # Load model into a new pipeline
    # ------------------------------
    new_pipeline = PredictionPipeline(symbol="AAPL", model_type="xgboost")
    new_pipeline.setup(start_date="2022-01-01", end_date="2024-01-01", horizon=5)
    new_pipeline.load_model("aapl_model.pkl")
    new_pipeline.predict_latest()
