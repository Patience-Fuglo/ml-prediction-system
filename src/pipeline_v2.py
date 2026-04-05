import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .data_collector import download_stock_data, clean_data, add_returns, save_data, load_data
from .features_v2 import prepare_all_features_v2
from .splitter import simple_split
from .models import train_linear, train_random_forest, train_xgboost, predict, get_feature_importance
from .evaluation import rmse, mae, directional_accuracy, simulated_sharpe
from .ml_backtester_v2 import MLBacktesterV2
from .walk_forward import walk_forward_backtest
from .feature_selection import correlation_with_target
from .utils import log, ensure_directories, save_dataframe
from . import config


class PredictionPipelineV2:
    """
    Hedge-fund level ML prediction pipeline (V2).
    
    Features:
    - Enhanced feature engineering
    - Walk-forward retraining
    - Multi-stock support
    - Realistic backtesting (slippage, thresholds)
    - Comprehensive metrics
    - Full output saving
    """
    
    def __init__(
        self,
        symbols=None,
        start_date=None,
        end_date=None,
        horizon=None,
        model_type="rf",
    ):
        self.symbols = symbols or [config.DEFAULT_SYMBOL]
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        self.horizon = horizon or config.HORIZON
        self.model_type = model_type
        
        self.feature_cols = config.CORE_FEATURES
        self.data = {}
        self.models = {}
        self.results = {}
        
        ensure_directories()
    
    def setup(self, download=True):
        """Download and prepare data for all symbols."""
        log(f"Setting up pipeline for {len(self.symbols)} symbol(s)")
        
        for symbol in self.symbols:
            try:
                log(f"Processing {symbol}...")
                
                if download:
                    df = download_stock_data(symbol, self.start_date, self.end_date)
                    df = clean_data(df)
                    df = add_returns(df)
                    save_data(df, f"{config.DATA_DIR}/{symbol}_data.csv")
                else:
                    df = load_data(f"{config.DATA_DIR}/{symbol}_data.csv")
                
                # Feature engineering
                df = prepare_all_features_v2(df, horizon=self.horizon)
                save_data(df, f"{config.DATA_DIR}/{symbol}_features_v2.csv")
                
                self.data[symbol] = df
                
            except Exception as e:
                log(f"Error processing {symbol}: {e}")
                continue
        
        log(f"Setup complete: {len(self.data)} symbols ready")
        return self
    
    def train(self, symbol=None):
        """Train model for a specific symbol or all symbols."""
        symbols_to_train = [symbol] if symbol else self.symbols
        
        for sym in symbols_to_train:
            if sym not in self.data:
                log(f"No data for {sym}, skipping")
                continue
            
            log(f"Training model for {sym}...")
            
            df = self.data[sym]
            model_df = df[self.feature_cols + ["target", "Close"]].copy()
            
            # Split
            train_df, val_df, test_df = simple_split(
                model_df, 
                train_pct=config.TRAIN_PCT, 
                val_pct=config.VAL_PCT
            )
            
            X_train = train_df[self.feature_cols]
            y_train = train_df["target"]
            X_val = val_df[self.feature_cols]
            y_val = val_df["target"]
            
            # Train
            if self.model_type == "linear":
                model = train_linear(X_train, y_train)
            elif self.model_type == "rf":
                model = train_random_forest(X_train, y_train)
            else:
                model = train_xgboost(X_train, y_train)
            
            # Validate
            val_preds = predict(model, X_val)
            val_rmse = rmse(y_val, val_preds)
            val_da = directional_accuracy(y_val, val_preds)
            
            self.models[sym] = model
            
            log(f"{sym} trained: Val RMSE={val_rmse:.6f}, Val DA={val_da:.2f}%")
            
            # Save model
            model_path = f"{config.MODELS_DIR}/{sym}_{self.model_type}_model.pkl"
            joblib.dump(model, model_path)
            log(f"Saved model to {model_path}")
        
        return self
    
    def walk_forward_train(self, symbol=None):
        """Run walk-forward retraining for a symbol."""
        sym = symbol or self.symbols[0]
        
        if sym not in self.data:
            log(f"No data for {sym}")
            return None
        
        log(f"Running walk-forward for {sym}...")
        
        df = self.data[sym]
        model_df = df[self.feature_cols + ["target"]].copy()
        
        results = walk_forward_backtest(
            df=model_df,
            feature_cols=self.feature_cols,
            train_size=config.WF_TRAIN_SIZE,
            test_size=config.WF_TEST_SIZE,
            model_type=self.model_type,
        )
        
        self.results[f"{sym}_walk_forward"] = results
        
        # Save
        results["predictions_df"].to_csv(f"{config.REPORTS_DIR}/{sym}_wf_predictions.csv")
        results["fold_results"].to_csv(f"{config.REPORTS_DIR}/{sym}_wf_folds.csv")
        
        return results
    
    def backtest(self, symbol=None):
        """Run enhanced backtest for a symbol."""
        sym = symbol or self.symbols[0]
        
        if sym not in self.models:
            log(f"No model for {sym}, training first...")
            self.train(sym)
        
        df = self.data[sym]
        model = self.models[sym]
        
        model_df = df[self.feature_cols + ["target", "Close"]].copy()
        train_df, val_df, test_df = simple_split(
            model_df, 
            train_pct=config.TRAIN_PCT, 
            val_pct=config.VAL_PCT
        )
        
        X_test = test_df[self.feature_cols]
        y_test = test_df["target"]
        test_prices = test_df["Close"]
        
        log(f"Running backtest for {sym}...")
        
        backtester = MLBacktesterV2(
            model=model,
            starting_cash=config.STARTING_CASH,
            fee_rate=config.FEE_RATE,
            slippage=config.SLIPPAGE,
            spread=config.SPREAD,
            long_threshold=config.LONG_THRESHOLD,
            short_threshold=config.SHORT_THRESHOLD,
        )
        
        bt_results = backtester.run(X_test, y_test, test_prices)
        backtester.print_report()
        
        # Save outputs
        backtester.save_trades(f"{config.REPORTS_DIR}/{sym}_trades.csv")
        backtester.plot_equity_curves(f"{config.PLOTS_DIR}/{sym}_equity.png")
        
        self.results[f"{sym}_backtest"] = bt_results
        
        return backtester, bt_results
    
    def feature_analysis(self, symbol=None):
        """Analyze feature importance and correlations."""
        sym = symbol or self.symbols[0]
        
        if sym not in self.data:
            return None
        
        df = self.data[sym]
        
        log(f"Feature analysis for {sym}...")
        
        # Correlations
        X = df[self.feature_cols]
        y = df["target"]
        corr = correlation_with_target(X, y)
        print("\nFeature correlations with target:")
        print(corr.to_string())
        
        # RFE (simplified)
        if sym in self.models:
            from sklearn.feature_selection import RFE
            model_for_rfe = train_random_forest(X, y)
            selector = RFE(estimator=model_for_rfe, n_features_to_select=8, step=1)
            selector.fit(X, y)
            selected = X.columns[selector.support_].tolist()
            print(f"\nRFE selected features: {selected}")
        
        # Model importance
        if sym in self.models:
            importances = get_feature_importance(self.models[sym], self.feature_cols)
            
            print("\nModel feature importance:")
            for i, (feat, imp) in enumerate(importances[:10], 1):
                print(f"{i:>2}. {feat:<20} {imp:.6f}")
            
            # Save
            imp_df = pd.DataFrame(importances, columns=["feature", "importance"])
            imp_df.to_csv(f"{config.REPORTS_DIR}/{sym}_feature_importance.csv", index=False)
        
        return corr, selected
    
    def predict_latest(self, symbol=None):
        """Generate prediction for latest data point."""
        sym = symbol or self.symbols[0]
        
        if sym not in self.models or sym not in self.data:
            log(f"Model or data not ready for {sym}")
            return None
        
        df = self.data[sym]
        model = self.models[sym]
        
        latest = df[self.feature_cols].iloc[-1:].copy()
        pred = predict(model, latest)[0]
        
        # Generate signal
        if pred > config.LONG_THRESHOLD:
            signal = "BUY"
        elif pred < config.SHORT_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD (neutral zone)"
        
        print(f"\n=== LATEST PREDICTION FOR {sym} ===")
        print(f"Date: {df.index[-1]}")
        print(f"Predicted {self.horizon}-day return: {pred*100:.2f}%")
        print(f"Signal: {signal}")
        print(f"Thresholds: LONG > {config.LONG_THRESHOLD*100:.1f}%, SHORT < {config.SHORT_THRESHOLD*100:.1f}%")
        
        return pred, signal
    
    def generate_report(self, symbol=None):
        """Generate comprehensive performance report."""
        sym = symbol or self.symbols[0]
        
        report_data = []
        
        # Collect all metrics
        if f"{sym}_backtest" in self.results:
            bt = self.results[f"{sym}_backtest"]
            report_data.append(["BACKTEST RESULTS", ""])
            report_data.append(["Total Return", f"{bt['total_return']*100:.2f}%"])
            report_data.append(["CAGR", f"{bt['cagr']*100:.2f}%"])
            report_data.append(["Sharpe Ratio", f"{bt['sharpe']:.4f}"])
            report_data.append(["Sortino Ratio", f"{bt['sortino']:.4f}"])
            report_data.append(["Max Drawdown", f"{bt['max_drawdown']*100:.2f}%"])
            report_data.append(["Calmar Ratio", f"{bt['calmar']:.4f}"])
            report_data.append(["Hit Ratio", f"{bt['hit_ratio_pct']:.2f}%"])
            report_data.append(["Profit Factor", f"{bt['profit_factor']:.4f}"])
            report_data.append(["Exposure", f"{bt['exposure_pct']:.2f}%"])
            report_data.append(["Total Trades", f"{bt['total_trades']}"])
            report_data.append(["Total Costs", f"${bt['total_fees'] + bt['total_slippage']:.2f}"])
        
        if f"{sym}_walk_forward" in self.results:
            wf = self.results[f"{sym}_walk_forward"]
            report_data.append(["", ""])
            report_data.append(["WALK-FORWARD RESULTS", ""])
            report_data.append(["Overall RMSE", f"{wf['overall_rmse']:.6f}"])
            report_data.append(["Overall Dir Accuracy", f"{wf['overall_da']:.2f}%"])
            report_data.append(["Overall Sharpe", f"{wf['overall_sharpe']:.4f}"])
        
        report_df = pd.DataFrame(report_data, columns=["Metric", "Value"])
        report_df.to_csv(f"{config.REPORTS_DIR}/{sym}_report.csv", index=False)
        log(f"Report saved to {config.REPORTS_DIR}/{sym}_report.csv")
        
        return report_df
    
    def run_full_pipeline(self, symbol=None):
        """Run complete pipeline: setup, train, backtest, analyze."""
        sym = symbol or self.symbols[0]
        
        print("\n" + "=" * 80)
        print(f"RUNNING FULL PIPELINE FOR {sym}")
        print("=" * 80)
        
        # 1. Setup (download if needed)
        if sym not in self.data:
            self.setup(download=True)
        
        # 2. Train
        self.train(sym)
        
        # 3. Backtest
        self.backtest(sym)
        
        # 4. Walk-forward
        self.walk_forward_train(sym)
        
        # 5. Feature analysis
        self.feature_analysis(sym)
        
        # 6. Generate report
        self.generate_report(sym)
        
        # 7. Latest prediction
        self.predict_latest(sym)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nOutputs saved to:")
        print(f"  - {config.DATA_DIR}/")
        print(f"  - {config.MODELS_DIR}/")
        print(f"  - {config.REPORTS_DIR}/")
        print(f"  - {config.PLOTS_DIR}/")
        
        return self


if __name__ == "__main__":
    # Run full pipeline for single stock
    pipeline = PredictionPipelineV2(
        symbols=["AAPL"],
        start_date="2020-01-01",
        end_date="2024-01-01",
        horizon=5,
        model_type="rf",
    )
    
    pipeline.run_full_pipeline("AAPL")
