import os
import pandas as pd
import numpy as np

from .data_collector import download_stock_data, clean_data, add_returns, save_data
from .features_v2 import prepare_all_features_v2
from .splitter import simple_split
from .models import train_random_forest, train_xgboost, predict, get_feature_importance
from .evaluation import rmse, directional_accuracy, simulated_sharpe
from .ml_backtester_v2 import MLBacktesterV2
from .utils import log, ensure_directories, save_dataframe
from . import config


def run_multi_stock_analysis(
    symbols,
    start_date,
    end_date,
    feature_cols,
    model_type="rf",
):
    """
    Run analysis across multiple stocks.
    
    Returns per-stock results and pooled model results.
    """
    log(f"Starting multi-stock analysis for {len(symbols)} symbols")
    ensure_directories()
    
    all_data = {}
    per_stock_results = []
    
    # ========================
    # 1. Download and process each stock
    # ========================
    log("Downloading and processing data...")
    
    for symbol in symbols:
        try:
            log(f"Processing {symbol}...")
            
            # Download
            df = download_stock_data(symbol, start_date, end_date)
            df = clean_data(df)
            df = add_returns(df)
            
            # Features
            df = prepare_all_features_v2(df, horizon=config.HORIZON)
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Save
            save_data(df, f"data/{symbol}_features_v2.csv")
            
            all_data[symbol] = df
            
        except Exception as e:
            log(f"Error processing {symbol}: {e}")
            continue
    
    log(f"Successfully processed {len(all_data)} stocks")
    
    # ========================
    # 2. Per-stock models
    # ========================
    log("\nTraining per-stock models...")
    
    for symbol, df in all_data.items():
        try:
            model_df = df[feature_cols + ["target", "Close"]].copy()
            
            train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)
            
            X_train = train_df[feature_cols]
            y_train = train_df["target"]
            X_test = test_df[feature_cols]
            y_test = test_df["target"]
            test_prices = test_df["Close"]
            
            # Train
            if model_type == "rf":
                model = train_random_forest(X_train, y_train)
            else:
                model = train_xgboost(X_train, y_train)
            
            # Predict
            preds = predict(model, X_test)
            
            # Metrics
            stock_rmse = rmse(y_test, preds)
            stock_da = directional_accuracy(y_test, preds)
            stock_sharpe = simulated_sharpe(y_test, preds)
            
            # Backtest
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
            
            per_stock_results.append({
                "symbol": symbol,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "rmse": stock_rmse,
                "directional_accuracy": stock_da,
                "sharpe": stock_sharpe,
                "bt_return": bt_results["total_return"],
                "bt_sharpe": bt_results["sharpe"],
                "bt_max_dd": bt_results["max_drawdown"],
                "bt_trades": bt_results["total_trades"],
            })
            
        except Exception as e:
            log(f"Error training {symbol}: {e}")
            continue
    
    results_df = pd.DataFrame(per_stock_results)
    
    print("\n" + "=" * 80)
    print("PER-STOCK MODEL RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # ========================
    # 3. Pooled model (all stocks together)
    # ========================
    log("\nTraining pooled model across all stocks...")
    
    # Combine all data
    combined_df = pd.concat(all_data.values(), ignore_index=False)
    combined_df = combined_df.sort_index()
    
    log(f"Combined dataset: {len(combined_df)} rows")
    
    model_df = combined_df[feature_cols + ["target", "Close", "symbol"]].copy()
    
    # Time-based split
    train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    
    # Train pooled model
    if model_type == "rf":
        pooled_model = train_random_forest(X_train, y_train)
    else:
        pooled_model = train_xgboost(X_train, y_train)
    
    # Predict
    pooled_preds = predict(pooled_model, X_test)
    
    # Metrics
    pooled_rmse = rmse(y_test, pooled_preds)
    pooled_da = directional_accuracy(y_test, pooled_preds)
    pooled_sharpe = simulated_sharpe(y_test, pooled_preds)
    
    print("\n" + "=" * 80)
    print("POOLED MODEL RESULTS (All Stocks Combined)")
    print("=" * 80)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"RMSE: {pooled_rmse:.6f}")
    print(f"Directional Accuracy: {pooled_da:.2f}%")
    print(f"Simulated Sharpe: {pooled_sharpe:.4f}")
    
    # Feature importance
    print("\n=== POOLED MODEL FEATURE IMPORTANCE ===")
    importances = get_feature_importance(pooled_model, feature_cols)
    for i, (feat, imp) in enumerate(importances[:10], 1):
        print(f"{i:>2}. {feat:<20} {imp:.6f}")
    
    # ========================
    # 4. Compare per-stock vs pooled
    # ========================
    print("\n" + "=" * 80)
    print("COMPARISON: Per-Stock vs Pooled")
    print("=" * 80)
    
    avg_per_stock_da = results_df["directional_accuracy"].mean()
    avg_per_stock_sharpe = results_df["sharpe"].mean()
    
    print(f"{'Metric':<30} {'Per-Stock (avg)':>15} {'Pooled':>15}")
    print("-" * 62)
    print(f"{'Directional Accuracy':<30} {avg_per_stock_da:>14.2f}% {pooled_da:>14.2f}%")
    print(f"{'Simulated Sharpe':<30} {avg_per_stock_sharpe:>15.4f} {pooled_sharpe:>15.4f}")
    
    if pooled_da > avg_per_stock_da:
        print("\n✅ Pooled model has BETTER directional accuracy")
    else:
        print("\n❌ Per-stock models have better directional accuracy on average")
    
    # Save results
    results_df.to_csv("reports/multi_stock_results.csv", index=False)
    log("Saved multi-stock results to reports/multi_stock_results.csv")
    
    return {
        "per_stock_results": results_df,
        "pooled_model": pooled_model,
        "pooled_metrics": {
            "rmse": pooled_rmse,
            "directional_accuracy": pooled_da,
            "sharpe": pooled_sharpe,
        },
        "all_data": all_data,
    }


if __name__ == "__main__":
    ensure_directories()
    
    # Use subset for testing (full list in config.SYMBOLS)
    test_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    
    results = run_multi_stock_analysis(
        symbols=test_symbols,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        feature_cols=config.CORE_FEATURES,
        model_type=config.MODEL_TYPE,
    )
