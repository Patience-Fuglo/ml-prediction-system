#!/usr/bin/env python3
"""
ML Stock Predictor - Main Entry Point

This script runs the complete machine learning prediction pipeline.

Usage:
    python run_pipeline.py                     # Run single stock (AAPL)
    python run_pipeline.py --multi             # Run multi-stock analysis
    python run_pipeline.py --symbol MSFT       # Run for specific symbol
    python run_pipeline.py --walk-forward      # Run walk-forward only
    python run_pipeline.py --model xgboost     # Use XGBoost model
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import log, ensure_directories
from src import config


def run_single_stock(symbol="AAPL", model_type="rf"):
    """Run full pipeline for a single stock."""
    from src.pipeline_v2 import PredictionPipelineV2
    
    log(f"Starting single-stock pipeline for {symbol}")
    
    pipeline = PredictionPipelineV2(
        symbols=[symbol],
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        horizon=config.HORIZON,
        model_type=model_type,
    )
    
    pipeline.run_full_pipeline(symbol)
    return pipeline


def run_multi_stock(model_type="rf"):
    """Run multi-stock analysis."""
    from src.multi_stock import run_multi_stock_analysis
    
    log("Starting multi-stock analysis")
    
    results = run_multi_stock_analysis(
        symbols=config.SYMBOLS[:5],
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        feature_cols=config.CORE_FEATURES,
        model_type=model_type,
    )
    
    return results


def run_walk_forward(symbol="AAPL", model_type="rf"):
    """Run walk-forward backtest only."""
    from src.walk_forward import walk_forward_backtest
    from src.data_collector import load_data
    from src.features_v2 import prepare_all_features_v2
    
    log(f"Running walk-forward for {symbol}")
    
    df = load_data(f"data/{symbol}_data.csv")
    df = prepare_all_features_v2(df, horizon=config.HORIZON)
    
    model_df = df[config.CORE_FEATURES + ["target"]].copy()
    
    results = walk_forward_backtest(
        df=model_df,
        feature_cols=config.CORE_FEATURES,
        train_size=config.WF_TRAIN_SIZE,
        test_size=config.WF_TEST_SIZE,
        model_type=model_type,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ML Stock Prediction & Backtesting System"
    )
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="AAPL",
        help="Stock symbol (default: AAPL)"
    )
    parser.add_argument(
        "--multi", 
        action="store_true",
        help="Run multi-stock analysis"
    )
    parser.add_argument(
        "--walk-forward", 
        action="store_true",
        help="Run walk-forward backtest only"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="rf",
        choices=["linear", "rf", "xgboost"],
        help="Model type (default: rf)"
    )
    
    args = parser.parse_args()
    
    ensure_directories()
    
    print("\n" + "=" * 70)
    print("📈 ML STOCK PREDICTION & BACKTESTING SYSTEM")
    print("=" * 70)
    print(f"Model: {args.model.upper()}")
    print(f"Horizon: {config.HORIZON} days")
    print(f"Date range: {config.START_DATE} to {config.END_DATE}")
    print("=" * 70 + "\n")
    
    if args.multi:
        results = run_multi_stock(args.model)
    elif args.walk_forward:
        results = run_walk_forward(args.symbol, args.model)
    else:
        results = run_single_stock(args.symbol, args.model)
    
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
