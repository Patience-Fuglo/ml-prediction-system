import numpy as np
import pandas as pd

from .models import train_random_forest, train_xgboost, predict
from .splitter import separate_features_target
from .features_v2 import prepare_all_features_v2
from .data_collector import load_data
from .evaluation import rmse, directional_accuracy, simulated_sharpe
from .utils import log
from . import config


def walk_forward_backtest(
    df,
    feature_cols,
    train_size=252,
    test_size=21,
    model_type="rf",
):
    """
    Walk-forward backtest with rolling retraining.
    
    - Train on past `train_size` days
    - Predict next `test_size` days
    - Roll forward and retrain
    """
    log(f"Starting walk-forward backtest: train={train_size}, test={test_size}")
    
    all_predictions = []
    all_actuals = []
    all_dates = []
    fold_results = []
    
    n = len(df)
    start = 0
    fold = 0
    
    while start + train_size + test_size <= n:
        fold += 1
        
        # Get train/test indices
        train_end = start + train_size
        test_end = train_end + test_size
        
        train_df = df.iloc[start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        
        # Separate features and target
        X_train = train_df[feature_cols].copy()
        y_train = train_df["target"].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df["target"].copy()
        
        # Train model
        if model_type == "rf":
            model = train_random_forest(X_train, y_train)
        else:
            model = train_xgboost(X_train, y_train)
        
        # Predict
        preds = predict(model, X_test)
        
        # Store results
        all_predictions.extend(preds)
        all_actuals.extend(y_test.values)
        all_dates.extend(test_df.index.tolist())
        
        # Fold metrics
        fold_rmse = rmse(y_test, preds)
        fold_da = directional_accuracy(y_test, preds)
        fold_sharpe = simulated_sharpe(y_test, preds)
        
        fold_results.append({
            "fold": fold,
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "rmse": fold_rmse,
            "directional_accuracy": fold_da,
            "sharpe": fold_sharpe,
        })
        
        # Roll forward
        start += test_size
    
    log(f"Completed {fold} walk-forward folds")
    
    # Aggregate results
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    overall_rmse = rmse(all_actuals, all_predictions)
    overall_da = directional_accuracy(all_actuals, all_predictions)
    overall_sharpe = simulated_sharpe(all_actuals, all_predictions)
    
    results_df = pd.DataFrame(fold_results)
    
    print("\n=== WALK-FORWARD RESULTS BY FOLD ===")
    print(results_df.to_string(index=False))
    
    print("\n=== OVERALL WALK-FORWARD METRICS ===")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Overall Directional Accuracy: {overall_da:.2f}%")
    print(f"Overall Simulated Sharpe: {overall_sharpe:.4f}")
    
    print("\n=== FOLD STATISTICS ===")
    print(f"Mean RMSE: {results_df['rmse'].mean():.6f} ± {results_df['rmse'].std():.6f}")
    print(f"Mean Dir Acc: {results_df['directional_accuracy'].mean():.2f}% ± {results_df['directional_accuracy'].std():.2f}%")
    print(f"Mean Sharpe: {results_df['sharpe'].mean():.4f} ± {results_df['sharpe'].std():.4f}")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "date": all_dates,
        "actual": all_actuals,
        "predicted": all_predictions,
        "correct_direction": np.sign(all_actuals) == np.sign(all_predictions),
    }).set_index("date")
    
    return {
        "predictions_df": predictions_df,
        "fold_results": results_df,
        "overall_rmse": overall_rmse,
        "overall_da": overall_da,
        "overall_sharpe": overall_sharpe,
    }


if __name__ == "__main__":
    # Load data
    log("Loading data...")
    df = load_data("data/AAPL_data.csv")
    df = prepare_all_features_v2(df, horizon=5)
    
    # Feature columns
    feature_cols = config.CORE_FEATURES
    
    model_df = df[feature_cols + ["target"]].copy()
    
    # Run walk-forward
    results = walk_forward_backtest(
        df=model_df,
        feature_cols=feature_cols,
        train_size=config.WF_TRAIN_SIZE,
        test_size=config.WF_TEST_SIZE,
        model_type=config.MODEL_TYPE,
    )
    
    # Save predictions
    results["predictions_df"].to_csv("reports/walk_forward_predictions.csv")
    results["fold_results"].to_csv("reports/walk_forward_folds.csv")
    log("Saved walk-forward results to reports/")
