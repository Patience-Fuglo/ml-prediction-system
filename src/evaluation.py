import numpy as np
import pandas as pd


def rmse(actual, predicted):
    """
    Root Mean Squared Error.
    Lower is better.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    """
    Mean Absolute Error.
    Lower is better.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.mean(np.abs(actual - predicted))


def directional_accuracy(actual, predicted):
    """
    Percentage of times model gets direction correct.
    Above 50% is better than random.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.mean(np.sign(actual) == np.sign(predicted)) * 100


def simulated_sharpe(actual, predicted):
    """
    Simple trading rule:
    - long when predicted > 0
    - short when predicted < 0 via sign(predicted)
    Then annualize Sharpe with sqrt(252).
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    strategy_returns = actual * np.sign(predicted)
    std = np.std(strategy_returns)

    if std == 0:
        return 0.0

    return np.mean(strategy_returns) / std * np.sqrt(252)


def evaluate_model(actual, predicted, model_name):
    """
    Compute all metrics and print them.
    """
    results = {
        "Model": model_name,
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "Directional Accuracy (%)": directional_accuracy(actual, predicted),
        "Simulated Sharpe": simulated_sharpe(actual, predicted),
    }

    print(f"\n=== {model_name} ===")
    print(f"RMSE:                   {results['RMSE']:.6f}")
    print(f"MAE:                    {results['MAE']:.6f}")
    print(f"Directional Accuracy:   {results['Directional Accuracy (%)']:.2f}%")
    print(f"Simulated Sharpe:       {results['Simulated Sharpe']:.4f}")

    return results


def compare_all_models(models_results):
    """
    Print a side-by-side comparison table.
    Best:
    - RMSE: lowest
    - MAE: lowest
    - Directional Accuracy: highest
    - Simulated Sharpe: highest
    """
    results_df = pd.DataFrame(models_results)

    best_rmse = results_df["RMSE"].min()
    best_mae = results_df["MAE"].min()
    best_da = results_df["Directional Accuracy (%)"].max()
    best_sharpe = results_df["Simulated Sharpe"].max()

    print("\n=== MODEL COMPARISON ===")
    print(
        f"{'Model':<18} {'RMSE':>12} {'MAE':>12} {'Dir Acc %':>12} {'Sharpe':>12}"
    )
    print("-" * 70)

    for _, row in results_df.iterrows():
        rmse_str = f"{row['RMSE']:.6f}"
        mae_str = f"{row['MAE']:.6f}"
        da_str = f"{row['Directional Accuracy (%)']:.2f}"
        sharpe_str = f"{row['Simulated Sharpe']:.4f}"

        if row["RMSE"] == best_rmse:
            rmse_str += " *"
        if row["MAE"] == best_mae:
            mae_str += " *"
        if row["Directional Accuracy (%)"] == best_da:
            da_str += " *"
        if row["Simulated Sharpe"] == best_sharpe:
            sharpe_str += " *"

        print(
            f"{row['Model']:<18} {rmse_str:>12} {mae_str:>12} {da_str:>12} {sharpe_str:>12}"
        )

    print("\n* = best value for that metric")


if __name__ == "__main__":
    from models import train_linear, train_random_forest, train_xgboost, predict
    from splitter import simple_split, separate_features_target
    from features import prepare_all_features
    from data_collector import load_data
    
    # Load ML-1 data
    df = load_data("data/AAPL_data.csv")

    # Build ML-2 features
    df = prepare_all_features(df, horizon=5)

    # Use focused ML features
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

    model_df = df[selected_features + ["target"]].copy()

    # ML-3 chronological split
    train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)

    X_train, y_train = separate_features_target(train_df)
    X_val, y_val = separate_features_target(val_df)

    # ML-4 models
    linear_model = train_linear(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Predictions
    linear_preds = predict(linear_model, X_val)
    rf_preds = predict(rf_model, X_val)
    xgb_preds = predict(xgb_model, X_val)

    # Evaluate
    results = []
    results.append(evaluate_model(y_val, linear_preds, "Linear Regression"))
    results.append(evaluate_model(y_val, rf_preds, "Random Forest"))
    results.append(evaluate_model(y_val, xgb_preds, "XGBoost"))

    # Compare
    compare_all_models(results)
