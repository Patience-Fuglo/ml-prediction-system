import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

from .splitter import simple_split, separate_features_target
from .features import prepare_all_features
from .data_collector import load_data
from .evaluation import rmse, directional_accuracy
from .models import train_random_forest, predict


def correlation_with_target(X, y):
    """
    Correlate each feature with target and sort by absolute correlation.
    """
    correlations = X.corrwith(y)
    correlations = correlations.sort_values(key=lambda s: s.abs(), ascending=False)
    return correlations


def remove_redundant_features(X, threshold=0.9):
    """
    Drop one feature from pairs with correlation above threshold.
    """
    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    dropped_features = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    X_reduced = X.drop(columns=dropped_features).copy()
    return X_reduced, dropped_features


def recursive_elimination(model, X_train, y_train, X_test, y_test, n_features=10):
    """
    Use RFE to keep the top n_features.
    """
    selector = RFE(estimator=model, n_features_to_select=n_features, step=1)
    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.support_].tolist()

    print(f"\nRFE selected {len(selected_features)} features:")
    print(selected_features)

    return selected_features


def compare_feature_sets(X_train, y_train, X_test, y_test, feature_sets):
    """
    Train XGBoost on each feature set and compare RMSE + directional accuracy.
    """
    results = []

    for set_name, cols in feature_sets.items():
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            objective="reg:squarederror"
        )

        model.fit(X_train[cols], y_train)
        preds = predict(model, X_test[cols])

        set_rmse = rmse(y_test, preds)
        set_da = directional_accuracy(y_test, preds)

        results.append({
            "feature_set": set_name,
            "num_features": len(cols),
            "rmse": set_rmse,
            "directional_accuracy": set_da
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["directional_accuracy", "rmse"],
        ascending=[False, True]
    )

    print("\n=== FEATURE SET COMPARISON (XGBoost) ===")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    # Load ML-1 data
    df = load_data("data/AAPL_data.csv")

    # Build ML-2 features
    df = prepare_all_features(df, horizon=5)

    # Start with fuller feature set
    all_features = [
        "daily_return",
        "log_return",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_signal",
        "BB_middle",
        "BB_upper",
        "BB_lower",
        "BB_position",
        "volume_sma",
        "volume_ratio"
    ]

    model_df = df[all_features + ["target"]].copy()

    # Use ML-3 chronological split
    train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)

    # For ML-7, use train as train and val as test for selection/validation
    X_train, y_train = separate_features_target(train_df)
    X_test, y_test = separate_features_target(val_df)

    print("\n=== STEP 1: CORRELATION WITH TARGET ===")
    corr_series = correlation_with_target(X_train, y_train)
    print(corr_series)

    # Keep all non-trivial features; you can tighten later if wanted
    corr_filtered_features = corr_series[ corr_series.abs() > 0.01 ].index.tolist()

    print("\nFeatures after correlation filter:")
    print(corr_filtered_features)

    print("\n=== STEP 2: REMOVE REDUNDANT FEATURES ===")
    X_corr_filtered = X_train[corr_filtered_features].copy()
    X_non_redundant, dropped_features = remove_redundant_features(X_corr_filtered, threshold=0.9)

    non_redundant_features = X_non_redundant.columns.tolist()

    print("\nDropped redundant features:")
    print(dropped_features)

    print("\nFeatures after redundancy removal:")
    print(non_redundant_features)

    print("\n=== STEP 3: RECURSIVE FEATURE ELIMINATION ===")
    rfe_base_model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror"
    )

    rfe_n = min(10, len(non_redundant_features))
    rfe_features = recursive_elimination(
        model=rfe_base_model,
        X_train=X_train[non_redundant_features],
        y_train=y_train,
        X_test=X_test[non_redundant_features],
        y_test=y_test,
        n_features=rfe_n
    )

    print("\n=== STEP 4: COMPARE FEATURE SETS ===")
    top_10_corr = corr_series.index[:10].tolist()

    feature_sets = {
        "all_features": all_features,
        "corr_filtered": corr_filtered_features,
        "non_redundant": non_redundant_features,
        "top_10_corr": top_10_corr,
        "rfe_selected": rfe_features
    }

    results_df = compare_feature_sets(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_sets=feature_sets
    )

    print("\n=== SUMMARY ===")
    print(f"Starting features: {len(all_features)}")
    print(f"After correlation filter: {len(corr_filtered_features)}")
    print(f"After redundancy removal: {len(non_redundant_features)}")
    print(f"After RFE: {len(rfe_features)}")

    print("\nBest-performing feature set:")
    best_row = results_df.iloc[0]
    print(best_row.to_dict())

    print("\n=== OPTIONAL: RETRAIN BEST MODEL TYPE (Random Forest) WITH REDUCED FEATURES ===")
    # Retrain your best model family from ML-5 using the RFE feature set
    rf_model = train_random_forest(X_train[rfe_features], y_train)
    rf_preds = predict(rf_model, X_test[rfe_features])

    print(f"Random Forest with RFE features RMSE: {rmse(y_test, rf_preds):.6f}")
    print(f"Random Forest with RFE features Directional Accuracy: {directional_accuracy(y_test, rf_preds):.2f}%")
