import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from splitter import simple_split, separate_features_target
from features import prepare_all_features
from data_collector import load_data


def train_linear(X_train, y_train):
    """
    Train a linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train a random forest regressor.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost regressor.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Make predictions and return as numpy array.
    """
    return np.asarray(model.predict(X_test))


def get_feature_importance(model, feature_names):
    """
    Return sorted feature importances.
    - Linear model: abs(coefficients)
    - Tree models: feature_importances_
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not support feature importance extraction.")

    pairs = list(zip(feature_names, importances))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs


def print_top_features(importances, n=10):
    """
    Print top N most important features.
    """
    for i, (name, importance) in enumerate(importances[:n], start=1):
        print(f"{i:>2}. {name:<15} {importance:.6f}")


if __name__ == "__main__":
    # Load data from ML-1
    df = load_data("data/AAPL_data.csv")

    # Rebuild ML-2 features
    df = prepare_all_features(df, horizon=5)

    # ML feature set
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

    # Keep only chosen features + target
    model_df = df[selected_features + ["target"]].copy()

    # ML-3 split
    train_df, val_df, test_df = simple_split(model_df, train_pct=0.7, val_pct=0.15)

    X_train, y_train = separate_features_target(train_df)
    X_val, y_val = separate_features_target(val_df)

    # Train models
    linear_model = train_linear(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Predict on validation
    linear_preds = predict(linear_model, X_val)
    rf_preds = predict(rf_model, X_val)
    xgb_preds = predict(xgb_model, X_val)

    print("\nValidation prediction shapes:")
    print("Linear:", linear_preds.shape)
    print("Random Forest:", rf_preds.shape)
    print("XGBoost:", xgb_preds.shape)

    # Feature importance
    linear_importance = get_feature_importance(linear_model, X_train.columns)
    rf_importance = get_feature_importance(rf_model, X_train.columns)
    xgb_importance = get_feature_importance(xgb_model, X_train.columns)

    print("\n=== Top 5 Features: Linear Regression ===")
    print_top_features(linear_importance, n=5)

    print("\n=== Top 5 Features: Random Forest ===")
    print_top_features(rf_importance, n=5)

    print("\n=== Top 5 Features: XGBoost ===")
    print_top_features(xgb_importance, n=5)
