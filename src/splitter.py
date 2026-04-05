import pandas as pd


DROP_COLUMNS = ["target", "Close", "Open", "High", "Low", "Volume", "Adj Close"]


def simple_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15):
    """
    Split chronologically into train / validation / test.
    Never shuffle time series data.
    """
    n = len(df)

    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def separate_features_target(split_df: pd.DataFrame):
    """
    Separate X and y from one split.
    """
    X = split_df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    y = split_df["target"].copy()
    return X, y


def walk_forward_split(df: pd.DataFrame, train_size: int = 252, test_size: int = 21):
    """
    Generator for walk-forward validation.

    Example:
    - train: rows 0-251, test: 252-272
    - train: rows 21-272, test: 273-293
    """
    start = 0
    n = len(df)

    while start + train_size + test_size <= n:
        train_df = df.iloc[start:start + train_size].copy()
        test_df = df.iloc[start + train_size:start + train_size + test_size].copy()

        yield train_df, test_df

        start += test_size


def validate_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Ensure all training dates are before all test dates.
    """
    train_last = train_df.index.max()
    test_first = test_df.index.min()

    if train_last >= test_first:
        raise ValueError(
            f"Data leakage detected: train end date {train_last} is not before test start date {test_first}"
        )


def print_split_info(name: str, df: pd.DataFrame):
    """
    Print split summary.
    """
    print(f"\n{name}")
    print(f"Samples: {len(df)}")
    print(f"Start date: {df.index.min()}")
    print(f"End date:   {df.index.max()}")


if __name__ == "__main__":
    from features import prepare_all_features
    from data_collector import load_data

    # Load ML-1 output
    df = load_data("data/AAPL_data.csv")

    # Rebuild ML-2 features
    df = prepare_all_features(df, horizon=5)

    # -------------------------
    # 1) Simple chronological split
    # -------------------------
    train_df, val_df, test_df = simple_split(df, train_pct=0.7, val_pct=0.15)

    validate_no_leakage(train_df, val_df)
    validate_no_leakage(val_df, test_df)

    print("\n=== SIMPLE SPLIT ===")
    print_split_info("TRAIN", train_df)
    print_split_info("VALIDATION", val_df)
    print_split_info("TEST", test_df)

    X_train, y_train = separate_features_target(train_df)
    X_val, y_val = separate_features_target(val_df)
    X_test, y_test = separate_features_target(test_df)

    print("\nX_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nFeature columns used for modeling:")
    print(X_train.columns.tolist())

    # -------------------------
    # 2) Walk-forward split
    # -------------------------
    print("\n=== WALK-FORWARD SPLITS (first 3) ===")
    wf_generator = walk_forward_split(df, train_size=252, test_size=21)

    for i, (wf_train, wf_test) in enumerate(wf_generator, start=1):
        validate_no_leakage(wf_train, wf_test)

        print(f"\nWalk-forward split #{i}")
        print_split_info("WF TRAIN", wf_train)
        print_split_info("WF TEST", wf_test)

        if i == 3:
            break
