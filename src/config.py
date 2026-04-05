# Configuration file for ML Predictor V2

# ===== DATA SETTINGS =====
SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "JPM", "XOM", "SPY", "QQQ"]
DEFAULT_SYMBOL = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

# ===== FEATURE SETTINGS =====
SMA_WINDOWS = [5, 10, 20, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
ATR_PERIOD = 14
VOLATILITY_WINDOW = 20

# ===== MODEL SETTINGS =====
HORIZON = 5  # Days ahead to predict
MODEL_TYPE = "rf"  # "linear", "rf", "xgboost"
RANDOM_STATE = 42

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1

# ===== SPLITTING SETTINGS =====
TRAIN_PCT = 0.7
VAL_PCT = 0.15

# Walk-forward settings
WF_TRAIN_SIZE = 252  # ~1 year
WF_TEST_SIZE = 21    # ~1 month

# ===== TRADING SETTINGS =====
STARTING_CASH = 100000
FEE_RATE = 0.001       # 0.1% per trade
SLIPPAGE = 0.0005      # 0.05% slippage
SPREAD = 0.0001        # 0.01% bid-ask spread

# Signal thresholds (avoid weak signals)
LONG_THRESHOLD = 0.005   # pred > 0.5% to go long
SHORT_THRESHOLD = -0.005 # pred < -0.5% to go short

# Position sizing
MAX_POSITION_PCT = 1.0   # Max % of portfolio in one position

# ===== OUTPUT SETTINGS =====
SAVE_TRADES = True
SAVE_PREDICTIONS = True
SAVE_REPORTS = True
SAVE_PLOTS = True

# ===== PATHS =====
DATA_DIR = "data"
MODELS_DIR = "models_saved"
REPORTS_DIR = "reports"
PLOTS_DIR = "plots"

# ===== FEATURE COLUMNS =====
# Core features (from ML-7 analysis)
CORE_FEATURES = [
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

# Extended features (V2)
EXTENDED_FEATURES = CORE_FEATURES + [
    "vol_20",
    "mom_5",
    "mom_10",
    "mom_20",
    "price_to_sma20",
    "macd_hist",
    "ATR",
    "return_zscore",
]
