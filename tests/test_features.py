"""
Tests for features module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import (
    add_moving_averages,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_volume_features,
    add_target,
    prepare_all_features,
)


class TestAddMovingAverages:
    """Tests for add_moving_averages function."""
    
    def test_adds_sma_columns(self, sample_stock_data):
        """Test that SMA columns are added."""
        df = add_moving_averages(sample_stock_data)
        
        for w in [5, 10, 20, 50]:
            assert f"SMA_{w}" in df.columns
    
    def test_custom_windows(self, sample_stock_data):
        """Test with custom window sizes."""
        df = add_moving_averages(sample_stock_data, windows=[3, 7, 14])
        
        assert "SMA_3" in df.columns
        assert "SMA_7" in df.columns
        assert "SMA_14" in df.columns
        assert "SMA_5" not in df.columns
    
    def test_sma_calculation(self):
        """Test SMA calculation is correct."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }, index=dates)
        
        df = add_moving_averages(df, windows=[3])
        
        # SMA_3 at index 2 should be (1+2+3)/3 = 2
        assert np.isclose(df["SMA_3"].iloc[2], 2.0)
        # SMA_3 at last index should be (8+9+10)/3 = 9
        assert np.isclose(df["SMA_3"].iloc[-1], 9.0)
    
    def test_nan_for_insufficient_data(self, sample_stock_data):
        """Test that NaN is returned when insufficient data."""
        df = add_moving_averages(sample_stock_data, windows=[50])
        
        # First 49 values should be NaN
        assert df["SMA_50"].iloc[:49].isna().all()
        assert df["SMA_50"].iloc[49:].notna().all()


class TestAddRSI:
    """Tests for add_rsi function."""
    
    def test_adds_rsi_column(self, sample_stock_data):
        """Test that RSI column is added."""
        df = add_rsi(sample_stock_data)
        
        assert "RSI" in df.columns
    
    def test_rsi_range(self, sample_stock_data):
        """Test that RSI is between 0 and 100."""
        df = add_rsi(sample_stock_data)
        rsi_valid = df["RSI"].dropna()
        
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()
    
    def test_rsi_custom_period(self, sample_stock_data):
        """Test RSI with custom period."""
        df_14 = add_rsi(sample_stock_data, period=14)
        df_7 = add_rsi(sample_stock_data, period=7)
        
        # Different periods should produce different values
        assert not df_14["RSI"].equals(df_7["RSI"])


class TestAddMACD:
    """Tests for add_macd function."""
    
    def test_adds_macd_columns(self, sample_stock_data):
        """Test that MACD and MACD_signal columns are added."""
        df = add_macd(sample_stock_data)
        
        assert "MACD" in df.columns
        assert "MACD_signal" in df.columns
    
    def test_macd_is_ema_difference(self, sample_stock_data):
        """Test that MACD is the difference of EMAs."""
        df = sample_stock_data.copy()
        
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        expected_macd = ema_12 - ema_26
        
        df = add_macd(df)
        
        pd.testing.assert_series_equal(df["MACD"], expected_macd, check_names=False)


class TestAddBollingerBands:
    """Tests for add_bollinger_bands function."""
    
    def test_adds_bollinger_columns(self, sample_stock_data):
        """Test that Bollinger Band columns are added."""
        df = add_bollinger_bands(sample_stock_data)
        
        assert "BB_middle" in df.columns
        assert "BB_upper" in df.columns
        assert "BB_lower" in df.columns
        assert "BB_position" in df.columns
    
    def test_upper_above_middle_above_lower(self, sample_stock_data):
        """Test band ordering."""
        df = add_bollinger_bands(sample_stock_data)
        df = df.dropna()
        
        assert (df["BB_upper"] > df["BB_middle"]).all()
        assert (df["BB_middle"] > df["BB_lower"]).all()
    
    def test_bb_position_range(self, sample_stock_data):
        """Test BB position is typically between 0 and 1."""
        df = add_bollinger_bands(sample_stock_data)
        df = df.dropna()
        
        # Most values should be between 0 and 1 (within bands)
        in_range = (df["BB_position"] >= 0) & (df["BB_position"] <= 1)
        assert in_range.mean() > 0.7  # At least 70% within bands


class TestAddVolumeFeatures:
    """Tests for add_volume_features function."""
    
    def test_adds_volume_columns(self, sample_stock_data):
        """Test that volume feature columns are added."""
        df = add_volume_features(sample_stock_data)
        
        assert "volume_sma" in df.columns
        assert "volume_ratio" in df.columns
    
    def test_volume_ratio_around_one(self, sample_stock_data):
        """Test that volume ratio averages around 1."""
        df = add_volume_features(sample_stock_data)
        df = df.dropna()
        
        # Mean volume ratio should be close to 1
        assert np.isclose(df["volume_ratio"].mean(), 1.0, atol=0.2)


class TestAddTarget:
    """Tests for add_target function."""
    
    def test_adds_target_column(self, sample_stock_data):
        """Test that target column is added."""
        df = add_target(sample_stock_data)
        
        assert "target" in df.columns
    
    def test_target_calculation(self):
        """Test target calculation is correct."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
        }, index=dates)
        
        df = add_target(df, horizon=5)
        
        # Target at index 0 should be (110/100) - 1 = 0.10
        assert np.isclose(df["target"].iloc[0], 0.10)
        # Last 5 targets should be NaN (no future data)
        assert df["target"].iloc[-5:].isna().all()
    
    def test_target_custom_horizon(self, sample_stock_data):
        """Test target with custom horizon."""
        df_5 = add_target(sample_stock_data, horizon=5)
        df_10 = add_target(sample_stock_data, horizon=10)
        
        # Different horizons produce different targets
        assert not df_5["target"].equals(df_10["target"])


class TestPrepareAllFeatures:
    """Tests for prepare_all_features function."""
    
    def test_returns_dataframe(self, sample_stock_data):
        """Test that function returns a DataFrame."""
        df = prepare_all_features(sample_stock_data)
        
        assert isinstance(df, pd.DataFrame)
    
    def test_no_nan_values(self, sample_stock_data):
        """Test that no NaN values remain."""
        df = prepare_all_features(sample_stock_data)
        
        assert df.isna().sum().sum() == 0
    
    def test_contains_all_features(self, sample_stock_data, feature_columns):
        """Test that all expected features are present."""
        df = prepare_all_features(sample_stock_data)
        
        for col in feature_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_contains_target(self, sample_stock_data):
        """Test that target column is present."""
        df = prepare_all_features(sample_stock_data)
        
        assert "target" in df.columns
