"""
Tests for data_collector module.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collector import (
    clean_data,
    add_returns,
    save_data,
    load_data,
)


class TestCleanData:
    """Tests for clean_data function."""
    
    def test_removes_nan_values(self, sample_stock_data):
        """Test that NaN values are removed."""
        df = sample_stock_data.copy()
        df.iloc[0, 0] = np.nan
        df.iloc[5, 1] = np.nan
        
        cleaned = clean_data(df)
        
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) < len(df)
    
    def test_removes_duplicate_dates(self):
        """Test that duplicate index values are removed."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1001, 1002, 1003, 1004],
        }, index=dates)
        
        # Add duplicate date
        df = pd.concat([df, df.iloc[[2]]])
        assert len(df) == 6
        
        cleaned = clean_data(df)
        assert len(cleaned) == 5
        assert not cleaned.index.duplicated().any()
    
    def test_warns_on_non_positive_prices(self, sample_stock_data):
        """Test that warning is raised for non-positive prices."""
        df = sample_stock_data.copy()
        df.iloc[10, df.columns.get_loc("Close")] = 0
        
        with pytest.warns(UserWarning, match="Non-positive values"):
            clean_data(df)


class TestAddReturns:
    """Tests for add_returns function."""
    
    def test_adds_daily_return_column(self, sample_stock_data):
        """Test that daily_return column is added."""
        df = add_returns(sample_stock_data)
        
        assert "daily_return" in df.columns
    
    def test_adds_log_return_column(self, sample_stock_data):
        """Test that log_return column is added."""
        df = add_returns(sample_stock_data)
        
        assert "log_return" in df.columns
    
    def test_daily_return_calculation(self):
        """Test daily return calculation is correct."""
        dates = pd.date_range(start="2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "Close": [100.0, 110.0, 99.0],
        }, index=dates)
        
        df = add_returns(df)
        
        assert np.isnan(df["daily_return"].iloc[0])
        assert np.isclose(df["daily_return"].iloc[1], 0.10)
        assert np.isclose(df["daily_return"].iloc[2], -0.10)
    
    def test_log_return_calculation(self):
        """Test log return calculation is correct."""
        dates = pd.date_range(start="2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "Close": [100.0, 110.0, 99.0],
        }, index=dates)
        
        df = add_returns(df)
        
        assert np.isclose(df["log_return"].iloc[1], np.log(110/100))
        assert np.isclose(df["log_return"].iloc[2], np.log(99/110))
    
    def test_raises_error_without_close_column(self):
        """Test that KeyError is raised if Close column missing."""
        df = pd.DataFrame({"Open": [100, 101, 102]})
        
        with pytest.raises(KeyError, match="Close"):
            add_returns(df)


class TestSaveLoadData:
    """Tests for save_data and load_data functions."""
    
    def test_save_and_load_roundtrip(self, sample_stock_data):
        """Test that data can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_data.csv")
            
            save_data(sample_stock_data, filepath)
            loaded = load_data(filepath)
            
            assert loaded.shape == sample_stock_data.shape
            pd.testing.assert_index_equal(loaded.index, sample_stock_data.index)
    
    def test_save_creates_directory(self, sample_stock_data):
        """Test that save_data creates directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "nested", "test_data.csv")
            
            save_data(sample_stock_data, filepath)
            
            assert os.path.exists(filepath)
    
    def test_load_parses_dates(self, sample_stock_data):
        """Test that dates are parsed correctly on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_data.csv")
            
            save_data(sample_stock_data, filepath)
            loaded = load_data(filepath)
            
            assert isinstance(loaded.index, pd.DatetimeIndex)
