"""
Tests for splitter module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from splitter import (
    simple_split,
    separate_features_target,
    walk_forward_split,
    validate_no_leakage,
)


class TestSimpleSplit:
    """Tests for simple_split function."""
    
    def test_returns_three_dataframes(self, sample_feature_data):
        """Test that three DataFrames are returned."""
        train, val, test = simple_split(sample_feature_data)
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
    
    def test_split_proportions(self, sample_feature_data):
        """Test that split proportions are approximately correct."""
        train, val, test = simple_split(sample_feature_data, train_pct=0.7, val_pct=0.15)
        
        total = len(sample_feature_data)
        
        assert np.isclose(len(train) / total, 0.7, atol=0.02)
        assert np.isclose(len(val) / total, 0.15, atol=0.02)
        assert np.isclose(len(test) / total, 0.15, atol=0.02)
    
    def test_no_data_loss(self, sample_feature_data):
        """Test that no data is lost in split."""
        train, val, test = simple_split(sample_feature_data)
        
        assert len(train) + len(val) + len(test) == len(sample_feature_data)
    
    def test_chronological_order(self, sample_feature_data):
        """Test that splits maintain chronological order."""
        train, val, test = simple_split(sample_feature_data)
        
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()
    
    def test_custom_proportions(self, sample_feature_data):
        """Test with custom split proportions."""
        train, val, test = simple_split(sample_feature_data, train_pct=0.6, val_pct=0.2)
        
        total = len(sample_feature_data)
        
        assert np.isclose(len(train) / total, 0.6, atol=0.02)
        assert np.isclose(len(val) / total, 0.2, atol=0.02)


class TestSeparateFeaturesTarget:
    """Tests for separate_features_target function."""
    
    def test_returns_x_and_y(self, sample_feature_data):
        """Test that X and y are returned."""
        X, y = separate_features_target(sample_feature_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
    
    def test_target_not_in_features(self, sample_feature_data):
        """Test that target is not in features."""
        X, y = separate_features_target(sample_feature_data)
        
        assert "target" not in X.columns
    
    def test_price_columns_not_in_features(self, sample_feature_data):
        """Test that price columns are removed."""
        X, y = separate_features_target(sample_feature_data)
        
        price_cols = ["Close", "Open", "High", "Low", "Volume", "Adj Close"]
        for col in price_cols:
            assert col not in X.columns
    
    def test_y_matches_target(self, sample_feature_data):
        """Test that y equals the target column."""
        X, y = separate_features_target(sample_feature_data)
        
        pd.testing.assert_series_equal(y, sample_feature_data["target"], check_names=False)
    
    def test_same_length(self, sample_feature_data):
        """Test that X and y have same length."""
        X, y = separate_features_target(sample_feature_data)
        
        assert len(X) == len(y)


class TestWalkForwardSplit:
    """Tests for walk_forward_split function."""
    
    def test_yields_train_test_pairs(self, sample_feature_data):
        """Test that generator yields train/test pairs."""
        generator = walk_forward_split(sample_feature_data, train_size=100, test_size=20)
        
        train, test = next(generator)
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
    
    def test_train_size_correct(self, sample_feature_data):
        """Test that train size matches specified."""
        generator = walk_forward_split(sample_feature_data, train_size=100, test_size=20)
        
        for train, test in generator:
            assert len(train) == 100
            break
    
    def test_test_size_correct(self, sample_feature_data):
        """Test that test size matches specified."""
        generator = walk_forward_split(sample_feature_data, train_size=100, test_size=20)
        
        for train, test in generator:
            assert len(test) == 20
            break
    
    def test_no_overlap(self, sample_feature_data):
        """Test that train and test don't overlap."""
        generator = walk_forward_split(sample_feature_data, train_size=100, test_size=20)
        
        for train, test in generator:
            assert train.index.max() < test.index.min()
            break
    
    def test_rolling_window(self, sample_feature_data):
        """Test that window rolls forward."""
        generator = walk_forward_split(sample_feature_data, train_size=100, test_size=20)
        
        prev_train_end = None
        for i, (train, test) in enumerate(generator):
            if prev_train_end is not None:
                # Each new train window should start later
                assert train.index.min() > prev_train_end - pd.Timedelta(days=30)
            prev_train_end = train.index.max()
            
            if i >= 2:
                break
    
    def test_generates_multiple_folds(self, sample_feature_data):
        """Test that multiple folds are generated."""
        generator = walk_forward_split(sample_feature_data, train_size=100, test_size=20)
        
        count = sum(1 for _ in generator)
        
        assert count > 1


class TestValidateNoLeakage:
    """Tests for validate_no_leakage function."""
    
    def test_passes_valid_split(self, sample_feature_data):
        """Test that valid split passes validation."""
        train, val, test = simple_split(sample_feature_data)
        
        # Should not raise
        validate_no_leakage(train, val)
        validate_no_leakage(val, test)
    
    def test_fails_on_overlap(self, sample_feature_data):
        """Test that overlapping data raises error."""
        train, val, _ = simple_split(sample_feature_data)
        
        # Create overlap by including some train data in test
        overlapping_test = pd.concat([train.iloc[-10:], val])
        
        with pytest.raises(ValueError, match="leakage"):
            validate_no_leakage(train, overlapping_test)
    
    def test_fails_on_reversed_order(self, sample_feature_data):
        """Test that reversed order raises error."""
        train, val, _ = simple_split(sample_feature_data)
        
        with pytest.raises(ValueError, match="leakage"):
            validate_no_leakage(val, train)  # Wrong order
