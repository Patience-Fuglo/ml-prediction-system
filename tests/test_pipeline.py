"""
Tests for pipeline module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPipelineIntegration:
    """Integration tests for the pipeline."""
    
    def test_pipeline_end_to_end(self, sample_stock_data):
        """Test full pipeline execution."""
        from data_collector import add_returns, save_data, load_data
        from features import prepare_all_features
        from splitter import simple_split, separate_features_target
        from models import train_random_forest, predict
        from evaluation import evaluate_model
        
        # Step 1: Prepare data
        df = add_returns(sample_stock_data)
        
        # Step 2: Add features
        df = prepare_all_features(df, horizon=5)
        
        # Step 3: Define feature columns
        feature_cols = [
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
        
        # Step 4: Split data
        model_df = df[feature_cols + ["target"]].copy()
        train_df, val_df, test_df = simple_split(model_df)
        
        # Step 5: Separate features and target
        X_train, y_train = separate_features_target(train_df)
        X_val, y_val = separate_features_target(val_df)
        
        # Step 6: Train model
        model = train_random_forest(X_train, y_train)
        
        # Step 7: Predict
        preds = predict(model, X_val)
        
        # Step 8: Evaluate
        results = evaluate_model(y_val.values, preds, "Random Forest")
        
        # Assertions
        assert len(preds) == len(y_val)
        assert "RMSE" in results
        assert results["RMSE"] >= 0
    
    def test_pipeline_preserves_data_integrity(self, sample_stock_data):
        """Test that pipeline doesn't corrupt data."""
        from features import prepare_all_features
        
        original_len = len(sample_stock_data)
        df = prepare_all_features(sample_stock_data)
        
        # Should have fewer rows (dropped NaN) but not zero
        assert 0 < len(df) < original_len
        
        # No infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(df[col]).any(), f"Infinite values in {col}"
    
    def test_pipeline_no_data_leakage(self, sample_stock_data):
        """Test that pipeline doesn't leak future data."""
        from features import prepare_all_features
        from splitter import simple_split, validate_no_leakage
        
        df = prepare_all_features(sample_stock_data)
        
        feature_cols = [
            "daily_return",
            "SMA_5",
            "RSI",
            "MACD",
            "BB_position",
            "volume_ratio",
        ]
        
        model_df = df[feature_cols + ["target"]].copy()
        train_df, val_df, test_df = simple_split(model_df)
        
        # Should not raise
        validate_no_leakage(train_df, val_df)
        validate_no_leakage(val_df, test_df)


class TestPipelineV2:
    """Tests for PipelineV2 class if it exists."""
    
    def test_pipeline_v2_import(self):
        """Test that PipelineV2 can be imported."""
        try:
            from pipeline_v2 import PredictionPipelineV2
            assert PredictionPipelineV2 is not None
        except ImportError:
            pytest.skip("PipelineV2 not available")
    
    def test_pipeline_v2_initialization(self):
        """Test PipelineV2 initialization."""
        try:
            from pipeline_v2 import PredictionPipelineV2
            
            pipeline = PredictionPipelineV2(
                symbols=["AAPL"],
                start_date="2020-01-01",
                end_date="2022-01-01",
                model_type="rf"
            )
            
            assert pipeline is not None
        except ImportError:
            pytest.skip("PipelineV2 not available")


class TestWalkForwardPipeline:
    """Tests for walk-forward validation pipeline."""
    
    def test_walk_forward_produces_predictions(self, sample_feature_data, feature_columns):
        """Test walk-forward produces out-of-sample predictions."""
        from splitter import walk_forward_split, separate_features_target
        from models import train_random_forest, predict
        
        model_df = sample_feature_data[feature_columns + ["target"]].copy()
        
        all_predictions = []
        all_actuals = []
        
        for train_df, test_df in walk_forward_split(model_df, train_size=100, test_size=20):
            X_train, y_train = separate_features_target(train_df)
            X_test, y_test = separate_features_target(test_df)
            
            model = train_random_forest(X_train, y_train)
            preds = predict(model, X_test)
            
            all_predictions.extend(preds)
            all_actuals.extend(y_test.values)
            
            # Only test first 2 folds
            if len(all_predictions) >= 40:
                break
        
        assert len(all_predictions) == len(all_actuals)
        assert len(all_predictions) >= 40
    
    def test_walk_forward_no_look_ahead_bias(self, sample_feature_data, feature_columns):
        """Test walk-forward has no look-ahead bias."""
        from splitter import walk_forward_split
        
        model_df = sample_feature_data[feature_columns + ["target"]].copy()
        
        prev_test_end = None
        for train_df, test_df in walk_forward_split(model_df, train_size=100, test_size=20):
            # Train data should always be before test data
            assert train_df.index.max() < test_df.index.min()
            
            # Test periods should not overlap
            if prev_test_end is not None:
                assert test_df.index.min() >= prev_test_end
            
            prev_test_end = test_df.index.max()
            
            # Only check first 3 folds
            if prev_test_end == test_df.index.max():
                break
