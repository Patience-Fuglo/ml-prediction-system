"""
Tests for evaluation module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import (
    rmse,
    mae,
    directional_accuracy,
    simulated_sharpe,
    evaluate_model,
)


class TestRMSE:
    """Tests for rmse function."""
    
    def test_perfect_prediction(self):
        """Test RMSE is zero for perfect predictions."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        
        assert rmse(actual, predicted) == 0.0
    
    def test_known_value(self):
        """Test RMSE with known values."""
        actual = np.array([1, 2, 3])
        predicted = np.array([2, 2, 2])  # Errors: 1, 0, 1
        
        # RMSE = sqrt((1 + 0 + 1) / 3) = sqrt(2/3)
        expected = np.sqrt(2/3)
        assert np.isclose(rmse(actual, predicted), expected)
    
    def test_always_positive(self, sample_predictions):
        """Test RMSE is always positive."""
        actual, predicted = sample_predictions
        
        assert rmse(actual, predicted) >= 0
    
    def test_symmetric(self):
        """Test RMSE is symmetric."""
        actual = np.array([1, 2, 3])
        predicted = np.array([2, 3, 4])
        
        assert rmse(actual, predicted) == rmse(predicted, actual)


class TestMAE:
    """Tests for mae function."""
    
    def test_perfect_prediction(self):
        """Test MAE is zero for perfect predictions."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        
        assert mae(actual, predicted) == 0.0
    
    def test_known_value(self):
        """Test MAE with known values."""
        actual = np.array([1, 2, 3])
        predicted = np.array([2, 2, 2])  # Errors: 1, 0, 1
        
        # MAE = (1 + 0 + 1) / 3 = 2/3
        expected = 2/3
        assert np.isclose(mae(actual, predicted), expected)
    
    def test_always_positive(self, sample_predictions):
        """Test MAE is always positive."""
        actual, predicted = sample_predictions
        
        assert mae(actual, predicted) >= 0
    
    def test_less_sensitive_to_outliers(self):
        """Test MAE is less sensitive to outliers than RMSE."""
        actual = np.array([1, 2, 3, 100])  # Outlier at 100
        predicted = np.array([1, 2, 3, 3])
        
        mae_value = mae(actual, predicted)
        rmse_value = rmse(actual, predicted)
        
        # RMSE should be larger due to outlier (less strict assertion)
        assert rmse_value > mae_value


class TestDirectionalAccuracy:
    """Tests for directional_accuracy function."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy for perfect direction predictions."""
        actual = np.array([0.1, -0.1, 0.2, -0.2])
        predicted = np.array([0.05, -0.05, 0.1, -0.15])
        
        assert directional_accuracy(actual, predicted) == 100.0
    
    def test_zero_accuracy(self):
        """Test 0% accuracy for completely wrong predictions."""
        actual = np.array([0.1, -0.1, 0.2, -0.2])
        predicted = np.array([-0.05, 0.05, -0.1, 0.15])
        
        assert directional_accuracy(actual, predicted) == 0.0
    
    def test_range(self, sample_predictions):
        """Test directional accuracy is between 0 and 100."""
        actual, predicted = sample_predictions
        
        da = directional_accuracy(actual, predicted)
        
        assert 0 <= da <= 100
    
    def test_known_value(self):
        """Test with known directional accuracy."""
        actual = np.array([0.1, -0.1, 0.2, -0.2])
        predicted = np.array([0.05, 0.05, 0.1, -0.15])  # 3/4 correct
        
        assert directional_accuracy(actual, predicted) == 75.0


class TestSimulatedSharpe:
    """Tests for simulated_sharpe function."""
    
    def test_positive_sharpe_for_good_predictions(self):
        """Test positive Sharpe for correlated predictions."""
        np.random.seed(42)
        actual = np.random.normal(0.001, 0.02, 252)
        predicted = actual + np.random.normal(0, 0.01, 252)  # Correlated
        
        sharpe = simulated_sharpe(actual, predicted)
        
        assert sharpe > 0
    
    def test_zero_sharpe_for_constant_returns(self):
        """Test Sharpe is zero when std is zero."""
        actual = np.array([0.0, 0.0, 0.0, 0.0])
        predicted = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = simulated_sharpe(actual, predicted)
        
        assert sharpe == 0.0
    
    def test_annualization(self, sample_predictions):
        """Test that Sharpe ratio is annualized."""
        actual, predicted = sample_predictions
        
        sharpe = simulated_sharpe(actual, predicted)
        
        # Calculate manually without annualization
        strategy_returns = actual * np.sign(predicted)
        daily_sharpe = np.mean(strategy_returns) / np.std(strategy_returns)
        
        # Should be multiplied by sqrt(252)
        expected = daily_sharpe * np.sqrt(252)
        assert np.isclose(sharpe, expected)


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_returns_dict(self, sample_predictions):
        """Test that a dictionary is returned."""
        actual, predicted = sample_predictions
        
        results = evaluate_model(actual, predicted, "Test Model")
        
        assert isinstance(results, dict)
    
    def test_contains_all_metrics(self, sample_predictions):
        """Test that all metrics are in result."""
        actual, predicted = sample_predictions
        
        results = evaluate_model(actual, predicted, "Test Model")
        
        assert "Model" in results
        assert "RMSE" in results
        assert "MAE" in results
        assert "Directional Accuracy (%)" in results
        assert "Simulated Sharpe" in results
    
    def test_model_name_correct(self, sample_predictions):
        """Test that model name is stored correctly."""
        actual, predicted = sample_predictions
        
        results = evaluate_model(actual, predicted, "My Model")
        
        assert results["Model"] == "My Model"
    
    def test_metrics_match_individual_functions(self, sample_predictions):
        """Test that metrics match individual function results."""
        actual, predicted = sample_predictions
        
        results = evaluate_model(actual, predicted, "Test")
        
        assert np.isclose(results["RMSE"], rmse(actual, predicted))
        assert np.isclose(results["MAE"], mae(actual, predicted))
        assert np.isclose(results["Directional Accuracy (%)"], directional_accuracy(actual, predicted))
        assert np.isclose(results["Simulated Sharpe"], simulated_sharpe(actual, predicted))
