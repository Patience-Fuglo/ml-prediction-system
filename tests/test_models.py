"""
Tests for models module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    train_linear,
    train_random_forest,
    train_xgboost,
    predict,
    get_feature_importance,
)


@pytest.fixture
def training_data(sample_feature_data, feature_columns):
    """Prepare training data."""
    df = sample_feature_data.copy()
    X = df[feature_columns].iloc[:300]
    y = df["target"].iloc[:300]
    return X, y


@pytest.fixture
def test_data(sample_feature_data, feature_columns):
    """Prepare test data."""
    df = sample_feature_data.copy()
    X = df[feature_columns].iloc[300:]
    y = df["target"].iloc[300:]
    return X, y


class TestTrainLinear:
    """Tests for train_linear function."""
    
    def test_returns_model(self, training_data):
        """Test that a model is returned."""
        X, y = training_data
        model = train_linear(X, y)
        
        assert model is not None
        assert hasattr(model, "predict")
    
    def test_has_coefficients(self, training_data):
        """Test that model has coefficients."""
        X, y = training_data
        model = train_linear(X, y)
        
        assert hasattr(model, "coef_")
        assert len(model.coef_) == X.shape[1]
    
    def test_can_predict(self, training_data, test_data):
        """Test that model can make predictions."""
        X_train, y_train = training_data
        X_test, _ = test_data
        
        model = train_linear(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(X_test)


class TestTrainRandomForest:
    """Tests for train_random_forest function."""
    
    def test_returns_model(self, training_data):
        """Test that a model is returned."""
        X, y = training_data
        model = train_random_forest(X, y)
        
        assert model is not None
        assert hasattr(model, "predict")
    
    def test_has_feature_importances(self, training_data):
        """Test that model has feature importances."""
        X, y = training_data
        model = train_random_forest(X, y)
        
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]
    
    def test_can_predict(self, training_data, test_data):
        """Test that model can make predictions."""
        X_train, y_train = training_data
        X_test, _ = test_data
        
        model = train_random_forest(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(X_test)
    
    def test_deterministic_with_seed(self, training_data):
        """Test that training is deterministic with same seed."""
        X, y = training_data
        
        model1 = train_random_forest(X, y)
        model2 = train_random_forest(X, y)
        
        # Feature importances should be identical
        np.testing.assert_array_almost_equal(
            model1.feature_importances_,
            model2.feature_importances_
        )


class TestTrainXGBoost:
    """Tests for train_xgboost function."""
    
    def test_returns_model(self, training_data):
        """Test that a model is returned."""
        X, y = training_data
        model = train_xgboost(X, y)
        
        assert model is not None
        assert hasattr(model, "predict")
    
    def test_has_feature_importances(self, training_data):
        """Test that model has feature importances."""
        X, y = training_data
        model = train_xgboost(X, y)
        
        assert hasattr(model, "feature_importances_")
    
    def test_can_predict(self, training_data, test_data):
        """Test that model can make predictions."""
        X_train, y_train = training_data
        X_test, _ = test_data
        
        model = train_xgboost(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(X_test)


class TestPredict:
    """Tests for predict function."""
    
    def test_returns_numpy_array(self, training_data, test_data):
        """Test that predictions are numpy array."""
        X_train, y_train = training_data
        X_test, _ = test_data
        
        model = train_linear(X_train, y_train)
        preds = predict(model, X_test)
        
        assert isinstance(preds, np.ndarray)
    
    def test_correct_shape(self, training_data, test_data):
        """Test that predictions have correct shape."""
        X_train, y_train = training_data
        X_test, _ = test_data
        
        model = train_random_forest(X_train, y_train)
        preds = predict(model, X_test)
        
        assert preds.shape == (len(X_test),)
    
    def test_all_models_produce_predictions(self, training_data, test_data):
        """Test that all model types produce predictions."""
        X_train, y_train = training_data
        X_test, _ = test_data
        
        models = [
            train_linear(X_train, y_train),
            train_random_forest(X_train, y_train),
            train_xgboost(X_train, y_train),
        ]
        
        for model in models:
            preds = predict(model, X_test)
            assert len(preds) == len(X_test)
            assert not np.isnan(preds).any()


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""
    
    def test_returns_list_of_tuples(self, training_data):
        """Test that result is list of tuples."""
        X, y = training_data
        model = train_random_forest(X, y)
        
        importances = get_feature_importance(model, X.columns)
        
        assert isinstance(importances, list)
        assert all(isinstance(item, tuple) for item in importances)
    
    def test_all_features_included(self, training_data, feature_columns):
        """Test that all features are in result."""
        X, y = training_data
        model = train_random_forest(X, y)
        
        importances = get_feature_importance(model, X.columns)
        feature_names = [name for name, _ in importances]
        
        assert set(feature_names) == set(feature_columns)
    
    def test_sorted_descending(self, training_data):
        """Test that importances are sorted descending."""
        X, y = training_data
        model = train_random_forest(X, y)
        
        importances = get_feature_importance(model, X.columns)
        values = [val for _, val in importances]
        
        assert values == sorted(values, reverse=True)
    
    def test_works_with_linear_model(self, training_data):
        """Test that it works with linear model coefficients."""
        X, y = training_data
        model = train_linear(X, y)
        
        importances = get_feature_importance(model, X.columns)
        
        assert len(importances) == X.shape[1]
    
    def test_works_with_xgboost(self, training_data):
        """Test that it works with XGBoost."""
        X, y = training_data
        model = train_xgboost(X, y)
        
        importances = get_feature_importance(model, X.columns)
        
        assert len(importances) == X.shape[1]
