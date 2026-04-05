"""
Tests for ml_backtester module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import train_random_forest, predict


class MockModel:
    """Mock model for testing backtester without actual ML."""
    
    def __init__(self, predictions):
        self.predictions = predictions
        self._call_count = 0
    
    def predict(self, X):
        return self.predictions[:len(X)]


@pytest.fixture
def backtest_data(sample_feature_data, feature_columns):
    """Prepare data for backtesting."""
    df = sample_feature_data.copy()
    X = df[feature_columns].iloc[300:350]
    y = df["target"].iloc[300:350]
    prices = df["Close"].iloc[300:350]
    return X, y, prices


class TestMLBacktesterBasics:
    """Basic tests for MLBacktester class."""
    
    def test_initialization(self):
        """Test backtester can be initialized."""
        from ml_backtester import MLBacktester
        
        mock_model = MockModel(np.array([0.01] * 50))
        backtester = MLBacktester(
            model=mock_model,
            starting_cash=100000,
            fee_rate=0.001
        )
        
        assert backtester.starting_cash == 100000
        assert backtester.fee_rate == 0.001
    
    def test_reset(self):
        """Test reset functionality."""
        from ml_backtester import MLBacktester
        
        mock_model = MockModel(np.array([0.01] * 50))
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        
        # Modify state
        backtester.cash = 50000
        backtester.shares = 100
        
        # Reset
        backtester.reset()
        
        assert backtester.cash == 100000
        assert backtester.shares == 0
        assert not backtester.in_position
    
    def test_run_returns_results(self, backtest_data):
        """Test that run returns results dictionary."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        mock_model = MockModel(np.array([0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        results = backtester.run(X, y, prices)
        
        assert isinstance(results, dict)


class TestTradingLogic:
    """Tests for trading logic."""
    
    def test_buys_on_positive_prediction(self, backtest_data):
        """Test that backtester buys when prediction > 0."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        # All positive predictions = should buy immediately
        mock_model = MockModel(np.array([0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        # Should have at least one BUY trade
        buy_trades = [t for t in backtester.trade_log if t["action"] == "BUY"]
        assert len(buy_trades) >= 1
    
    def test_sells_on_negative_prediction(self, backtest_data):
        """Test that backtester sells when prediction <= 0."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        # Positive then negative = buy then sell
        preds = np.array([0.01] * 10 + [-0.01] * (len(X) - 10))
        mock_model = MockModel(preds)
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        # Should have sell trades
        sell_trades = [t for t in backtester.trade_log if "SELL" in t["action"]]
        assert len(sell_trades) >= 1
    
    def test_no_trade_when_already_in_position(self, backtest_data):
        """Test no duplicate buys when already in position."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        # All positive = would try to buy multiple times
        mock_model = MockModel(np.array([0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        # Should only have 1 BUY (and 1 FINAL SELL at end)
        buy_trades = [t for t in backtester.trade_log if t["action"] == "BUY"]
        assert len(buy_trades) == 1
    
    def test_fees_deducted(self, backtest_data):
        """Test that fees are deducted from trades."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        preds = np.array([0.01] * 10 + [-0.01] * (len(X) - 10))
        mock_model = MockModel(preds)
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000, fee_rate=0.001)
        backtester.run(X, y, prices)
        
        assert backtester.total_fees_paid > 0
    
    def test_liquidates_at_end(self, backtest_data):
        """Test that position is liquidated at end if still holding."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        # All positive = hold until end
        mock_model = MockModel(np.array([0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        # Should have FINAL SELL at end
        final_sells = [t for t in backtester.trade_log if t["action"] == "FINAL SELL"]
        assert len(final_sells) == 1


class TestMetricsCalculation:
    """Tests for metrics calculation."""
    
    def test_portfolio_history_recorded(self, backtest_data):
        """Test that portfolio history is recorded."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        mock_model = MockModel(np.array([0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        assert len(backtester.portfolio_history) == len(X)
    
    def test_benchmark_history_recorded(self, backtest_data):
        """Test that benchmark history is recorded."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        mock_model = MockModel(np.array([0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        assert len(backtester.benchmark_history) == len(X)
    
    def test_trade_count_correct(self, backtest_data):
        """Test that trade count is tracked correctly."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        # Multiple buy/sell cycles
        preds = np.array(
            [0.01] * 10 + [-0.01] * 10 + [0.01] * 10 + [-0.01] * (len(X) - 30)
        )
        mock_model = MockModel(preds)
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        assert backtester.total_trades == len(backtester.trade_log)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_all_negative_predictions(self, backtest_data):
        """Test behavior with all negative predictions (no trades)."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        mock_model = MockModel(np.array([-0.01] * len(X)))
        
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        backtester.run(X, y, prices)
        
        # No trades should occur
        assert len(backtester.trade_log) == 0
        assert backtester.cash == 100000
    
    def test_mismatched_indices_raises_error(self, backtest_data):
        """Test that mismatched indices raise error."""
        from ml_backtester import MLBacktester
        
        X, y, prices = backtest_data
        # Create prices with different index
        wrong_prices = pd.Series(prices.values, index=range(len(prices)))
        
        mock_model = MockModel(np.array([0.01] * len(X)))
        backtester = MLBacktester(model=mock_model, starting_cash=100000)
        
        with pytest.raises(ValueError, match="matching indices"):
            backtester.run(X, y, wrong_prices)
