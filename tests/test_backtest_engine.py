"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting.strategy import BaseStrategy, BuyAndHoldStrategy, MeanReversionStrategy
from backtesting.engine import BacktestEngine
from backtesting.portfolio import Portfolio, VectorizedPortfolio
from backtesting.metrics import PerformanceMetrics


class TestStrategy:
    """Test strategy implementations."""
    
    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        
        # Create price series with upward trend
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": [1000] * 100,
        })
    
    def test_buy_and_hold_strategy(self):
        """Test buy and hold strategy."""
        strategy = BuyAndHoldStrategy(self.df)
        signals = strategy.generate_signals()
        
        assert len(signals) == len(self.df)
        assert (signals == 1).all()  # Should always be long
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy."""
        strategy = MeanReversionStrategy(self.df, window=20)
        signals = strategy.generate_signals()
        
        assert len(signals) == len(self.df)
        assert signals.isin([-1, 0, 1]).all()  # Should be -1, 0, or 1
    
    def test_strategy_data_preparation(self):
        """Test that strategy prepares data correctly."""
        strategy = BuyAndHoldStrategy(self.df)
        
        # Check that returns are calculated
        assert "returns" in strategy.data.columns
        
        # Check that timestamp is index
        assert isinstance(strategy.data.index, pd.DatetimeIndex)


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_init(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_cash=100000)
        
        assert portfolio.cash == 100000
        assert portfolio.initial_cash == 100000
        assert len(portfolio.positions) == 0
    
    def test_execute_trade_buy(self):
        """Test executing a buy trade."""
        portfolio = Portfolio(initial_cash=100000, commission=0.001)
        
        success = portfolio.execute_trade(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
        )
        
        assert success
        assert portfolio.positions["AAPL"] == 100
        assert portfolio.cash < 100000  # Cash reduced by cost + commission
    
    def test_execute_trade_sell(self):
        """Test executing a sell trade."""
        portfolio = Portfolio(initial_cash=100000)
        
        # First buy
        portfolio.execute_trade("AAPL", 100, 150.0)
        cash_after_buy = portfolio.cash
        
        # Then sell
        portfolio.execute_trade("AAPL", -50, 155.0)
        
        assert portfolio.positions["AAPL"] == 50
        assert portfolio.cash > cash_after_buy  # Cash increased from sale
    
    def test_insufficient_cash(self):
        """Test trade with insufficient cash."""
        portfolio = Portfolio(initial_cash=1000)
        
        success = portfolio.execute_trade(
            symbol="AAPL",
            quantity=100,
            price=150.0,
        )
        
        assert not success  # Should fail due to insufficient cash
        assert portfolio.cash == 1000  # Cash unchanged


class TestVectorizedPortfolio:
    """Test vectorized portfolio."""
    
    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        self.prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)
        self.signals = pd.Series([1] * 100, index=dates)  # Always long
    
    def test_backtest_signals(self):
        """Test backtesting with signals."""
        portfolio = VectorizedPortfolio(initial_cash=100000)
        
        results = portfolio.backtest_signals(
            prices=self.prices,
            signals=self.signals,
        )
        
        assert isinstance(results, pd.DataFrame)
        assert "equity" in results.columns
        assert "net_returns" in results.columns
        assert len(results) == len(self.prices)
    
    def test_costs_applied(self):
        """Test that costs are properly applied."""
        portfolio = VectorizedPortfolio(
            initial_cash=100000,
            commission=0.01,  # High commission for testing
            slippage=0.01,
        )
        
        results = portfolio.backtest_signals(
            prices=self.prices,
            signals=self.signals,
        )
        
        # Check that costs are non-zero
        assert results["costs"].sum() > 0


class TestBacktestEngine:
    """Test backtesting engine."""
    
    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.df = pd.DataFrame({
            "timestamp": dates,
            "close": prices,
            "volume": [1000] * 100,
        })
        
        self.strategy = BuyAndHoldStrategy(self.df)
    
    def test_engine_init(self):
        """Test engine initialization."""
        engine = BacktestEngine(self.strategy)
        
        assert engine.strategy == self.strategy
        assert engine.initial_cash == 1_000_000
    
    def test_run_backtest(self):
        """Test running a backtest."""
        engine = BacktestEngine(self.strategy, initial_cash=100000)
        
        summary = engine.run()
        
        assert "strategy" in summary
        assert "final_equity" in summary
        assert "total_return" in summary
        assert "sharpe_ratio" in summary
    
    def test_get_equity_curve(self):
        """Test getting equity curve."""
        engine = BacktestEngine(self.strategy)
        engine.run()
        
        equity_curve = engine.get_equity_curve()
        
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) > 0
    
    def test_get_returns(self):
        """Test getting returns."""
        engine = BacktestEngine(self.strategy)
        engine.run()
        
        returns = engine.get_returns()
        
        assert isinstance(returns, pd.Series)
        assert len(returns) > 0


class TestPerformanceMetrics:
    """Test performance metrics."""
    
    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        
        # Create sample backtest results
        np.random.seed(42)  # For reproducibility
        returns = np.random.randn(100) * 0.01  # 1% daily volatility
        equity = pd.Series(100000 * (1 + returns).cumprod(), index=dates)
        
        self.results = pd.DataFrame({
            "equity": equity,
            "net_returns": returns,
        }, index=dates)
        
        self.metrics = PerformanceMetrics(self.results)
    
    def test_total_return(self):
        """Test total return calculation."""
        total_return = self.metrics.total_return()
        
        assert isinstance(total_return, float)
        assert -1 <= total_return <= 10  # Reasonable range
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.metrics.sharpe_ratio()
        
        assert isinstance(sharpe, float)
        assert -10 <= sharpe <= 10  # Reasonable range
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        max_dd = self.metrics.max_drawdown()
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Should be negative
    
    def test_cagr(self):
        """Test CAGR calculation."""
        cagr = self.metrics.cagr()
        
        assert isinstance(cagr, float)
    
    def test_volatility(self):
        """Test volatility calculation."""
        vol = self.metrics.volatility()
        
        assert isinstance(vol, float)
        assert vol >= 0
    
    def test_win_rate(self):
        """Test win rate calculation."""
        win_rate = self.metrics.win_rate()
        
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        all_metrics = self.metrics.get_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert "total_return" in all_metrics
        assert "sharpe_ratio" in all_metrics
        assert "max_drawdown" in all_metrics


def test_import_backtesting_modules():
    """Test that all backtesting modules can be imported."""
    from backtesting import BaseStrategy, Portfolio, BacktestEngine, PerformanceMetrics, Visualizer
    
    assert BaseStrategy is not None
    assert Portfolio is not None
    assert BacktestEngine is not None
    assert PerformanceMetrics is not None
    assert Visualizer is not None
