"""Tests for risk monitoring module."""

import pytest
import pandas as pd
import numpy as np
from backtesting.risk_monitor import (
    RiskMonitor, PositionSizer, StopLossManager,
    RiskLevel
)


@pytest.fixture
def sample_equity_curve():
    """Create sample equity curve for testing."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create equity curve with some drawdown
    values = [100000]
    for _ in range(251):
        change = np.random.normal(0.0005, 0.01)
        values.append(values[-1] * (1 + change))
    
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    return returns


class TestRiskMonitor:
    """Test RiskMonitor functionality."""
    
    def test_monitor_init(self):
        """Test risk monitor initialization."""
        monitor = RiskMonitor(
            max_drawdown_threshold=0.20,
            var_confidence=0.95,
            position_limit=0.25,
            daily_loss_limit=0.05
        )
        
        assert monitor.max_drawdown_threshold == 0.20
        assert monitor.var_confidence == 0.95
        assert monitor.position_limit == 0.25
        assert monitor.daily_loss_limit == 0.05
        assert len(monitor.alerts) == 0
    
    def test_calculate_var(self, sample_returns):
        """Test VaR calculation."""
        monitor = RiskMonitor(var_confidence=0.95)
        var = monitor.calculate_var(sample_returns)
        
        assert isinstance(var, float)
        # VaR should be negative (loss)
        assert var <= 0
    
    def test_calculate_cvar(self, sample_returns):
        """Test CVaR calculation."""
        monitor = RiskMonitor(var_confidence=0.95)
        cvar = monitor.calculate_cvar(sample_returns)
        
        assert isinstance(cvar, float)
        # CVaR should be more negative than VaR
        var = monitor.calculate_var(sample_returns)
        assert cvar <= var
    
    def test_calculate_drawdown(self, sample_equity_curve):
        """Test drawdown calculation."""
        monitor = RiskMonitor()
        drawdown = monitor.calculate_drawdown(sample_equity_curve)
        
        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(sample_equity_curve)
        # All drawdowns should be <= 0
        assert (drawdown <= 0).all()
    
    def test_calculate_max_drawdown(self, sample_equity_curve):
        """Test maximum drawdown calculation."""
        monitor = RiskMonitor()
        max_dd = monitor.calculate_max_drawdown(sample_equity_curve)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0
    
    def test_check_position_limits_pass(self):
        """Test position limit check - within limits."""
        monitor = RiskMonitor(position_limit=0.25)
        positions = {"AAPL": 20000, "MSFT": 15000}
        portfolio_value = 100000
        
        alerts = monitor.check_position_limits(positions, portfolio_value)
        
        assert len(alerts) == 0
    
    def test_check_position_limits_fail(self):
        """Test position limit check - exceeds limits."""
        monitor = RiskMonitor(position_limit=0.25)
        positions = {"AAPL": 30000, "MSFT": 15000}
        portfolio_value = 100000
        
        alerts = monitor.check_position_limits(positions, portfolio_value)
        
        assert len(alerts) > 0
        assert alerts[0].metric == "position_size"
        assert alerts[0].level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_check_drawdown_pass(self, sample_equity_curve):
        """Test drawdown check - within threshold."""
        monitor = RiskMonitor(max_drawdown_threshold=0.50)
        alert = monitor.check_drawdown(sample_equity_curve)
        
        # With 50% threshold, should likely pass
        assert alert is None or alert.level != RiskLevel.CRITICAL
    
    def test_check_drawdown_fail(self):
        """Test drawdown check - exceeds threshold."""
        monitor = RiskMonitor(max_drawdown_threshold=0.05)
        
        # Create equity curve with significant drawdown
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = list(range(100000, 80000, -200))
        equity_curve = pd.Series(values, index=dates)
        
        alert = monitor.check_drawdown(equity_curve)
        
        assert alert is not None
        assert alert.metric == "max_drawdown"
        assert alert.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_check_daily_loss_pass(self):
        """Test daily loss check - within limit."""
        monitor = RiskMonitor(daily_loss_limit=0.05)
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02, 0.01, -0.015, 0.005, 0.01, -0.01], index=dates)
        
        alert = monitor.check_daily_loss(returns)
        assert alert is None
    
    def test_check_daily_loss_fail(self):
        """Test daily loss check - exceeds limit."""
        monitor = RiskMonitor(daily_loss_limit=0.05)
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02, 0.01, -0.015, 0.005, 0.01, -0.08], index=dates)
        
        alert = monitor.check_daily_loss(returns)
        
        assert alert is not None
        assert alert.metric == "daily_loss"
        assert alert.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_monitor_comprehensive(self, sample_equity_curve, sample_returns):
        """Test comprehensive monitoring."""
        monitor = RiskMonitor()
        positions = {"AAPL": 20000, "MSFT": 15000}
        
        alerts = monitor.monitor(sample_equity_curve, positions, sample_returns)
        
        assert isinstance(alerts, list)
        assert len(monitor.metrics_history) > 0
        
        # Check metrics history
        latest_metrics = monitor.metrics_history[-1]
        assert 'var' in latest_metrics
        assert 'cvar' in latest_metrics
        assert 'max_drawdown' in latest_metrics


class TestPositionSizer:
    """Test PositionSizer functionality."""
    
    def test_sizer_init(self):
        """Test position sizer initialization."""
        sizer = PositionSizer(risk_per_trade=0.02, max_position_size=0.25)
        
        assert sizer.risk_per_trade == 0.02
        assert sizer.max_position_size == 0.25
    
    def test_kelly_criterion(self):
        """Test Kelly criterion calculation."""
        sizer = PositionSizer()
        
        win_rate = 0.6
        avg_win = 0.05
        avg_loss = 0.03
        
        kelly = sizer.kelly_criterion(win_rate, avg_win, avg_loss)
        
        assert isinstance(kelly, float)
        assert 0 <= kelly <= sizer.max_position_size
    
    def test_volatility_based_sizing(self):
        """Test volatility-based position sizing."""
        sizer = PositionSizer(max_position_size=0.30)
        
        portfolio_value = 100000
        asset_volatility = 0.20
        target_volatility = 0.15
        
        position_size = sizer.volatility_based_sizing(
            portfolio_value, asset_volatility, target_volatility
        )
        
        assert isinstance(position_size, float)
        assert position_size > 0
        assert position_size <= portfolio_value * sizer.max_position_size
    
    def test_fixed_risk_sizing(self):
        """Test fixed risk position sizing."""
        sizer = PositionSizer(risk_per_trade=0.02, max_position_size=0.25)
        
        portfolio_value = 100000
        entry_price = 150.0
        stop_loss_price = 145.0
        
        position_size = sizer.fixed_risk_sizing(
            portfolio_value, entry_price, stop_loss_price
        )
        
        assert isinstance(position_size, float)
        assert position_size > 0
        
        # Check that max position size is respected
        max_shares = (portfolio_value * sizer.max_position_size) / entry_price
        assert position_size <= max_shares


class TestStopLossManager:
    """Test StopLossManager functionality."""
    
    def test_manager_init(self):
        """Test stop loss manager initialization."""
        manager = StopLossManager(initial_stop_pct=0.05, trailing_stop_pct=0.03)
        
        assert manager.initial_stop_pct == 0.05
        assert manager.trailing_stop_pct == 0.03
        assert len(manager.stops) == 0
    
    def test_set_initial_stop_long(self):
        """Test setting initial stop for long position."""
        manager = StopLossManager(initial_stop_pct=0.05)
        
        stop_price = manager.set_initial_stop("AAPL", 100.0, side="long")
        
        assert stop_price == 95.0  # 5% below entry
        assert "AAPL" in manager.stops
        assert manager.stops["AAPL"] == 95.0
    
    def test_set_initial_stop_short(self):
        """Test setting initial stop for short position."""
        manager = StopLossManager(initial_stop_pct=0.05)
        
        stop_price = manager.set_initial_stop("AAPL", 100.0, side="short")
        
        assert stop_price == 105.0  # 5% above entry
        assert "AAPL" in manager.stops
    
    def test_update_trailing_stop_long(self):
        """Test updating trailing stop for long position."""
        manager = StopLossManager(initial_stop_pct=0.05, trailing_stop_pct=0.03)
        
        # Set initial stop
        manager.set_initial_stop("AAPL", 100.0, side="long")
        
        # Price moves up
        new_stop = manager.update_trailing_stop("AAPL", 110.0, side="long")
        
        assert new_stop is not None
        assert new_stop > 95.0  # Should move up
        assert new_stop == 110.0 * 0.97  # 3% trailing stop
    
    def test_trailing_stop_doesnt_move_down(self):
        """Test that trailing stop doesn't move down for long."""
        manager = StopLossManager(initial_stop_pct=0.05, trailing_stop_pct=0.03)
        
        manager.set_initial_stop("AAPL", 100.0, side="long")
        manager.update_trailing_stop("AAPL", 110.0, side="long")
        
        # Price moves down - stop shouldn't move
        new_stop = manager.update_trailing_stop("AAPL", 105.0, side="long")
        
        assert new_stop is None  # Stop didn't move
        assert manager.stops["AAPL"] == 110.0 * 0.97
    
    def test_check_stop_hit_long(self):
        """Test checking if stop is hit for long position."""
        manager = StopLossManager()
        
        manager.set_initial_stop("AAPL", 100.0, side="long")
        
        # Price above stop - not hit
        assert not manager.check_stop_hit("AAPL", 96.0, side="long")
        
        # Price at or below stop - hit
        assert manager.check_stop_hit("AAPL", 95.0, side="long")
        assert manager.check_stop_hit("AAPL", 94.0, side="long")
    
    def test_remove_stop(self):
        """Test removing a stop."""
        manager = StopLossManager()
        
        manager.set_initial_stop("AAPL", 100.0, side="long")
        assert "AAPL" in manager.stops
        
        manager.remove_stop("AAPL")
        assert "AAPL" not in manager.stops


def test_risk_management_integration():
    """Test integrated risk management workflow."""
    # Create monitor and sizer
    monitor = RiskMonitor(
        max_drawdown_threshold=0.20,
        position_limit=0.25,
        daily_loss_limit=0.05
    )
    sizer = PositionSizer(risk_per_trade=0.02)
    stop_manager = StopLossManager(initial_stop_pct=0.05)
    
    # Simulate portfolio
    portfolio_value = 100000
    entry_price = 150.0
    stop_loss_price = 145.0
    
    # Calculate position size
    position_size = sizer.fixed_risk_sizing(portfolio_value, entry_price, stop_loss_price)
    position_value = position_size * entry_price
    
    # Set stop loss
    stop_price = stop_manager.set_initial_stop("AAPL", entry_price, side="long")
    
    # Check position limits
    positions = {"AAPL": position_value}
    alerts = monitor.check_position_limits(positions, portfolio_value)
    
    # Position should be within limits
    assert len(alerts) == 0
    
    # Check stop (stop price is at 150 * 0.95 = 142.5)
    assert not stop_manager.check_stop_hit("AAPL", 146.0, side="long")
    assert not stop_manager.check_stop_hit("AAPL", 143.0, side="long")
    assert stop_manager.check_stop_hit("AAPL", 142.0, side="long")
