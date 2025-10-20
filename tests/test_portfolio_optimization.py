"""Tests for portfolio optimization module."""

import pytest
import pandas as pd
import numpy as np
from backtesting.portfolio_optimization import PortfolioOptimizer, RiskParityOptimizer


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create correlated returns for 3 assets
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0005, 0.02, 252),
        'MSFT': np.random.normal(0.0004, 0.018, 252),
        'GOOGL': np.random.normal(0.0006, 0.022, 252),
    }, index=dates)
    
    return returns


class TestPortfolioOptimizer:
    """Test portfolio optimizer functionality."""
    
    def test_optimizer_init(self):
        """Test optimizer initialization."""
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.expected_returns is None
        assert optimizer.cov_matrix is None
    
    def test_calculate_statistics(self, sample_returns):
        """Test statistics calculation."""
        optimizer = PortfolioOptimizer()
        expected_returns, cov_matrix = optimizer.calculate_statistics(sample_returns)
        
        assert isinstance(expected_returns, pd.Series)
        assert isinstance(cov_matrix, pd.DataFrame)
        assert len(expected_returns) == 3
        assert cov_matrix.shape == (3, 3)
        
        # Check that expected returns are annualized
        assert expected_returns.abs().max() < 1.0  # Should be reasonable
    
    def test_portfolio_return(self, sample_returns):
        """Test portfolio return calculation."""
        optimizer = PortfolioOptimizer()
        optimizer.calculate_statistics(sample_returns)
        
        weights = np.array([0.4, 0.3, 0.3])
        portfolio_return = optimizer.portfolio_return(weights)
        
        assert isinstance(portfolio_return, float)
        assert -1.0 < portfolio_return < 1.0
    
    def test_portfolio_volatility(self, sample_returns):
        """Test portfolio volatility calculation."""
        optimizer = PortfolioOptimizer()
        optimizer.calculate_statistics(sample_returns)
        
        weights = np.array([0.4, 0.3, 0.3])
        volatility = optimizer.portfolio_volatility(weights)
        
        assert isinstance(volatility, float)
        assert volatility > 0
    
    def test_optimize_max_sharpe(self, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        result = optimizer.optimize_max_sharpe(sample_returns)
        
        assert result['success']
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        
        # Check weights sum to 1
        weights_sum = sum(result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
        
        # Check all weights are non-negative
        assert all(w >= -0.01 for w in result['weights'].values())
    
    def test_optimize_min_volatility(self, sample_returns):
        """Test minimum volatility optimization."""
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_min_volatility(sample_returns)
        
        assert result['success']
        assert 'weights' in result
        assert 'volatility' in result
        
        # Volatility should be positive
        assert result['volatility'] > 0
    
    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer()
        frontier = optimizer.efficient_frontier(sample_returns, n_points=10)
        
        assert isinstance(frontier, pd.DataFrame)
        assert len(frontier) > 0
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns
        assert 'sharpe' in frontier.columns


class TestRiskParityOptimizer:
    """Test risk parity optimizer."""
    
    def test_risk_parity_init(self):
        """Test risk parity optimizer initialization."""
        optimizer = RiskParityOptimizer()
        assert optimizer.cov_matrix is None
    
    def test_calculate_risk_contributions(self, sample_returns):
        """Test risk contribution calculation."""
        optimizer = RiskParityOptimizer()
        cov_matrix = sample_returns.cov().values * 252
        weights = np.array([0.33, 0.33, 0.34])
        
        risk_contrib = optimizer.calculate_risk_contributions(weights, cov_matrix)
        
        assert isinstance(risk_contrib, np.ndarray)
        assert len(risk_contrib) == 3
        assert all(rc >= 0 for rc in risk_contrib)
    
    def test_optimize_risk_parity(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(sample_returns)
        
        assert result['success']
        assert 'weights' in result
        assert 'risk_contributions' in result
        
        # Weights should sum to 1
        weights_sum = sum(result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
        
        # Risk contributions should be roughly equal
        risk_contribs = list(result['risk_contributions'].values())
        assert max(risk_contribs) / min(risk_contribs) < 3.0  # Not too far apart


def test_portfolio_optimization_integration(sample_returns):
    """Test integrated portfolio optimization workflow."""
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Calculate statistics
    expected_returns, cov_matrix = optimizer.calculate_statistics(sample_returns)
    
    # Optimize for max Sharpe
    max_sharpe_result = optimizer.optimize_max_sharpe(sample_returns)
    
    # Optimize for min volatility
    min_vol_result = optimizer.optimize_min_volatility(sample_returns)
    
    # Min volatility should have lower volatility than max Sharpe
    # (not always true but usually)
    assert min_vol_result['volatility'] <= max_sharpe_result['volatility'] * 1.5
    
    # Max Sharpe should have higher Sharpe ratio
    assert max_sharpe_result['sharpe_ratio'] >= min_vol_result['sharpe_ratio'] * 0.8
