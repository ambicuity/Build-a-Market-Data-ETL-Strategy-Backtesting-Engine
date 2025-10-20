"""Multi-asset portfolio optimization using modern portfolio theory."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Portfolio optimizer using mean-variance optimization."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.expected_returns: Optional[pd.Series] = None
        self.cov_matrix: Optional[pd.DataFrame] = None
    
    def calculate_statistics(
        self,
        returns_df: pd.DataFrame,
        method: str = "historical"
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate expected returns and covariance matrix.
        
        Args:
            returns_df: DataFrame with asset returns (columns = assets)
            method: Method for calculating expected returns ('historical', 'ewm')
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        if method == "historical":
            expected_returns = returns_df.mean() * 252  # Annualized
        elif method == "ewm":
            # Exponentially weighted moving average
            expected_returns = returns_df.ewm(span=60).mean().iloc[-1] * 252
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252
        
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        
        return expected_returns, cov_matrix
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return.
        
        Args:
            weights: Asset weights
            
        Returns:
            Expected portfolio return
        """
        return np.dot(weights, self.expected_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation).
        
        Args:
            weights: Asset weights
            
        Returns:
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio.
        
        Args:
            weights: Asset weights
            
        Returns:
            Sharpe ratio (negative for minimization)
        """
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return -(ret - self.risk_free_rate) / vol if vol > 0 else 0
    
    def optimize_max_sharpe(
        self,
        returns_df: pd.DataFrame,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            returns_df: DataFrame with asset returns
            bounds: Bounds for each asset weight (default: 0 to 1)
            constraints: Additional optimization constraints
            
        Returns:
            Dictionary with optimal weights and portfolio statistics
        """
        self.calculate_statistics(returns_df)
        n_assets = len(returns_df.columns)
        
        # Initial guess: equal weights
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Default bounds: 0 to 1 for each asset
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Default constraint: weights sum to 1
        if constraints is None:
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        # Optimize
        result = minimize(
            self.portfolio_sharpe,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        return {
            "weights": dict(zip(returns_df.columns, optimal_weights)),
            "expected_return": self.portfolio_return(optimal_weights),
            "volatility": self.portfolio_volatility(optimal_weights),
            "sharpe_ratio": -result.fun,
            "success": result.success
        }
    
    def optimize_min_volatility(
        self,
        returns_df: pd.DataFrame,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """Optimize portfolio for minimum volatility.
        
        Args:
            returns_df: DataFrame with asset returns
            bounds: Bounds for each asset weight
            constraints: Additional optimization constraints
            
        Returns:
            Dictionary with optimal weights and portfolio statistics
        """
        self.calculate_statistics(returns_df)
        n_assets = len(returns_df.columns)
        
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        if constraints is None:
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        # Optimize for minimum volatility
        result = minimize(
            self.portfolio_volatility,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        return {
            "weights": dict(zip(returns_df.columns, optimal_weights)),
            "expected_return": self.portfolio_return(optimal_weights),
            "volatility": result.fun,
            "sharpe_ratio": (self.portfolio_return(optimal_weights) - self.risk_free_rate) / result.fun,
            "success": result.success
        }
    
    def efficient_frontier(
        self,
        returns_df: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """Calculate efficient frontier.
        
        Args:
            returns_df: DataFrame with asset returns
            n_points: Number of points on the frontier
            
        Returns:
            DataFrame with efficient frontier points
        """
        self.calculate_statistics(returns_df)
        n_assets = len(returns_df.columns)
        
        # Get range of target returns
        min_vol_result = self.optimize_min_volatility(returns_df)
        max_sharpe_result = self.optimize_max_sharpe(returns_df)
        
        min_return = min_vol_result["expected_return"]
        max_return = self.expected_returns.max()
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_points = []
        
        for target_ret in target_returns:
            # Optimize for minimum volatility at target return
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x: self.portfolio_return(x) - target_ret}
            ]
            
            init_weights = np.array([1.0 / n_assets] * n_assets)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                frontier_points.append({
                    "return": target_ret,
                    "volatility": result.fun,
                    "sharpe": (target_ret - self.risk_free_rate) / result.fun if result.fun > 0 else 0
                })
        
        return pd.DataFrame(frontier_points)


class RiskParityOptimizer:
    """Risk parity portfolio optimization."""
    
    def __init__(self):
        """Initialize risk parity optimizer."""
        self.cov_matrix: Optional[pd.DataFrame] = None
    
    def calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contribution of each asset.
        
        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix
            
        Returns:
            Array of risk contributions
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib
    
    def risk_parity_objective(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Objective function for risk parity optimization.
        
        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix
            
        Returns:
            Objective value (sum of squared deviations from equal risk)
        """
        risk_contrib = self.calculate_risk_contributions(weights, cov_matrix)
        target_risk = 1.0 / len(weights)
        return np.sum((risk_contrib - target_risk) ** 2)
    
    def optimize(
        self,
        returns_df: pd.DataFrame
    ) -> Dict[str, any]:
        """Optimize portfolio for risk parity.
        
        Args:
            returns_df: DataFrame with asset returns
            
        Returns:
            Dictionary with optimal weights and risk contributions
        """
        # Calculate covariance matrix
        cov_matrix = returns_df.cov().values * 252  # Annualized
        self.cov_matrix = cov_matrix
        
        n_assets = len(returns_df.columns)
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        # Optimize
        result = minimize(
            self.risk_parity_objective,
            init_weights,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        risk_contrib = self.calculate_risk_contributions(optimal_weights, cov_matrix)
        
        return {
            "weights": dict(zip(returns_df.columns, optimal_weights)),
            "risk_contributions": dict(zip(returns_df.columns, risk_contrib)),
            "success": result.success
        }
