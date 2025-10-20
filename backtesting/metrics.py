"""Performance metrics for backtesting."""

import pandas as pd
import numpy as np
from typing import Optional


class PerformanceMetrics:
    """Calculate performance metrics for backtesting."""
    
    def __init__(
        self,
        results: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """Initialize performance metrics calculator.
        
        Args:
            results: DataFrame with backtest results (must have 'equity' or 'net_returns')
            risk_free_rate: Annual risk-free rate (as decimal)
            periods_per_year: Number of trading periods per year
        """
        self.results = results
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # Extract returns
        if "net_returns" in results.columns:
            self.returns = results["net_returns"]
        elif "equity" in results.columns:
            self.returns = results["equity"].pct_change().fillna(0)
        else:
            raise ValueError("Results must have 'net_returns' or 'equity' column")
    
    def total_return(self) -> float:
        """Calculate total return.
        
        Returns:
            Total return as decimal
        """
        if "equity" in self.results.columns:
            initial = self.results["equity"].iloc[0]
            final = self.results["equity"].iloc[-1]
            return (final - initial) / initial
        else:
            return (1 + self.returns).prod() - 1
    
    def cagr(self) -> float:
        """Calculate Compound Annual Growth Rate.
        
        Returns:
            CAGR as decimal
        """
        total_return = self.total_return()
        n_periods = len(self.returns)
        n_years = n_periods / self.periods_per_year
        
        if n_years == 0:
            return 0.0
        
        return (1 + total_return) ** (1 / n_years) - 1
    
    def volatility(self, annualized: bool = True) -> float:
        """Calculate volatility (standard deviation of returns).
        
        Args:
            annualized: Whether to annualize the volatility
            
        Returns:
            Volatility as decimal
        """
        vol = self.returns.std()
        
        if annualized:
            vol *= np.sqrt(self.periods_per_year)
        
        return vol
    
    def sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sharpe Ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (uses instance default if None)
            
        Returns:
            Sharpe Ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Convert annual risk-free rate to per-period
        rf_per_period = risk_free_rate / self.periods_per_year
        
        excess_returns = self.returns - rf_per_period
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std()
        
        # Annualize
        return sharpe * np.sqrt(self.periods_per_year)
    
    def sortino_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sortino Ratio (uses downside deviation).
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino Ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        rf_per_period = risk_free_rate / self.periods_per_year
        excess_returns = self.returns - rf_per_period
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_deviation = downside_returns.std()
        
        sortino = excess_returns.mean() / downside_deviation
        
        # Annualize
        return sortino * np.sqrt(self.periods_per_year)
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as negative decimal (e.g., -0.25 for 25% drawdown)
        """
        if "equity" in self.results.columns:
            equity = self.results["equity"]
        else:
            equity = (1 + self.returns).cumprod()
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        return drawdown.min()
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar Ratio (CAGR / |max_drawdown|).
        
        Returns:
            Calmar Ratio
        """
        cagr = self.cagr()
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        return cagr / max_dd
    
    def win_rate(self) -> float:
        """Calculate win rate (percentage of profitable trades).
        
        Returns:
            Win rate as decimal
        """
        winning_periods = (self.returns > 0).sum()
        total_periods = len(self.returns[self.returns != 0])
        
        if total_periods == 0:
            return 0.0
        
        return winning_periods / total_periods
    
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss).
        
        Returns:
            Profit factor
        """
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        
        return gains / losses
    
    def num_trades(self) -> int:
        """Calculate number of trades.
        
        Returns:
            Number of trades (non-zero return periods)
        """
        if "position" in self.results.columns:
            # Count position changes
            position_changes = self.results["position"].diff().fillna(0)
            return (position_changes != 0).sum()
        else:
            # Count non-zero returns
            return (self.returns != 0).sum()
    
    def max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning periods.
        
        Returns:
            Maximum consecutive wins
        """
        wins = (self.returns > 0).astype(int)
        
        if wins.sum() == 0:
            return 0
        
        # Find consecutive wins
        win_streaks = wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)
        
        return win_streaks.max()
    
    def max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing periods.
        
        Returns:
            Maximum consecutive losses
        """
        losses = (self.returns < 0).astype(int)
        
        if losses.sum() == 0:
            return 0
        
        # Find consecutive losses
        loss_streaks = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)
        
        return loss_streaks.max()
    
    def exposure(self) -> float:
        """Calculate market exposure (percentage of time in market).
        
        Returns:
            Exposure as decimal
        """
        if "position" in self.results.columns:
            in_market = (self.results["position"] != 0).sum()
            total_periods = len(self.results)
            return in_market / total_periods
        else:
            return 1.0  # Assume fully invested if no position data
    
    def average_win(self) -> float:
        """Calculate average winning return.
        
        Returns:
            Average win
        """
        wins = self.returns[self.returns > 0]
        
        if len(wins) == 0:
            return 0.0
        
        return wins.mean()
    
    def average_loss(self) -> float:
        """Calculate average losing return.
        
        Returns:
            Average loss
        """
        losses = self.returns[self.returns < 0]
        
        if len(losses) == 0:
            return 0.0
        
        return losses.mean()
    
    def get_all_metrics(self) -> dict:
        """Get all performance metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "total_return": self.total_return(),
            "cagr": self.cagr(),
            "volatility": self.volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "calmar_ratio": self.calmar_ratio(),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "num_trades": self.num_trades(),
            "max_consecutive_wins": self.max_consecutive_wins(),
            "max_consecutive_losses": self.max_consecutive_losses(),
            "exposure": self.exposure(),
            "average_win": self.average_win(),
            "average_loss": self.average_loss(),
        }
    
    def print_summary(self) -> None:
        """Print performance metrics summary."""
        metrics = self.get_all_metrics()
        
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Total Return:        {metrics['total_return']:>10.2%}")
        print(f"CAGR:                {metrics['cagr']:>10.2%}")
        print(f"Volatility:          {metrics['volatility']:>10.2%}")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"Number of Trades:    {metrics['num_trades']:>10}")
        print(f"Market Exposure:     {metrics['exposure']:>10.2%}")
        print(f"Average Win:         {metrics['average_win']:>10.2%}")
        print(f"Average Loss:        {metrics['average_loss']:>10.2%}")
        print("="*50 + "\n")
