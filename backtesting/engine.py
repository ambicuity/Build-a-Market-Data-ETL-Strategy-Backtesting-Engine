"""Backtesting engine for strategy testing."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from .strategy import BaseStrategy
from .portfolio import VectorizedPortfolio
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """High-performance vectorized backtesting engine."""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize backtest engine.
        
        Args:
            strategy: Trading strategy instance
            initial_cash: Initial portfolio value
            commission: Commission rate (as decimal)
            slippage: Slippage rate (as decimal)
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        self.portfolio = VectorizedPortfolio(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
        )
        
        self.results: Optional[pd.DataFrame] = None
        self.metrics: Optional[PerformanceMetrics] = None
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest.
        
        Returns:
            Dictionary containing backtest results and metrics
        """
        logger.info(f"Running backtest for strategy: {self.strategy.name}")
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # Get prices
        if "close" not in self.strategy.data.columns:
            raise ValueError("Strategy data must have 'close' column")
        
        prices = self.strategy.data["close"]
        
        # Run vectorized backtest
        self.results = self.portfolio.backtest_signals(
            prices=prices,
            signals=signals,
        )
        
        # Calculate performance metrics
        self.metrics = PerformanceMetrics(self.results)
        
        logger.info(f"Backtest complete. Final equity: ${self.results['equity'].iloc[-1]:,.2f}")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get backtest summary.
        
        Returns:
            Dictionary with results and metrics
        """
        if self.results is None or self.metrics is None:
            raise RuntimeError("Backtest has not been run yet")
        
        return {
            "strategy": self.strategy.name,
            "initial_cash": self.initial_cash,
            "final_equity": self.results["equity"].iloc[-1],
            "total_return": self.metrics.total_return(),
            "sharpe_ratio": self.metrics.sharpe_ratio(),
            "sortino_ratio": self.metrics.sortino_ratio(),
            "max_drawdown": self.metrics.max_drawdown(),
            "calmar_ratio": self.metrics.calmar_ratio(),
            "cagr": self.metrics.cagr(),
            "volatility": self.metrics.volatility(),
            "win_rate": self.metrics.win_rate(),
            "profit_factor": self.metrics.profit_factor(),
            "num_trades": self.metrics.num_trades(),
        }
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve.
        
        Returns:
            Series with equity values over time
        """
        if self.results is None:
            raise RuntimeError("Backtest has not been run yet")
        
        return self.results["equity"]
    
    def get_returns(self) -> pd.Series:
        """Get returns series.
        
        Returns:
            Series with returns
        """
        if self.results is None:
            raise RuntimeError("Backtest has not been run yet")
        
        return self.results["net_returns"]
    
    def get_positions(self) -> pd.Series:
        """Get positions over time.
        
        Returns:
            Series with position sizes
        """
        if self.results is None:
            raise RuntimeError("Backtest has not been run yet")
        
        return self.results["position"]


class EventDrivenEngine:
    """Event-driven backtesting engine for more realistic simulation."""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize event-driven engine.
        
        Args:
            strategy: Trading strategy instance
            initial_cash: Initial portfolio value
            commission: Commission rate
            slippage: Slippage rate
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        from .portfolio import Portfolio
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
        )
        
        self.results = []
    
    def run(self) -> Dict[str, Any]:
        """Run event-driven backtest.
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running event-driven backtest: {self.strategy.name}")
        
        self.portfolio.reset()
        signals = self.strategy.generate_signals()
        data = self.strategy.data
        
        current_position = 0
        equity_history = []
        
        for idx, (timestamp, row) in enumerate(data.iterrows()):
            price = row["close"]
            signal = signals.iloc[idx] if idx < len(signals) else 0
            
            # Calculate desired position
            desired_position = signal
            
            # Calculate trade size
            trade_size = desired_position - current_position
            
            # Execute trade if needed
            if trade_size != 0:
                # Simple position sizing: use 100 shares per signal
                shares = trade_size * 100
                
                if self.portfolio.execute_trade(
                    symbol="ASSET",
                    quantity=shares,
                    price=price,
                    timestamp=timestamp,
                ):
                    current_position = desired_position
            
            # Record portfolio value
            portfolio_value = self.portfolio.calculate_portfolio_value({"ASSET": price * current_position * 100})
            equity_history.append({
                "timestamp": timestamp,
                "equity": portfolio_value,
                "position": current_position,
                "price": price,
            })
        
        results_df = pd.DataFrame(equity_history)
        results_df.set_index("timestamp", inplace=True)
        
        self.results = results_df
        
        # Calculate metrics
        metrics = PerformanceMetrics(results_df)
        
        return {
            "strategy": self.strategy.name,
            "initial_cash": self.initial_cash,
            "final_equity": results_df["equity"].iloc[-1],
            "total_return": metrics.total_return(),
            "sharpe_ratio": metrics.sharpe_ratio(),
            "max_drawdown": metrics.max_drawdown(),
            "num_trades": len(self.portfolio.history),
        }
    
    def get_results(self) -> pd.DataFrame:
        """Get backtest results.
        
        Returns:
            DataFrame with backtest results
        """
        if not isinstance(self.results, pd.DataFrame):
            raise RuntimeError("Backtest has not been run yet")
        
        return self.results
