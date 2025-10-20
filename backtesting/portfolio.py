"""Portfolio management for backtesting."""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class Portfolio:
    """Portfolio manager for backtesting."""
    
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize portfolio.
        
        Args:
            initial_cash: Initial cash balance
            commission: Commission rate (as decimal, e.g., 0.001 = 0.1%)
            slippage: Slippage rate (as decimal)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        # Portfolio state
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.history = []
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions = {}
        self.history = []
    
    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> bool:
        """Execute a trade.
        
        Args:
            symbol: Asset symbol
            quantity: Quantity to trade (positive = buy, negative = sell)
            price: Execution price
            timestamp: Timestamp of trade
            
        Returns:
            True if trade was executed, False otherwise
        """
        if quantity == 0:
            return False
        
        # Calculate costs
        trade_value = abs(quantity) * price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        total_cost = trade_value + commission_cost + slippage_cost
        
        # Check if we have enough cash for buys
        if quantity > 0 and self.cash < total_cost:
            return False
        
        # Update positions and cash
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        self.positions[symbol] += quantity
        
        if quantity > 0:  # Buy
            self.cash -= total_cost
        else:  # Sell
            self.cash += trade_value - commission_cost - slippage_cost
        
        # Record trade
        self.history.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": trade_value,
            "commission": commission_cost,
            "slippage": slippage_cost,
        })
        
        return True
    
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Current position size
        """
        return self.positions.get(symbol, 0)
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            prices: Dictionary of current prices for each symbol
            
        Returns:
            Total portfolio value
        """
        position_value = sum(
            self.positions.get(symbol, 0) * price
            for symbol, price in prices.items()
        )
        
        return self.cash + position_value
    
    def get_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        if not self.history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.history)
    
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve.
        
        Args:
            equity_curve: Series of portfolio values over time
            
        Returns:
            Series of returns
        """
        return equity_curve.pct_change().fillna(0)
    
    @property
    def total_value(self) -> float:
        """Get total portfolio value (cash only, no positions valued)."""
        return self.cash


class VectorizedPortfolio:
    """Vectorized portfolio for fast backtesting."""
    
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize vectorized portfolio.
        
        Args:
            initial_cash: Initial cash balance
            commission: Commission rate
            slippage: Slippage rate
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
    
    def backtest_signals(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_size: float = 1.0,
    ) -> pd.DataFrame:
        """Backtest trading signals using vectorized operations.
        
        Args:
            prices: Series of prices
            signals: Series of trading signals (1, 0, -1)
            position_size: Fraction of portfolio to use per trade
            
        Returns:
            DataFrame with backtest results
        """
        # Align signals and prices
        signals = signals.reindex(prices.index, fill_value=0)
        
        # Calculate position changes
        positions = signals * position_size
        trades = positions.diff().fillna(positions)
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Calculate strategy returns (with position)
        strategy_returns = positions.shift(1) * returns
        
        # Apply costs
        cost_rate = self.commission + self.slippage
        costs = abs(trades) * cost_rate
        
        # Net returns after costs
        net_returns = strategy_returns - costs
        
        # Calculate equity curve
        equity_curve = (1 + net_returns).cumprod() * self.initial_cash
        
        # Create results DataFrame
        results = pd.DataFrame({
            "price": prices,
            "signal": signals,
            "position": positions,
            "returns": returns,
            "strategy_returns": strategy_returns,
            "costs": costs,
            "net_returns": net_returns,
            "equity": equity_curve,
        })
        
        return results
    
    def backtest_multi_asset(
        self,
        prices_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        weights: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Backtest multiple assets simultaneously.
        
        Args:
            prices_df: DataFrame with prices for multiple assets
            signals_df: DataFrame with signals for multiple assets
            weights: Optional weights for each asset
            
        Returns:
            DataFrame with combined backtest results
        """
        if weights is None:
            # Equal weight
            n_assets = len(prices_df.columns)
            weights = pd.DataFrame(
                1.0 / n_assets,
                index=prices_df.index,
                columns=prices_df.columns
            )
        
        # Calculate returns for each asset
        returns = prices_df.pct_change().fillna(0)
        
        # Apply signals and weights
        positions = signals_df * weights
        strategy_returns = positions.shift(1) * returns
        
        # Calculate costs for each asset
        trades = positions.diff().fillna(positions)
        cost_rate = self.commission + self.slippage
        costs = abs(trades) * cost_rate
        
        # Aggregate across assets
        total_returns = strategy_returns.sum(axis=1)
        total_costs = costs.sum(axis=1)
        net_returns = total_returns - total_costs
        
        # Calculate equity curve
        equity_curve = (1 + net_returns).cumprod() * self.initial_cash
        
        return pd.DataFrame({
            "net_returns": net_returns,
            "equity": equity_curve,
            "total_costs": total_costs,
        })
