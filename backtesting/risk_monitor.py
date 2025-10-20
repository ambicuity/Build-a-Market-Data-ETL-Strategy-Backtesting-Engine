"""Real-time risk monitoring and management system."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import warnings


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: pd.Timestamp
    metric: str
    value: float
    threshold: float
    level: RiskLevel
    message: str


class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(
        self,
        max_drawdown_threshold: float = 0.20,
        var_confidence: float = 0.95,
        position_limit: float = 0.25,
        daily_loss_limit: float = 0.05
    ):
        """Initialize risk monitor.
        
        Args:
            max_drawdown_threshold: Maximum acceptable drawdown
            var_confidence: Confidence level for VaR calculation
            position_limit: Maximum position size as fraction of portfolio
            daily_loss_limit: Maximum daily loss as fraction of portfolio
        """
        self.max_drawdown_threshold = max_drawdown_threshold
        self.var_confidence = var_confidence
        self.position_limit = position_limit
        self.daily_loss_limit = daily_loss_limit
        
        self.alerts: List[RiskAlert] = []
        self.metrics_history: List[Dict] = []
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence: Confidence level (default: use instance setting)
            
        Returns:
            VaR value
        """
        if confidence is None:
            confidence = self.var_confidence
        
        return returns.quantile(1 - confidence)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            
        Returns:
            CVaR value
        """
        if confidence is None:
            confidence = self.var_confidence
        
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Drawdown series
        """
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Maximum drawdown value
        """
        drawdown = self.calculate_drawdown(equity_curve)
        return drawdown.min()
    
    def check_position_limits(
        self,
        positions: Dict[str, float],
        portfolio_value: float
    ) -> List[RiskAlert]:
        """Check if position sizes exceed limits.
        
        Args:
            positions: Dictionary of positions (symbol -> value)
            portfolio_value: Total portfolio value
            
        Returns:
            List of risk alerts
        """
        alerts = []
        timestamp = pd.Timestamp.now()
        
        for symbol, position_value in positions.items():
            position_fraction = abs(position_value) / portfolio_value
            
            if position_fraction > self.position_limit:
                level = RiskLevel.HIGH if position_fraction > self.position_limit * 1.5 else RiskLevel.MEDIUM
                
                alert = RiskAlert(
                    timestamp=timestamp,
                    metric="position_size",
                    value=position_fraction,
                    threshold=self.position_limit,
                    level=level,
                    message=f"Position size for {symbol} exceeds limit: {position_fraction:.2%} > {self.position_limit:.2%}"
                )
                alerts.append(alert)
        
        return alerts
    
    def check_drawdown(
        self,
        equity_curve: pd.Series,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Optional[RiskAlert]:
        """Check if drawdown exceeds threshold.
        
        Args:
            equity_curve: Series of portfolio values
            timestamp: Current timestamp
            
        Returns:
            Risk alert if threshold exceeded
        """
        max_dd = abs(self.calculate_max_drawdown(equity_curve))
        
        if max_dd > self.max_drawdown_threshold:
            if timestamp is None:
                timestamp = equity_curve.index[-1]
            
            level = RiskLevel.CRITICAL if max_dd > self.max_drawdown_threshold * 1.5 else RiskLevel.HIGH
            
            return RiskAlert(
                timestamp=timestamp,
                metric="max_drawdown",
                value=max_dd,
                threshold=self.max_drawdown_threshold,
                level=level,
                message=f"Maximum drawdown exceeds threshold: {max_dd:.2%} > {self.max_drawdown_threshold:.2%}"
            )
        
        return None
    
    def check_daily_loss(
        self,
        returns: pd.Series,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Optional[RiskAlert]:
        """Check if daily loss exceeds limit.
        
        Args:
            returns: Series of returns
            timestamp: Current timestamp
            
        Returns:
            Risk alert if limit exceeded
        """
        if len(returns) == 0:
            return None
        
        latest_return = returns.iloc[-1]
        
        if latest_return < -self.daily_loss_limit:
            if timestamp is None:
                timestamp = returns.index[-1]
            
            level = RiskLevel.CRITICAL if latest_return < -self.daily_loss_limit * 2 else RiskLevel.HIGH
            
            return RiskAlert(
                timestamp=timestamp,
                metric="daily_loss",
                value=abs(latest_return),
                threshold=self.daily_loss_limit,
                level=level,
                message=f"Daily loss exceeds limit: {abs(latest_return):.2%} > {self.daily_loss_limit:.2%}"
            )
        
        return None
    
    def monitor(
        self,
        equity_curve: pd.Series,
        positions: Dict[str, float],
        returns: pd.Series,
        timestamp: Optional[pd.Timestamp] = None
    ) -> List[RiskAlert]:
        """Run comprehensive risk monitoring.
        
        Args:
            equity_curve: Series of portfolio values
            positions: Current positions
            returns: Series of returns
            timestamp: Current timestamp
            
        Returns:
            List of risk alerts
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        alerts = []
        
        # Check position limits
        portfolio_value = equity_curve.iloc[-1]
        position_alerts = self.check_position_limits(positions, portfolio_value)
        alerts.extend(position_alerts)
        
        # Check drawdown
        dd_alert = self.check_drawdown(equity_curve, timestamp)
        if dd_alert:
            alerts.append(dd_alert)
        
        # Check daily loss
        loss_alert = self.check_daily_loss(returns, timestamp)
        if loss_alert:
            alerts.append(loss_alert)
        
        # Calculate and store risk metrics
        metrics = {
            "timestamp": timestamp,
            "var": self.calculate_var(returns),
            "cvar": self.calculate_cvar(returns),
            "max_drawdown": abs(self.calculate_max_drawdown(equity_curve)),
            "portfolio_value": portfolio_value,
            "num_alerts": len(alerts)
        }
        self.metrics_history.append(metrics)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    def get_alerts_df(self) -> pd.DataFrame:
        """Get all alerts as DataFrame.
        
        Returns:
            DataFrame with alert history
        """
        if not self.alerts:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "metric": a.metric,
                "value": a.value,
                "threshold": a.threshold,
                "level": a.level.value,
                "message": a.message
            }
            for a in self.alerts
        ])
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics history as DataFrame.
        
        Returns:
            DataFrame with metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self.alerts = []


class PositionSizer:
    """Position sizing based on risk management principles."""
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.25
    ):
        """Initialize position sizer.
        
        Args:
            risk_per_trade: Maximum risk per trade as fraction of portfolio
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """Calculate optimal position size using Kelly criterion.
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (absolute value)
            
        Returns:
            Optimal position size as fraction
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fractional Kelly (typically 25-50% of full Kelly)
        kelly_fraction *= 0.25
        
        # Cap at maximum position size
        return min(max(kelly_fraction, 0), self.max_position_size)
    
    def volatility_based_sizing(
        self,
        portfolio_value: float,
        asset_volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """Calculate position size based on volatility targeting.
        
        Args:
            portfolio_value: Current portfolio value
            asset_volatility: Asset volatility (annualized std dev)
            target_volatility: Target portfolio volatility
            
        Returns:
            Position size in dollars
        """
        if asset_volatility == 0:
            return 0
        
        # Calculate position size to achieve target volatility
        position_fraction = target_volatility / asset_volatility
        position_fraction = min(position_fraction, self.max_position_size)
        
        return portfolio_value * position_fraction
    
    def fixed_risk_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """Calculate position size based on fixed risk per trade.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Number of shares/units
        """
        risk_amount = portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        # Check maximum position size
        max_shares = (portfolio_value * self.max_position_size) / entry_price
        
        return min(position_size, max_shares)


class StopLossManager:
    """Manage stop loss orders and trailing stops."""
    
    def __init__(
        self,
        initial_stop_pct: float = 0.05,
        trailing_stop_pct: float = 0.03
    ):
        """Initialize stop loss manager.
        
        Args:
            initial_stop_pct: Initial stop loss percentage
            trailing_stop_pct: Trailing stop percentage
        """
        self.initial_stop_pct = initial_stop_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.stops: Dict[str, float] = {}
        self.highest_prices: Dict[str, float] = {}
    
    def set_initial_stop(
        self,
        symbol: str,
        entry_price: float,
        side: str = "long"
    ) -> float:
        """Set initial stop loss.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Trade side ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if side == "long":
            stop_price = entry_price * (1 - self.initial_stop_pct)
        else:
            stop_price = entry_price * (1 + self.initial_stop_pct)
        
        self.stops[symbol] = stop_price
        self.highest_prices[symbol] = entry_price
        
        return stop_price
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        side: str = "long"
    ) -> Optional[float]:
        """Update trailing stop loss.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            side: Trade side ('long' or 'short')
            
        Returns:
            New stop loss price or None if not updated
        """
        if symbol not in self.stops:
            return None
        
        if side == "long":
            # Update highest price
            if current_price > self.highest_prices[symbol]:
                self.highest_prices[symbol] = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 - self.trailing_stop_pct)
                
                # Only move stop up, never down
                if new_stop > self.stops[symbol]:
                    self.stops[symbol] = new_stop
                    return new_stop
        else:
            # For short positions
            if current_price < self.highest_prices[symbol]:
                self.highest_prices[symbol] = current_price
                
                new_stop = current_price * (1 + self.trailing_stop_pct)
                
                # Only move stop down, never up
                if new_stop < self.stops[symbol]:
                    self.stops[symbol] = new_stop
                    return new_stop
        
        return None
    
    def check_stop_hit(
        self,
        symbol: str,
        current_price: float,
        side: str = "long"
    ) -> bool:
        """Check if stop loss has been hit.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            side: Trade side
            
        Returns:
            True if stop hit, False otherwise
        """
        if symbol not in self.stops:
            return False
        
        stop_price = self.stops[symbol]
        
        if side == "long":
            return current_price <= stop_price
        else:
            return current_price >= stop_price
    
    def remove_stop(self, symbol: str) -> None:
        """Remove stop loss for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        self.stops.pop(symbol, None)
        self.highest_prices.pop(symbol, None)
