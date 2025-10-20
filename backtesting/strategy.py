"""Base strategy class for backtesting."""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, data: pd.DataFrame, name: str = "Strategy"):
        """Initialize strategy.
        
        Args:
            data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            name: Strategy name
        """
        self.data = data.copy()
        self.name = name
        self._signals: Optional[pd.Series] = None
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Prepare data for strategy (calculate returns, etc.)."""
        # Calculate returns
        if "close" in self.data.columns:
            self.data["returns"] = self.data["close"].pct_change()
        
        # Set timestamp as index if it exists
        if "timestamp" in self.data.columns:
            self.data.set_index("timestamp", inplace=True)
    
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Generate trading signals.
        
        Returns:
            Series with values: 1 (long), -1 (short), 0 (neutral)
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def get_signals(self) -> pd.Series:
        """Get cached signals or generate new ones.
        
        Returns:
            Trading signals
        """
        if self._signals is None:
            self._signals = self.generate_signals()
        return self._signals
    
    def calculate_positions(self, signals: Optional[pd.Series] = None) -> pd.Series:
        """Calculate position sizes from signals.
        
        Args:
            signals: Trading signals (if None, uses cached signals)
            
        Returns:
            Position sizes
        """
        if signals is None:
            signals = self.get_signals()
        
        # Simple: position = signal (can be extended for position sizing)
        return signals


class MeanReversionStrategy(BaseStrategy):
    """Simple mean reversion strategy."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
        name: str = "MeanReversion",
    ):
        """Initialize mean reversion strategy.
        
        Args:
            data: OHLCV DataFrame
            window: Rolling window for mean/std calculation
            num_std: Number of standard deviations for entry signal
            name: Strategy name
        """
        self.window = window
        self.num_std = num_std
        super().__init__(data, name)
    
    def generate_signals(self) -> pd.Series:
        """Generate mean reversion signals.
        
        Returns:
            Trading signals: 1 (long), -1 (short), 0 (neutral)
        """
        close = self.data["close"]
        
        # Calculate rolling mean and std
        rolling_mean = close.rolling(window=self.window).mean()
        rolling_std = close.rolling(window=self.window).std()
        
        # Calculate z-score
        z_score = (close - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[z_score < -self.num_std] = 1   # Buy when price is low
        signals[z_score > self.num_std] = -1   # Sell when price is high
        
        return signals


class MovingAverageCrossStrategy(BaseStrategy):
    """Moving average crossover strategy."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        fast_window: int = 10,
        slow_window: int = 50,
        name: str = "MA_Cross",
    ):
        """Initialize moving average crossover strategy.
        
        Args:
            data: OHLCV DataFrame
            fast_window: Fast MA window
            slow_window: Slow MA window
            name: Strategy name
        """
        self.fast_window = fast_window
        self.slow_window = slow_window
        super().__init__(data, name)
    
    def generate_signals(self) -> pd.Series:
        """Generate MA crossover signals.
        
        Returns:
            Trading signals: 1 (long), 0 (neutral)
        """
        close = self.data["close"]
        
        # Calculate moving averages
        fast_ma = close.rolling(window=self.fast_window).mean()
        slow_ma = close.rolling(window=self.slow_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[fast_ma > slow_ma] = 1   # Long when fast > slow
        signals[fast_ma <= slow_ma] = 0  # Neutral otherwise
        
        return signals


class MomentumStrategy(BaseStrategy):
    """Momentum strategy based on recent returns."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
        threshold: float = 0.02,
        name: str = "Momentum",
    ):
        """Initialize momentum strategy.
        
        Args:
            data: OHLCV DataFrame
            lookback: Lookback period for momentum calculation
            threshold: Threshold for entry signal
            name: Strategy name
        """
        self.lookback = lookback
        self.threshold = threshold
        super().__init__(data, name)
    
    def generate_signals(self) -> pd.Series:
        """Generate momentum signals.
        
        Returns:
            Trading signals: 1 (long), -1 (short), 0 (neutral)
        """
        close = self.data["close"]
        
        # Calculate momentum (percent change over lookback period)
        momentum = close.pct_change(periods=self.lookback)
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[momentum > self.threshold] = 1    # Long on positive momentum
        signals[momentum < -self.threshold] = -1  # Short on negative momentum
        
        return signals


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy."""
    
    def __init__(self, data: pd.DataFrame, name: str = "BuyAndHold"):
        """Initialize buy and hold strategy.
        
        Args:
            data: OHLCV DataFrame
            name: Strategy name
        """
        super().__init__(data, name)
    
    def generate_signals(self) -> pd.Series:
        """Generate buy and hold signals (always long).
        
        Returns:
            Trading signals: 1 (always long)
        """
        return pd.Series(1, index=self.data.index)
