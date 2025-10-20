"""Machine learning strategy framework for algorithmic trading."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod
from backtesting.strategy import BaseStrategy


class MLStrategy(BaseStrategy):
    """Base class for machine learning based trading strategies."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        model: Any = None,
        features: Optional[List[str]] = None,
        lookback: int = 30,
        name: str = "ML_Strategy"
    ):
        """Initialize ML strategy.
        
        Args:
            data: OHLCV DataFrame
            model: Trained ML model (should have predict method)
            features: List of feature column names
            lookback: Lookback period for feature engineering
            name: Strategy name
        """
        self.model = model
        self.features = features or []
        self.lookback = lookback
        self.feature_data: Optional[pd.DataFrame] = None
        super().__init__(data, name)
    
    def _prepare_data(self) -> None:
        """Prepare data and engineer features."""
        super()._prepare_data()
        self.feature_data = self.engineer_features()
    
    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for the ML model.
        
        Returns:
            DataFrame with engineered features
        """
        raise NotImplementedError("Subclasses must implement engineer_features()")
    
    def generate_signals(self) -> pd.Series:
        """Generate trading signals using the ML model.
        
        Returns:
            Trading signals
        """
        if self.model is None:
            raise ValueError("Model not set. Train or load a model first.")
        
        if self.feature_data is None:
            self.feature_data = self.engineer_features()
        
        # Get feature columns
        X = self.feature_data[self.features].fillna(0)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Convert predictions to signals
        signals = pd.Series(predictions, index=X.index)
        
        return signals
    
    def train_model(
        self,
        train_data: pd.DataFrame,
        labels: pd.Series,
        model_class: Any,
        **model_kwargs
    ) -> Any:
        """Train the ML model.
        
        Args:
            train_data: Training data
            labels: Training labels
            model_class: ML model class (e.g., from sklearn)
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        # Engineer features on training data
        temp_data = self.data
        self.data = train_data
        feature_data = self.engineer_features()
        self.data = temp_data
        
        X = feature_data[self.features].fillna(0)
        y = labels.reindex(X.index).fillna(0)
        
        # Initialize and train model
        self.model = model_class(**model_kwargs)
        self.model.fit(X, y)
        
        return self.model


class TechnicalMLStrategy(MLStrategy):
    """ML strategy using technical indicators as features."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        model: Any = None,
        lookback: int = 30,
        name: str = "Technical_ML"
    ):
        """Initialize technical ML strategy.
        
        Args:
            data: OHLCV DataFrame
            model: Trained ML model
            lookback: Lookback period for technical indicators
            name: Strategy name
        """
        features = [
            "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
            "sma_fast", "sma_slow", "volume_sma_ratio",
            "returns_1d", "returns_5d", "volatility"
        ]
        super().__init__(data, model, features, lookback, name)
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer technical indicator features.
        
        Returns:
            DataFrame with technical features
        """
        df = self.data.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        # Bollinger Bands
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        df["bb_upper"] = sma + (2 * std)
        df["bb_lower"] = sma - (2 * std)
        
        # Moving Averages
        df["sma_fast"] = close.rolling(window=10).mean()
        df["sma_slow"] = close.rolling(window=50).mean()
        
        # Volume
        df["volume_sma_ratio"] = volume / volume.rolling(window=20).mean()
        
        # Returns
        df["returns_1d"] = close.pct_change()
        df["returns_5d"] = close.pct_change(periods=5)
        
        # Volatility
        df["volatility"] = close.pct_change().rolling(window=20).std()
        
        return df


class DeepLearningStrategy(MLStrategy):
    """Strategy using deep learning models (LSTM, CNN, etc.)."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        model: Any = None,
        sequence_length: int = 30,
        features: Optional[List[str]] = None,
        name: str = "DeepLearning"
    ):
        """Initialize deep learning strategy.
        
        Args:
            data: OHLCV DataFrame
            model: Trained deep learning model
            sequence_length: Length of input sequences
            features: List of feature names
            name: Strategy name
        """
        self.sequence_length = sequence_length
        if features is None:
            features = ["close", "volume", "returns", "volatility"]
        super().__init__(data, model, features, sequence_length, name)
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for deep learning.
        
        Returns:
            DataFrame with features
        """
        df = self.data.copy()
        
        # Basic features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()
        df["log_volume"] = np.log1p(df.get("volume", pd.Series(1, index=df.index)))
        
        # Price normalization
        df["close_normalized"] = df["close"] / df["close"].rolling(window=self.sequence_length).mean()
        
        return df
    
    def create_sequences(
        self,
        feature_data: pd.DataFrame,
        labels: Optional[pd.Series] = None
    ) -> tuple:
        """Create sequences for deep learning models.
        
        Args:
            feature_data: DataFrame with features
            labels: Optional labels for supervised learning
            
        Returns:
            Tuple of (X_sequences, y_sequences) as numpy arrays
        """
        X = feature_data[self.features].fillna(0).values
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if labels is not None:
                targets.append(labels.iloc[i + self.sequence_length])
        
        X_seq = np.array(sequences)
        y_seq = np.array(targets) if labels is not None else None
        
        return X_seq, y_seq


class ReinforcementLearningStrategy(BaseStrategy):
    """Strategy using reinforcement learning."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        agent: Any = None,
        state_size: int = 10,
        name: str = "RL_Strategy"
    ):
        """Initialize RL strategy.
        
        Args:
            data: OHLCV DataFrame
            agent: Trained RL agent
            state_size: Size of state representation
            name: Strategy name
        """
        self.agent = agent
        self.state_size = state_size
        self.current_position = 0
        super().__init__(data, name)
    
    def get_state(self, index: int) -> np.ndarray:
        """Get state representation at a given index.
        
        Args:
            index: Index in the data
            
        Returns:
            State vector
        """
        if index < self.state_size:
            return np.zeros(self.state_size)
        
        # Get recent price returns
        window = self.data["returns"].iloc[index - self.state_size:index]
        state = window.fillna(0).values
        
        return state
    
    def generate_signals(self) -> pd.Series:
        """Generate signals using the RL agent.
        
        Returns:
            Trading signals
        """
        if self.agent is None:
            raise ValueError("Agent not set. Train or load an agent first.")
        
        signals = pd.Series(0, index=self.data.index)
        
        for i in range(self.state_size, len(self.data)):
            state = self.get_state(i)
            action = self.agent.predict(state.reshape(1, -1))
            signals.iloc[i] = action
        
        return signals


class EnsembleStrategy(BaseStrategy):
    """Ensemble strategy combining multiple ML models."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategies: List[MLStrategy],
        weights: Optional[List[float]] = None,
        name: str = "Ensemble"
    ):
        """Initialize ensemble strategy.
        
        Args:
            data: OHLCV DataFrame
            strategies: List of ML strategies
            weights: Weights for each strategy (default: equal weights)
            name: Strategy name
        """
        self.strategies = strategies
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        self.weights = np.array(weights)
        super().__init__(data, name)
    
    def generate_signals(self) -> pd.Series:
        """Generate signals by combining multiple strategies.
        
        Returns:
            Ensemble trading signals
        """
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.get_signals()
            all_signals.append(signals)
        
        # Combine signals using weighted average
        signals_array = np.array([s.values for s in all_signals])
        ensemble_signals = np.average(signals_array, axis=0, weights=self.weights)
        
        # Convert to discrete signals
        result = pd.Series(ensemble_signals, index=self.data.index)
        result = result.apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))
        
        return result


class FeatureImportanceAnalyzer:
    """Analyze feature importance for ML strategies."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """Initialize feature importance analyzer.
        
        Args:
            model: Trained ML model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model.
        
        Returns:
            DataFrame with feature importance
        """
        # Try to get feature importance from the model
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_).flatten()
        else:
            raise ValueError("Model does not have feature importance or coefficients")
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        })
        
        return df.sort_values("importance", ascending=False)
    
    def plot_importance(self, top_n: int = 10) -> None:
        """Plot top N most important features.
        
        Args:
            top_n: Number of top features to plot
        """
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importance")
        plt.tight_layout()
        plt.show()


def create_train_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    shuffle: bool = False
) -> tuple:
    """Create train/test split for time series data.
    
    Args:
        data: Input DataFrame
        train_ratio: Ratio of data to use for training
        shuffle: Whether to shuffle (not recommended for time series)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if shuffle:
        data = data.sample(frac=1)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data


def walk_forward_validation(
    strategy: MLStrategy,
    data: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    test_size: int = 252
) -> List[Dict[str, Any]]:
    """Perform walk-forward validation.
    
    Args:
        strategy: ML strategy to validate
        data: Input data
        labels: Training labels
        n_splits: Number of validation splits
        test_size: Size of each test period
        
    Returns:
        List of validation results
    """
    results = []
    total_size = len(data)
    
    for i in range(n_splits):
        # Calculate split indices
        test_start = total_size - (n_splits - i) * test_size
        test_end = test_start + test_size
        
        if test_start < test_size:
            continue
        
        # Split data
        train_data = data.iloc[:test_start]
        test_data = data.iloc[test_start:test_end]
        train_labels = labels.iloc[:test_start]
        
        # Train strategy
        strategy.data = train_data
        strategy.feature_data = None  # Reset features
        
        # Generate signals on test data
        strategy.data = test_data
        strategy.feature_data = None
        signals = strategy.generate_signals()
        
        results.append({
            "split": i,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "signals": signals
        })
    
    return results
