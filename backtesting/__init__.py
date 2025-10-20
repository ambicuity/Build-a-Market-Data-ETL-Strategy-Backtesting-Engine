"""Backtesting Engine Module for Strategy Testing."""

from .strategy import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MovingAverageCrossStrategy,
    MomentumStrategy,
)
from .portfolio import Portfolio, VectorizedPortfolio
from .engine import BacktestEngine, EventDrivenEngine
from .metrics import PerformanceMetrics
from .visualization import Visualizer
from .portfolio_optimization import PortfolioOptimizer, RiskParityOptimizer
from .order_book import OrderBook, EventDrivenSimulator, Order, OrderSide, OrderType
from .ml_strategy import (
    MLStrategy,
    TechnicalMLStrategy,
    DeepLearningStrategy,
    ReinforcementLearningStrategy,
    EnsembleStrategy,
)
from .risk_monitor import RiskMonitor, PositionSizer, StopLossManager
from .derivatives import (
    BlackScholesModel,
    OptionStrategy,
    FuturesCalculator,
    DerivativesPortfolio,
    Option,
    FuturesContract,
)
from .paper_trading import PaperBroker, PaperTradingEngine, LiveDataFeed
from .dashboard import DashboardServer

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "MeanReversionStrategy",
    "MovingAverageCrossStrategy",
    "MomentumStrategy",
    "Portfolio",
    "VectorizedPortfolio",
    "BacktestEngine",
    "EventDrivenEngine",
    "PerformanceMetrics",
    "Visualizer",
    "PortfolioOptimizer",
    "RiskParityOptimizer",
    "OrderBook",
    "EventDrivenSimulator",
    "Order",
    "OrderSide",
    "OrderType",
    "MLStrategy",
    "TechnicalMLStrategy",
    "DeepLearningStrategy",
    "ReinforcementLearningStrategy",
    "EnsembleStrategy",
    "RiskMonitor",
    "PositionSizer",
    "StopLossManager",
    "BlackScholesModel",
    "OptionStrategy",
    "FuturesCalculator",
    "DerivativesPortfolio",
    "Option",
    "FuturesContract",
    "PaperBroker",
    "PaperTradingEngine",
    "LiveDataFeed",
    "DashboardServer",
]
