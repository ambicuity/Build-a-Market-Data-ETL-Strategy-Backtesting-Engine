"""Live paper trading integration for strategy testing."""

import pandas as pd
import numpy as np
import asyncio
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import aiohttp
from backtesting.order_book import Order, OrderSide, OrderType, OrderStatus, Trade


@dataclass
class PaperAccount:
    """Paper trading account."""
    account_id: str
    initial_cash: float
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize account cash."""
        self.cash = self.initial_cash
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            prices: Current market prices
            
        Returns:
            Total portfolio value
        """
        position_value = sum(
            qty * prices.get(symbol, 0)
            for symbol, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def to_dict(self) -> Dict:
        """Convert account to dictionary.
        
        Returns:
            Account data as dictionary
        """
        return {
            "account_id": self.account_id,
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": self.positions,
            "num_orders": len(self.orders),
            "num_trades": len(self.trades)
        }


class PaperBroker:
    """Paper trading broker with simulated order execution."""
    
    def __init__(
        self,
        initial_cash: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        latency_ms: int = 50
    ):
        """Initialize paper broker.
        
        Args:
            initial_cash: Initial cash balance
            commission: Commission rate
            slippage: Slippage rate
            latency_ms: Simulated order latency in milliseconds
        """
        self.account = PaperAccount("PAPER_001", initial_cash)
        self.commission = commission
        self.slippage = slippage
        self.latency_ms = latency_ms
        self.order_counter = 0
        self.trade_counter = 0
        self.market_data: Dict[str, float] = {}
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None
    ) -> Order:
        """Submit a paper order.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            order_type: Market or limit
            quantity: Order quantity
            price: Limit price (for limit orders)
            
        Returns:
            Order object
        """
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        self.order_counter += 1
        order = Order(
            order_id=f"PAPER_ORD_{self.order_counter:06d}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=pd.Timestamp.now()
        )
        
        self.account.orders.append(order)
        
        # Try to execute immediately for market orders
        if order_type == OrderType.MARKET:
            await self._execute_order(order)
        
        return order
    
    async def _execute_order(self, order: Order) -> None:
        """Execute a paper order.
        
        Args:
            order: Order to execute
        """
        if order.symbol not in self.market_data:
            order.status = OrderStatus.REJECTED
            return
        
        market_price = self.market_data[order.symbol]
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + self.slippage)
        else:
            execution_price = market_price * (1 - self.slippage)
        
        # Check for limit price
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and execution_price > order.price:
                return  # Can't execute
            if order.side == OrderSide.SELL and execution_price < order.price:
                return  # Can't execute
            execution_price = order.price
        
        # Calculate costs
        trade_value = order.quantity * execution_price
        commission_cost = trade_value * self.commission
        
        # Check cash/position availability
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission_cost
            if self.account.cash < total_cost:
                order.status = OrderStatus.REJECTED
                return
            
            self.account.cash -= total_cost
            self.account.positions[order.symbol] = \
                self.account.positions.get(order.symbol, 0) + order.quantity
        else:
            if self.account.positions.get(order.symbol, 0) < order.quantity:
                order.status = OrderStatus.REJECTED
                return
            
            proceeds = trade_value - commission_cost
            self.account.cash += proceeds
            self.account.positions[order.symbol] -= order.quantity
        
        # Update order status
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        
        # Record trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"PAPER_TRD_{self.trade_counter:06d}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=pd.Timestamp.now(),
            commission=commission_cost
        )
        self.account.trades.append(trade)
    
    def update_market_data(self, symbol: str, price: float) -> None:
        """Update market data for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
        """
        self.market_data[symbol] = price
    
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position quantity
        """
        return self.account.positions.get(symbol, 0)
    
    def get_account_summary(self) -> Dict:
        """Get account summary.
        
        Returns:
            Account summary dictionary
        """
        portfolio_value = self.account.get_portfolio_value(self.market_data)
        
        return {
            **self.account.to_dict(),
            "portfolio_value": portfolio_value,
            "pnl": portfolio_value - self.account.initial_cash,
            "return_pct": (portfolio_value / self.account.initial_cash - 1) * 100
        }


class LiveDataFeed:
    """Live market data feed (simulated or real)."""
    
    def __init__(self, symbols: List[str], update_interval: int = 1):
        """Initialize live data feed.
        
        Args:
            symbols: List of symbols to track
            update_interval: Update interval in seconds
        """
        self.symbols = symbols
        self.update_interval = update_interval
        self.latest_prices: Dict[str, float] = {}
        self.subscribers: List[Callable] = []
        self.running = False
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to market data updates.
        
        Args:
            callback: Callback function to receive updates
        """
        self.subscribers.append(callback)
    
    async def start(self, use_simulated: bool = True) -> None:
        """Start the data feed.
        
        Args:
            use_simulated: Use simulated data (True) or real API (False)
        """
        self.running = True
        
        if use_simulated:
            await self._simulated_feed()
        else:
            await self._real_feed()
    
    async def _simulated_feed(self) -> None:
        """Simulated market data feed."""
        # Initialize with random prices
        for symbol in self.symbols:
            self.latest_prices[symbol] = np.random.uniform(50, 200)
        
        while self.running:
            # Update prices with random walk
            for symbol in self.symbols:
                current_price = self.latest_prices[symbol]
                change_pct = np.random.normal(0, 0.001)  # 0.1% std dev
                new_price = current_price * (1 + change_pct)
                self.latest_prices[symbol] = new_price
            
            # Notify subscribers
            for callback in self.subscribers:
                await callback(self.latest_prices.copy())
            
            await asyncio.sleep(self.update_interval)
    
    async def _real_feed(self) -> None:
        """Real market data feed (placeholder for actual API integration)."""
        # This would integrate with a real data provider like Alpha Vantage,
        # IEX Cloud, Polygon.io, etc.
        raise NotImplementedError("Real data feed not implemented yet")
    
    def stop(self) -> None:
        """Stop the data feed."""
        self.running = False


class PaperTradingEngine:
    """Complete paper trading engine."""
    
    def __init__(
        self,
        symbols: List[str],
        initial_cash: float = 100_000,
        strategy: Optional[any] = None
    ):
        """Initialize paper trading engine.
        
        Args:
            symbols: List of symbols to trade
            initial_cash: Initial cash balance
            strategy: Trading strategy to execute
        """
        self.symbols = symbols
        self.broker = PaperBroker(initial_cash)
        self.data_feed = LiveDataFeed(symbols)
        self.strategy = strategy
        self.performance_history: List[Dict] = []
        
        # Subscribe to market data
        self.data_feed.subscribe(self._on_market_data)
    
    async def _on_market_data(self, prices: Dict[str, float]) -> None:
        """Handle market data updates.
        
        Args:
            prices: Current market prices
        """
        # Update broker market data
        for symbol, price in prices.items():
            self.broker.update_market_data(symbol, price)
        
        # Execute strategy if provided
        if self.strategy:
            await self._execute_strategy(prices)
        
        # Record performance
        self._record_performance()
    
    async def _execute_strategy(self, prices: Dict[str, float]) -> None:
        """Execute trading strategy.
        
        Args:
            prices: Current market prices
        """
        # This is a placeholder - actual implementation would depend on strategy interface
        # For example, strategy could have a generate_orders method
        if hasattr(self.strategy, 'generate_orders'):
            orders = self.strategy.generate_orders(prices, self.broker.account)
            
            for order_spec in orders:
                await self.broker.submit_order(**order_spec)
    
    def _record_performance(self) -> None:
        """Record performance snapshot."""
        summary = self.broker.get_account_summary()
        summary['timestamp'] = pd.Timestamp.now()
        self.performance_history.append(summary)
    
    async def start(self, duration_seconds: Optional[int] = None) -> None:
        """Start paper trading.
        
        Args:
            duration_seconds: Run for specified duration (None = indefinite)
        """
        print(f"Starting paper trading for symbols: {self.symbols}")
        print(f"Initial cash: ${self.broker.account.initial_cash:,.2f}")
        
        # Start data feed
        feed_task = asyncio.create_task(self.data_feed.start(use_simulated=True))
        
        # Wait for specified duration or until stopped
        if duration_seconds:
            await asyncio.sleep(duration_seconds)
            self.stop()
        
        await feed_task
    
    def stop(self) -> None:
        """Stop paper trading."""
        self.data_feed.stop()
        print("\nPaper trading stopped")
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print trading summary."""
        summary = self.broker.get_account_summary()
        
        print("\n=== Paper Trading Summary ===")
        print(f"Initial Cash: ${summary['initial_cash']:,.2f}")
        print(f"Current Cash: ${summary['cash']:,.2f}")
        print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"P&L: ${summary['pnl']:,.2f}")
        print(f"Return: {summary['return_pct']:.2f}%")
        print(f"Total Orders: {summary['num_orders']}")
        print(f"Total Trades: {summary['num_trades']}")
        
        print("\nPositions:")
        for symbol, qty in self.broker.account.positions.items():
            if qty != 0:
                price = self.broker.market_data.get(symbol, 0)
                value = qty * price
                print(f"  {symbol}: {qty:.2f} shares @ ${price:.2f} = ${value:,.2f}")
    
    def get_performance_df(self) -> pd.DataFrame:
        """Get performance history as DataFrame.
        
        Returns:
            DataFrame with performance metrics
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def export_results(self, filepath: str) -> None:
        """Export trading results to file.
        
        Args:
            filepath: Output file path
        """
        results = {
            "summary": self.broker.get_account_summary(),
            "performance_history": self.performance_history,
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "quantity": t.quantity,
                    "price": t.price,
                    "timestamp": t.timestamp.isoformat(),
                    "commission": t.commission
                }
                for t in self.broker.account.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results exported to {filepath}")


async def run_paper_trading_demo():
    """Demo function for paper trading."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    engine = PaperTradingEngine(symbols, initial_cash=100_000)
    
    # Run for 60 seconds
    await engine.start(duration_seconds=60)
    
    # Export results
    engine.export_results("paper_trading_results.json")


if __name__ == "__main__":
    asyncio.run(run_paper_trading_demo())
