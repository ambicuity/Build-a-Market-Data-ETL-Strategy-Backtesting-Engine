"""Order book implementation for event-driven backtesting."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import heapq


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    filled_quantity: float = 0
    status: OrderStatus = OrderStatus.PENDING
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.filled_quantity >= self.quantity


@dataclass
class Trade:
    """Trade execution data structure."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: pd.Timestamp
    commission: float = 0
    slippage: float = 0


class OrderBook:
    """Limit order book for a single symbol."""
    
    def __init__(self, symbol: str):
        """Initialize order book.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        # Buy orders: max heap (negate price for max heap behavior)
        self.buy_orders: List[Tuple[float, Order]] = []
        # Sell orders: min heap
        self.sell_orders: List[Tuple[float, Order]] = []
        self.orders: Dict[str, Order] = {}
    
    def add_order(self, order: Order) -> None:
        """Add order to the book.
        
        Args:
            order: Order to add
        """
        if order.order_type != OrderType.LIMIT:
            raise ValueError("Only limit orders can be added to order book")
        
        self.orders[order.order_id] = order
        order.status = OrderStatus.OPEN
        
        if order.side == OrderSide.BUY:
            # Use negative price for max heap
            heapq.heappush(self.buy_orders, (-order.price, order))
        else:
            heapq.heappush(self.sell_orders, (order.price, order))
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order from the book.
        
        Args:
            order_id: Order ID to remove
            
        Returns:
            Removed order or None
        """
        if order_id not in self.orders:
            return None
        
        order = self.orders.pop(order_id)
        order.status = OrderStatus.CANCELLED
        return order
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price.
        
        Returns:
            Best bid price or None
        """
        # Clean up filled orders
        while self.buy_orders and self.buy_orders[0][1].is_filled:
            heapq.heappop(self.buy_orders)
        
        if self.buy_orders:
            return -self.buy_orders[0][0]  # Negate back to positive
        return None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price.
        
        Returns:
            Best ask price or None
        """
        # Clean up filled orders
        while self.sell_orders and self.sell_orders[0][1].is_filled:
            heapq.heappop(self.sell_orders)
        
        if self.sell_orders:
            return self.sell_orders[0][0]
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price between best bid and ask.
        
        Returns:
            Mid price or None
        """
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread.
        
        Returns:
            Spread or None
        """
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Get order book depth.
        
        Args:
            levels: Number of price levels to return
            
        Returns:
            Dictionary with 'bids' and 'asks' lists of (price, quantity) tuples
        """
        bids = []
        asks = []
        
        # Get top buy orders
        buy_copy = sorted(self.buy_orders, reverse=True)[:levels]
        for neg_price, order in buy_copy:
            if not order.is_filled:
                bids.append((-neg_price, order.remaining_quantity))
        
        # Get top sell orders
        sell_copy = sorted(self.sell_orders)[:levels]
        for price, order in sell_copy:
            if not order.is_filled:
                asks.append((price, order.remaining_quantity))
        
        return {"bids": bids, "asks": asks}


class EventDrivenSimulator:
    """Event-driven backtesting simulator with order book."""
    
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize simulator.
        
        Args:
            initial_cash: Initial cash balance
            commission: Commission rate
            slippage: Slippage rate
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        # State
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.pending_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.order_counter = 0
        self.trade_counter = 0
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.cash = self.initial_cash
        self.positions = {}
        self.order_books = {}
        self.pending_orders = []
        self.trades = []
        self.order_counter = 0
        self.trade_counter = 0
    
    def submit_order(self, order: Order) -> str:
        """Submit an order to the simulator.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
        """
        if not order.order_id:
            self.order_counter += 1
            order.order_id = f"ORD_{self.order_counter:06d}"
        
        # Validate order
        if order.order_type == OrderType.LIMIT and order.price is None:
            order.status = OrderStatus.REJECTED
            return order.order_id
        
        # Add to pending orders
        self.pending_orders.append(order)
        
        return order.order_id
    
    def process_market_order(
        self,
        order: Order,
        current_price: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Process a market order.
        
        Args:
            order: Market order to process
            current_price: Current market price
            timestamp: Current timestamp
        """
        # Calculate execution price with slippage
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + self.slippage)
        else:
            execution_price = current_price * (1 - self.slippage)
        
        # Check if we have sufficient cash/position
        if order.side == OrderSide.BUY:
            cost = order.quantity * execution_price * (1 + self.commission)
            if self.cash < cost:
                order.status = OrderStatus.REJECTED
                return
        else:
            position = self.positions.get(order.symbol, 0)
            if position < order.quantity:
                order.status = OrderStatus.REJECTED
                return
        
        # Execute trade
        self._execute_trade(order, order.quantity, execution_price, timestamp)
    
    def process_limit_order(
        self,
        order: Order,
        current_price: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Process a limit order.
        
        Args:
            order: Limit order to process
            current_price: Current market price
            timestamp: Current timestamp
        """
        # Check if limit order can be filled immediately
        can_fill = False
        
        if order.side == OrderSide.BUY and current_price <= order.price:
            can_fill = True
        elif order.side == OrderSide.SELL and current_price >= order.price:
            can_fill = True
        
        if can_fill:
            # Execute at limit price
            self._execute_trade(order, order.quantity, order.price, timestamp)
        else:
            # Add to order book
            if order.symbol not in self.order_books:
                self.order_books[order.symbol] = OrderBook(order.symbol)
            
            self.order_books[order.symbol].add_order(order)
    
    def _execute_trade(
        self,
        order: Order,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Execute a trade.
        
        Args:
            order: Order being executed
            quantity: Quantity to execute
            price: Execution price
            timestamp: Execution timestamp
        """
        commission_cost = quantity * price * self.commission
        
        # Update positions and cash
        if order.side == OrderSide.BUY:
            cost = quantity * price + commission_cost
            if self.cash >= cost:
                self.cash -= cost
                self.positions[order.symbol] = self.positions.get(order.symbol, 0) + quantity
                order.filled_quantity += quantity
            else:
                order.status = OrderStatus.REJECTED
                return
        else:
            proceeds = quantity * price - commission_cost
            if self.positions.get(order.symbol, 0) >= quantity:
                self.cash += proceeds
                self.positions[order.symbol] -= quantity
                order.filled_quantity += quantity
            else:
                order.status = OrderStatus.REJECTED
                return
        
        # Update order status
        if order.is_filled:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Record trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"TRD_{self.trade_counter:06d}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission_cost,
            slippage=abs(price - order.price) * quantity if order.price else 0
        )
        self.trades.append(trade)
    
    def process_tick(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Process a market tick (price update).
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Current timestamp
        """
        # Process pending orders
        orders_to_remove = []
        
        for order in self.pending_orders:
            if order.symbol != symbol:
                continue
            
            if order.order_type == OrderType.MARKET:
                self.process_market_order(order, price, timestamp)
                orders_to_remove.append(order)
            elif order.order_type == OrderType.LIMIT:
                self.process_limit_order(order, price, timestamp)
                if order.status != OrderStatus.PENDING:
                    orders_to_remove.append(order)
        
        # Remove processed orders
        for order in orders_to_remove:
            self.pending_orders.remove(order)
        
        # Check limit orders in order book
        if symbol in self.order_books:
            self._match_orders(symbol, price, timestamp)
    
    def _match_orders(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Match limit orders in order book.
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Current timestamp
        """
        book = self.order_books[symbol]
        
        # Match buy orders
        while book.buy_orders:
            best_bid = book.get_best_bid()
            if best_bid is None or best_bid < price:
                break
            
            _, order = heapq.heappop(book.buy_orders)
            if not order.is_filled and order.status == OrderStatus.OPEN:
                self._execute_trade(order, order.remaining_quantity, order.price, timestamp)
        
        # Match sell orders
        while book.sell_orders:
            best_ask = book.get_best_ask()
            if best_ask is None or best_ask > price:
                break
            
            _, order = heapq.heappop(book.sell_orders)
            if not order.is_filled and order.status == OrderStatus.OPEN:
                self._execute_trade(order, order.remaining_quantity, order.price, timestamp)
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            prices: Current prices for each symbol
            
        Returns:
            Total portfolio value
        """
        position_value = sum(
            self.positions.get(symbol, 0) * price
            for symbol, price in prices.items()
        )
        return self.cash + position_value
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get all trades as DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "trade_id": t.trade_id,
                "order_id": t.order_id,
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "timestamp": t.timestamp,
                "commission": t.commission,
                "slippage": t.slippage
            }
            for t in self.trades
        ])
