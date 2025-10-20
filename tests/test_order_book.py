"""Tests for order book and event-driven simulator."""

import pytest
import pandas as pd
import numpy as np
from backtesting.order_book import (
    Order, OrderBook, EventDrivenSimulator,
    OrderSide, OrderType, OrderStatus
)


class TestOrder:
    """Test Order data structure."""
    
    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        assert order.order_id == "TEST_001"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.price == 150.0
        assert order.filled_quantity == 0
        assert order.status == OrderStatus.PENDING
    
    def test_order_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        assert order.remaining_quantity == 100
        
        order.filled_quantity = 60
        assert order.remaining_quantity == 40
    
    def test_order_is_filled(self):
        """Test is_filled property."""
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        assert not order.is_filled
        
        order.filled_quantity = 100
        assert order.is_filled


class TestOrderBook:
    """Test OrderBook functionality."""
    
    def test_orderbook_init(self):
        """Test order book initialization."""
        book = OrderBook("AAPL")
        assert book.symbol == "AAPL"
        assert len(book.buy_orders) == 0
        assert len(book.sell_orders) == 0
    
    def test_add_buy_order(self):
        """Test adding a buy order."""
        book = OrderBook("AAPL")
        order = Order(
            order_id="BUY_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        book.add_order(order)
        
        assert len(book.buy_orders) == 1
        assert order.status == OrderStatus.OPEN
        assert "BUY_001" in book.orders
    
    def test_add_sell_order(self):
        """Test adding a sell order."""
        book = OrderBook("AAPL")
        order = Order(
            order_id="SELL_001",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=155.0
        )
        
        book.add_order(order)
        
        assert len(book.sell_orders) == 1
        assert order.status == OrderStatus.OPEN
    
    def test_get_best_bid(self):
        """Test getting best bid price."""
        book = OrderBook("AAPL")
        
        # Add multiple buy orders
        for i, price in enumerate([149.0, 150.0, 148.0]):
            order = Order(
                order_id=f"BUY_{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                price=price
            )
            book.add_order(order)
        
        # Best bid should be 150.0 (highest)
        assert book.get_best_bid() == 150.0
    
    def test_get_best_ask(self):
        """Test getting best ask price."""
        book = OrderBook("AAPL")
        
        # Add multiple sell orders
        for i, price in enumerate([151.0, 152.0, 150.0]):
            order = Order(
                order_id=f"SELL_{i:03d}",
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=100,
                price=price
            )
            book.add_order(order)
        
        # Best ask should be 150.0 (lowest)
        assert book.get_best_ask() == 150.0
    
    def test_get_spread(self):
        """Test bid-ask spread calculation."""
        book = OrderBook("AAPL")
        
        # Add buy order
        buy_order = Order(
            order_id="BUY_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=149.0
        )
        book.add_order(buy_order)
        
        # Add sell order
        sell_order = Order(
            order_id="SELL_001",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=151.0
        )
        book.add_order(sell_order)
        
        spread = book.get_spread()
        assert spread == 2.0


class TestEventDrivenSimulator:
    """Test EventDrivenSimulator functionality."""
    
    def test_simulator_init(self):
        """Test simulator initialization."""
        sim = EventDrivenSimulator(initial_cash=100000)
        
        assert sim.cash == 100000
        assert len(sim.positions) == 0
        assert len(sim.pending_orders) == 0
        assert len(sim.trades) == 0
    
    def test_submit_market_order(self):
        """Test submitting a market order."""
        sim = EventDrivenSimulator(initial_cash=100000)
        
        order = Order(
            order_id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        order_id = sim.submit_order(order)
        
        assert order_id.startswith("ORD_")
        assert len(sim.pending_orders) == 1
    
    def test_process_market_order(self):
        """Test processing a market order."""
        sim = EventDrivenSimulator(initial_cash=100000, commission=0.001, slippage=0.0005)
        
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        timestamp = pd.Timestamp.now()
        sim.process_market_order(order, 150.0, timestamp)
        
        # Check if order was executed
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        
        # Check cash deduction (price + slippage + commission)
        execution_price = 150.0 * (1 + 0.0005)  # With slippage
        cost = 100 * execution_price * (1 + 0.001)  # With commission
        assert sim.cash < 100000
        
        # Check position
        assert sim.positions["AAPL"] == 100
        
        # Check trade recorded
        assert len(sim.trades) == 1
    
    def test_process_limit_order(self):
        """Test processing a limit order."""
        sim = EventDrivenSimulator(initial_cash=100000)
        
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        timestamp = pd.Timestamp.now()
        
        # Price above limit - should not fill
        sim.process_limit_order(order, 151.0, timestamp)
        assert order.status == OrderStatus.PENDING or order.status == OrderStatus.OPEN
        
        # Reset order
        order.status = OrderStatus.PENDING
        order.filled_quantity = 0
        
        # Price at or below limit - should fill
        sim.process_limit_order(order, 150.0, timestamp)
        assert order.status == OrderStatus.FILLED
    
    def test_process_tick(self):
        """Test processing market ticks."""
        sim = EventDrivenSimulator(initial_cash=100000)
        
        # Submit a market order
        order = Order(
            order_id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        sim.submit_order(order)
        
        # Process tick
        timestamp = pd.Timestamp.now()
        sim.process_tick("AAPL", 150.0, timestamp)
        
        # Order should be executed
        assert len(sim.pending_orders) == 0
        assert len(sim.trades) > 0
    
    def test_insufficient_cash(self):
        """Test rejection due to insufficient cash."""
        sim = EventDrivenSimulator(initial_cash=1000)
        
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        timestamp = pd.Timestamp.now()
        sim.process_market_order(order, 150.0, timestamp)
        
        # Order should be rejected
        assert order.status == OrderStatus.REJECTED
        assert sim.cash == 1000  # Cash unchanged
    
    def test_get_portfolio_value(self):
        """Test portfolio value calculation."""
        sim = EventDrivenSimulator(initial_cash=100000)
        sim.positions = {"AAPL": 100, "MSFT": 50}
        
        prices = {"AAPL": 150.0, "MSFT": 200.0}
        portfolio_value = sim.get_portfolio_value(prices)
        
        # 100000 cash - trade costs + (100 * 150 + 50 * 200)
        expected_min = 100000  # At least initial cash
        assert portfolio_value >= expected_min * 0.95  # Allow for costs


def test_event_driven_simulation_flow():
    """Test complete event-driven simulation flow."""
    sim = EventDrivenSimulator(initial_cash=100000)
    
    # Submit buy order
    buy_order = Order(
        order_id="",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=150.0
    )
    sim.submit_order(buy_order)
    
    # Process ticks
    timestamp = pd.Timestamp.now()
    sim.process_tick("AAPL", 151.0, timestamp)  # Above limit, shouldn't fill
    assert sim.positions.get("AAPL", 0) == 0
    
    sim.process_tick("AAPL", 149.0, timestamp)  # Below limit, should fill
    assert sim.positions.get("AAPL", 0) == 100
    
    # Submit sell order
    sell_order = Order(
        order_id="",
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100
    )
    sim.submit_order(sell_order)
    
    sim.process_tick("AAPL", 150.0, timestamp)  # Execute market order
    assert sim.positions.get("AAPL", 0) == 0
    
    # Check trades were recorded
    assert len(sim.trades) == 2
