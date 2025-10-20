# Advanced Features Guide

This guide covers the advanced features of the Market Data ETL & Backtesting Engine, including portfolio optimization, ML strategies, risk monitoring, derivatives, and more.

## Table of Contents

1. [Multi-Asset Portfolio Optimization](#multi-asset-portfolio-optimization)
2. [Event-Driven Simulator with Order Book](#event-driven-simulator-with-order-book)
3. [Machine Learning Strategy Framework](#machine-learning-strategy-framework)
4. [Real-Time Risk Monitoring](#real-time-risk-monitoring)
5. [Options and Futures Support](#options-and-futures-support)
6. [Live Paper Trading](#live-paper-trading)
7. [Web Dashboard](#web-dashboard)

---

## Multi-Asset Portfolio Optimization

The portfolio optimization module provides modern portfolio theory implementations for optimal asset allocation.

### Mean-Variance Optimization

Optimize portfolio weights for maximum Sharpe ratio or minimum volatility:

```python
from backtesting import PortfolioOptimizer
import pandas as pd

# Load historical returns data
returns_df = pd.read_parquet("data/multi_asset_returns.parquet")

# Create optimizer
optimizer = PortfolioOptimizer(risk_free_rate=0.02)

# Optimize for maximum Sharpe ratio
max_sharpe_result = optimizer.optimize_max_sharpe(returns_df)

print(f"Optimal Weights: {max_sharpe_result['weights']}")
print(f"Expected Return: {max_sharpe_result['expected_return']:.2%}")
print(f"Volatility: {max_sharpe_result['volatility']:.2%}")
print(f"Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.2f}")

# Optimize for minimum volatility
min_vol_result = optimizer.optimize_min_volatility(returns_df)
print(f"\nMin Volatility Weights: {min_vol_result['weights']}")
```

### Efficient Frontier

Calculate and visualize the efficient frontier:

```python
# Calculate efficient frontier
frontier = optimizer.efficient_frontier(returns_df, n_points=50)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(frontier['volatility'], frontier['return'], 'b-', linewidth=2)
plt.scatter(
    max_sharpe_result['volatility'],
    max_sharpe_result['expected_return'],
    marker='*', s=200, c='red', label='Max Sharpe'
)
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)
plt.show()
```

### Risk Parity Optimization

Allocate capital to equalize risk contribution across assets:

```python
from backtesting import RiskParityOptimizer

# Create risk parity optimizer
rp_optimizer = RiskParityOptimizer()

# Optimize
result = rp_optimizer.optimize(returns_df)

print(f"Risk Parity Weights: {result['weights']}")
print(f"Risk Contributions: {result['risk_contributions']}")
```

---

## Event-Driven Simulator with Order Book

Simulate realistic order execution with a limit order book:

```python
from backtesting import EventDrivenSimulator, Order, OrderSide, OrderType
import pandas as pd

# Initialize simulator
sim = EventDrivenSimulator(
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005
)

# Submit a limit buy order
buy_order = Order(
    order_id="",
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=100,
    price=150.0
)
order_id = sim.submit_order(buy_order)

# Process market ticks
timestamp = pd.Timestamp.now()
sim.process_tick("AAPL", 149.5, timestamp)  # Execute if price crosses limit

# Submit market order to close position
sell_order = Order(
    order_id="",
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.MARKET,
    quantity=100
)
sim.submit_order(sell_order)
sim.process_tick("AAPL", 151.0, timestamp)

# Get trades
trades_df = sim.get_trades_df()
print(trades_df)

# Get portfolio value
prices = {"AAPL": 151.0}
portfolio_value = sim.get_portfolio_value(prices)
print(f"Portfolio Value: ${portfolio_value:,.2f}")
```

### Order Book Depth

Access order book market depth:

```python
from backtesting import OrderBook

# Create order book
book = OrderBook("AAPL")

# Add orders
for price in [149.0, 149.5, 150.0]:
    order = Order(
        order_id=f"BUY_{price}",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=price
    )
    book.add_order(order)

# Get market data
print(f"Best Bid: {book.get_best_bid()}")
print(f"Best Ask: {book.get_best_ask()}")
print(f"Spread: {book.get_spread()}")

# Get depth
depth = book.get_depth(levels=5)
print(f"Bids: {depth['bids']}")
print(f"Asks: {depth['asks']}")
```

---

## Machine Learning Strategy Framework

Implement ML-based trading strategies with built-in feature engineering:

### Technical ML Strategy

Use technical indicators as features for ML models:

```python
from backtesting import TechnicalMLStrategy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
df = pd.read_parquet("data/AAPL_daily.parquet")

# Create labels (1 if next day return > 0, else 0)
df['returns'] = df['close'].pct_change()
df['label'] = (df['returns'].shift(-1) > 0).astype(int)

# Create strategy
strategy = TechnicalMLStrategy(
    data=df,
    lookback=30
)

# Train model
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
train_labels = train_data['label']

strategy.train_model(
    train_data,
    train_labels,
    RandomForestClassifier,
    n_estimators=100,
    random_state=42
)

# Generate signals on test data
test_data = df.iloc[train_size:]
strategy.data = test_data
signals = strategy.generate_signals()

print(f"Signals generated: {len(signals)}")
print(f"Buy signals: {(signals == 1).sum()}")
print(f"Sell signals: {(signals == -1).sum()}")
```

### Ensemble Strategy

Combine multiple ML models:

```python
from backtesting import EnsembleStrategy, TechnicalMLStrategy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Create multiple strategies
strategies = []

for model_class in [RandomForestClassifier, GradientBoostingClassifier, LogisticRegression]:
    strategy = TechnicalMLStrategy(data=df)
    strategy.train_model(train_data, train_labels, model_class)
    strategies.append(strategy)

# Create ensemble
ensemble = EnsembleStrategy(
    data=test_data,
    strategies=strategies,
    weights=[0.4, 0.4, 0.2]  # Custom weights
)

# Generate ensemble signals
ensemble_signals = ensemble.generate_signals()
```

### Feature Importance Analysis

Analyze which features are most important:

```python
from backtesting.ml_strategy import FeatureImportanceAnalyzer

# After training a model
analyzer = FeatureImportanceAnalyzer(
    model=strategy.model,
    feature_names=strategy.features
)

# Get importance
importance_df = analyzer.get_feature_importance()
print(importance_df.head(10))

# Plot
analyzer.plot_importance(top_n=15)
```

---

## Real-Time Risk Monitoring

Monitor portfolio risk in real-time with alerts:

```python
from backtesting import RiskMonitor, PositionSizer, StopLossManager
import pandas as pd

# Initialize risk monitor
monitor = RiskMonitor(
    max_drawdown_threshold=0.20,  # 20% max drawdown
    var_confidence=0.95,           # 95% VaR confidence
    position_limit=0.25,           # 25% max position size
    daily_loss_limit=0.05          # 5% max daily loss
)

# Calculate VaR and CVaR
returns = equity_curve.pct_change()
var = monitor.calculate_var(returns)
cvar = monitor.calculate_cvar(returns)
print(f"VaR (95%): {var:.2%}")
print(f"CVaR (95%): {cvar:.2%}")

# Monitor comprehensive risk
positions = {"AAPL": 20000, "MSFT": 15000}
alerts = monitor.monitor(equity_curve, positions, returns)

# Handle alerts
for alert in alerts:
    print(f"ALERT [{alert.level.value}]: {alert.message}")

# Get metrics history
metrics_df = monitor.get_metrics_df()
print(metrics_df.tail())
```

### Position Sizing

Calculate optimal position sizes:

```python
# Initialize position sizer
sizer = PositionSizer(
    risk_per_trade=0.02,    # Risk 2% per trade
    max_position_size=0.25  # Max 25% of portfolio
)

# Kelly Criterion
kelly_size = sizer.kelly_criterion(
    win_rate=0.6,
    avg_win=0.05,
    avg_loss=0.03
)
print(f"Kelly Position Size: {kelly_size:.2%}")

# Volatility-based sizing
position_dollars = sizer.volatility_based_sizing(
    portfolio_value=100_000,
    asset_volatility=0.20,
    target_volatility=0.15
)
print(f"Position Size: ${position_dollars:,.2f}")

# Fixed risk sizing
shares = sizer.fixed_risk_sizing(
    portfolio_value=100_000,
    entry_price=150.0,
    stop_loss_price=145.0
)
print(f"Shares: {shares:.0f}")
```

### Stop Loss Management

Implement dynamic stop losses:

```python
# Initialize stop loss manager
stop_manager = StopLossManager(
    initial_stop_pct=0.05,    # 5% initial stop
    trailing_stop_pct=0.03    # 3% trailing stop
)

# Set initial stop for long position
stop_price = stop_manager.set_initial_stop("AAPL", entry_price=100.0, side="long")
print(f"Initial Stop: ${stop_price:.2f}")

# Update as price moves
new_stop = stop_manager.update_trailing_stop("AAPL", current_price=110.0, side="long")
if new_stop:
    print(f"Trailing Stop Updated: ${new_stop:.2f}")

# Check if stop hit
if stop_manager.check_stop_hit("AAPL", current_price=95.0, side="long"):
    print("Stop loss hit! Exit position")
```

---

## Options and Futures Support

Trade derivatives with Black-Scholes pricing:

### Option Pricing

```python
from backtesting import BlackScholesModel, Option, OptionType
import pandas as pd

# Initialize pricing model
model = BlackScholesModel(risk_free_rate=0.02)

# Price a call option
call_price = model.price_call(
    spot_price=100.0,
    strike=105.0,
    time_to_expiry=0.25,  # 3 months
    volatility=0.20
)
print(f"Call Price: ${call_price:.2f}")

# Calculate Greeks
greeks = model.calculate_greeks(
    option_type=OptionType.CALL,
    spot_price=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    volatility=0.20
)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

### Option Strategies

```python
from backtesting import OptionStrategy

strategy_calculator = OptionStrategy(model)

# Covered Call
covered_call = strategy_calculator.covered_call(
    spot_price=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    volatility=0.20,
    shares=100
)
print(f"Max Profit: ${covered_call['max_profit']:.2f}")
print(f"Breakeven: ${covered_call['breakeven']:.2f}")

# Iron Condor
iron_condor = strategy_calculator.iron_condor(
    spot_price=100.0,
    lower_put_strike=90.0,
    upper_put_strike=95.0,
    lower_call_strike=105.0,
    upper_call_strike=110.0,
    time_to_expiry=0.25,
    volatility=0.20,
    contracts=1
)
print(f"Net Credit: ${iron_condor['net_credit']:.2f}")
print(f"Max Profit: ${iron_condor['max_profit']:.2f}")
```

### Futures Trading

```python
from backtesting import FuturesCalculator, FuturesContract, FuturesType
import pandas as pd

# Create futures contract
contract = FuturesContract(
    symbol="ESZ3",
    underlying="ES",
    contract_type=FuturesType.INDEX,
    expiry=pd.Timestamp('2023-12-15'),
    contract_size=50,  # E-mini S&P 500
    price=4500.0
)

# Calculate margin
calc = FuturesCalculator(margin_rate=0.10)
margin = calc.calculate_margin(contract, quantity=2)
print(f"Margin Required: ${margin:,.2f}")

# Calculate P&L
pnl = calc.calculate_pnl(
    entry_price=4500.0,
    current_price=4550.0,
    contract_size=50,
    quantity=2
)
print(f"P&L: ${pnl:,.2f}")
```

---

## Live Paper Trading

Test strategies in real-time with simulated execution:

```python
import asyncio
from backtesting import PaperTradingEngine

async def run_paper_trading():
    # Initialize engine
    engine = PaperTradingEngine(
        symbols=["AAPL", "MSFT", "GOOGL"],
        initial_cash=100_000
    )
    
    # Run for 1 hour
    await engine.start(duration_seconds=3600)
    
    # Export results
    engine.export_results("paper_trading_results.json")

# Run
asyncio.run(run_paper_trading())
```

### Custom Strategy Integration

```python
class MyPaperStrategy:
    def generate_orders(self, prices, account):
        """Generate orders based on current market prices."""
        orders = []
        
        for symbol, price in prices.items():
            # Simple example: buy if we don't have position
            if account.positions.get(symbol, 0) == 0:
                if account.cash > price * 100:
                    orders.append({
                        'symbol': symbol,
                        'side': OrderSide.BUY,
                        'order_type': OrderType.MARKET,
                        'quantity': 100
                    })
        
        return orders

# Use with engine
engine = PaperTradingEngine(
    symbols=["AAPL", "MSFT"],
    initial_cash=100_000,
    strategy=MyPaperStrategy()
)
```

---

## Web Dashboard

Monitor your trading strategies with a real-time web dashboard:

### Starting the Dashboard

```python
from backtesting import DashboardServer

# Initialize server
server = DashboardServer(host="0.0.0.0", port=5000, debug=False)

# Load existing data (optional)
server.load_data("paper_trading_results.json")

# Start server
server.run()
```

### Updating Dashboard Data

```python
import requests

# Update dashboard with new data
data = {
    "performance": {
        "timestamp": "2023-10-20T10:30:00",
        "portfolio_value": 105000,
        "cash": 50000,
        "pnl": 5000,
        "return_pct": 5.0
    },
    "positions": {
        "AAPL": 100,
        "MSFT": 50
    },
    "risk": {
        "var": -0.02,
        "max_drawdown": -0.05
    }
}

requests.post("http://localhost:5000/api/update", json=data)
```

### Dashboard Features

- **Real-time portfolio value chart** - Visualize performance over time
- **Position monitoring** - Track current holdings
- **Risk metrics** - Monitor VaR, drawdown, and other risk measures
- **Order history** - Review all executed trades
- **Auto-refresh** - Updates every 5 seconds

Access the dashboard at: `http://localhost:5000`

---

## Integration Examples

### Complete Trading System

```python
from backtesting import (
    TechnicalMLStrategy, BacktestEngine,
    RiskMonitor, PositionSizer, StopLossManager,
    PaperTradingEngine, DashboardServer
)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Train ML strategy
df = pd.read_parquet("data/AAPL_daily.parquet")
df['label'] = (df['close'].pct_change().shift(-1) > 0).astype(int)

strategy = TechnicalMLStrategy(data=df)
strategy.train_model(
    df.iloc[:800],
    df.iloc[:800]['label'],
    RandomForestClassifier,
    n_estimators=100
)

# 2. Backtest
engine = BacktestEngine(strategy, initial_cash=100_000)
summary = engine.run()
print(f"Backtest Return: {summary['total_return']:.2%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")

# 3. Risk monitoring
monitor = RiskMonitor()
equity_curve = engine.get_equity_curve()
returns = engine.get_returns()
alerts = monitor.monitor(equity_curve, {}, returns)

if alerts:
    print(f"⚠️ {len(alerts)} risk alerts")

# 4. Paper trading (if backtest looks good)
# async def run_live():
#     paper_engine = PaperTradingEngine(
#         symbols=["AAPL"],
#         initial_cash=100_000,
#         strategy=strategy
#     )
#     await paper_engine.start(duration_seconds=3600)

# 5. Start dashboard
# dashboard = DashboardServer()
# dashboard.run()
```

---

## Best Practices

1. **Always backtest first** - Test strategies on historical data before paper trading
2. **Use risk monitoring** - Set appropriate thresholds for drawdown, position size, and daily loss
3. **Diversify** - Use portfolio optimization to properly allocate across assets
4. **Validate ML models** - Use walk-forward validation and out-of-sample testing
5. **Monitor in real-time** - Use the dashboard to track strategy performance
6. **Start small** - Begin with paper trading before risking real capital
7. **Review regularly** - Analyze alerts and metrics to improve strategies

---

## Performance Tips

- Use vectorized backtesting for faster strategy testing
- Leverage multi-processing for data processing
- Cache ML model predictions
- Use appropriate database indexes for time-series queries
- Monitor memory usage when processing large datasets

---

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
2. **Memory issues**: Reduce batch sizes or use chunking for large datasets
3. **Slow backtests**: Switch to vectorized backtesting for simple strategies
4. **ML model not converging**: Adjust hyperparameters or try different algorithms
5. **Dashboard not loading**: Check that Flask is installed and port 5000 is available

For more help, see the main documentation or open an issue on GitHub.
