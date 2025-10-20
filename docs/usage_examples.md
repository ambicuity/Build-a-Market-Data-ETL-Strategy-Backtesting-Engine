# Usage Examples

This document provides practical examples of using the Market Data ETL & Strategy Backtesting Engine.

## Table of Contents
1. [ETL Pipeline](#etl-pipeline)
2. [Data Processing](#data-processing)
3. [Backtesting](#backtesting)
4. [Tools](#tools)

## ETL Pipeline

### Running the ETL Pipeline

```python
import asyncio
from etl import ETLPipeline

async def main():
    # Initialize pipeline with config
    pipeline = ETLPipeline("config/settings.yaml")
    
    # Run the pipeline
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom WebSocket Handler

```python
import asyncio
from etl import LiveETL, DatabaseWriter

async def main():
    # Initialize database writer
    db_writer = DatabaseWriter(
        host="localhost",
        database="market_data",
        user="postgres",
        password="password",
    )
    await db_writer.connect()
    
    # Initialize ETL client
    etl = LiveETL(
        ws_url="wss://your-feed.com/stream",
        db_writer=db_writer,
        symbols=["AAPL", "MSFT", "GOOGL"],
        queue_size=10000,
    )
    
    # Start streaming
    await etl.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Message Normalization

```python
from etl import normalize_trade

# Example raw message
raw_message = {
    "type": "trade",
    "data": {
        "t": 1609459200000,  # Unix timestamp
        "s": "AAPL",         # Symbol
        "p": 150.50,         # Price
        "v": 100,            # Volume
    }
}

# Normalize
normalized = normalize_trade(raw_message)
print(normalized)
# Output: {'timestamp': '2021-01-01T00:00:00', 'symbol': 'AAPL', 'price': 150.5, 'volume': 100}
```

## Data Processing

### Cleaning Data

```python
from data_processing import DataCleaner
import pandas as pd

# Load data
df = pd.read_csv("raw_ticks.csv")

# Initialize cleaner
cleaner = DataCleaner()

# Run cleaning pipeline
clean_df = cleaner.clean_pipeline_pandas(
    df,
    remove_outliers=True,
    deduplicate=True,
    validate_prices=True,
)

print(f"Original: {len(df)} rows, Clean: {len(clean_df)} rows")
```

### Converting Ticks to OHLCV

```python
from data_processing import TickToOHLCV

# Initialize converter
converter = TickToOHLCV(use_polars=True)

# Convert single file
converter.convert_file(
    input_path="data/AAPL_ticks.parquet",
    output_path="data/AAPL_ohlcv_1min.parquet",
    freq="1min",
)

# Batch convert directory
converter.batch_convert(
    input_dir="data/ticks",
    output_dir="data/ohlcv",
    freq="1min",
    pattern="*.parquet",
)
```

### Resampling Time Series

```python
from data_processing import Resampler
import pandas as pd

# Load OHLCV data
df = pd.read_parquet("data/AAPL_ohlcv_1min.parquet")

# Initialize resampler
resampler = Resampler()

# Resample to 5-minute bars
df_5min = resampler.resample(df, freq="5min")

# Resample to hourly bars
df_1h = resampler.resample(df, freq="1H")

# Save resampled data
df_5min.to_parquet("data/AAPL_ohlcv_5min.parquet")
```

### Parallel Processing

```python
from data_processing import Resampler

# Use multiple workers for parallel processing
resampler = Resampler(n_workers=8)

# Batch resample directory
resampler.batch_resample(
    input_dir="data/ohlcv_1min",
    output_dir="data/ohlcv_5min",
    freq="5min",
    parallel=True,
)
```

## Backtesting

### Simple Buy-and-Hold Strategy

```python
from backtesting import BuyAndHoldStrategy, BacktestEngine
import pandas as pd

# Load historical data
df = pd.read_parquet("data/AAPL_ohlcv_1D.parquet")

# Create strategy
strategy = BuyAndHoldStrategy(df, name="AAPL Buy & Hold")

# Create engine
engine = BacktestEngine(
    strategy=strategy,
    initial_cash=100000,
    commission=0.001,
    slippage=0.0005,
)

# Run backtest
summary = engine.run()

# Print results
print(f"Total Return: {summary['total_return']:.2%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
```

### Custom Strategy

```python
from backtesting import BaseStrategy, BacktestEngine
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    """Custom RSI-based strategy."""
    
    def __init__(self, data, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        super().__init__(data, name="RSI Strategy")
    
    def calculate_rsi(self):
        """Calculate RSI indicator."""
        delta = self.data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self):
        """Generate trading signals based on RSI."""
        rsi = self.calculate_rsi()
        
        signals = pd.Series(0, index=self.data.index)
        signals[rsi < self.rsi_oversold] = 1   # Buy when oversold
        signals[rsi > self.rsi_overbought] = -1  # Sell when overbought
        
        return signals

# Load data
df = pd.read_parquet("data/AAPL_ohlcv_1D.parquet")

# Create and run backtest
strategy = MyCustomStrategy(df, rsi_period=14)
engine = BacktestEngine(strategy, initial_cash=100000)
summary = engine.run()

print(summary)
```

### Mean Reversion Strategy

```python
from backtesting import MeanReversionStrategy, BacktestEngine
import pandas as pd

# Load data
df = pd.read_parquet("data/AAPL_ohlcv_1D.parquet")

# Create mean reversion strategy
strategy = MeanReversionStrategy(
    data=df,
    window=20,      # 20-day rolling window
    num_std=2.0,    # 2 standard deviations
)

# Run backtest
engine = BacktestEngine(strategy, initial_cash=100000)
summary = engine.run()

# Get detailed metrics
metrics = engine.metrics
print(metrics.print_summary())
```

### Comparing Multiple Strategies

```python
from backtesting import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MovingAverageCrossStrategy,
    BacktestEngine,
)
import pandas as pd

# Load data
df = pd.read_parquet("data/AAPL_ohlcv_1D.parquet")

# Define strategies
strategies = [
    BuyAndHoldStrategy(df.copy()),
    MeanReversionStrategy(df.copy(), window=20, num_std=2.0),
    MovingAverageCrossStrategy(df.copy(), fast_window=10, slow_window=50),
]

# Run backtests
results = []
for strategy in strategies:
    engine = BacktestEngine(strategy, initial_cash=100000)
    summary = engine.run()
    results.append(summary)

# Compare results
comparison = pd.DataFrame(results)
print(comparison[["strategy", "total_return", "sharpe_ratio", "max_drawdown"]])
```

### Visualization

```python
from backtesting import BuyAndHoldStrategy, BacktestEngine, Visualizer
import pandas as pd

# Run backtest
df = pd.read_parquet("data/AAPL_ohlcv_1D.parquet")
strategy = BuyAndHoldStrategy(df)
engine = BacktestEngine(strategy)
engine.run()

# Create visualizer
viz = Visualizer(engine.results)

# Plot equity curve
viz.plot_equity_curve(
    title="AAPL Buy & Hold",
    save_path="results/equity_curve.png",
)

# Plot drawdown
viz.plot_drawdown(
    title="Drawdown Analysis",
    save_path="results/drawdown.png",
)

# Create overview
viz.plot_overview(
    save_path="results/overview.png",
)

# Create interactive plot
viz.plot_interactive(
    title="Interactive Backtest Results",
    save_path="results/interactive.html",
)

# Generate full report
viz.create_report(
    output_dir="results/report",
    strategy_name="AAPL Buy & Hold",
)
```

## Tools

### Generating Mock Data

```python
# Generate mock tick data for testing
from tools.generate_mock_ticks import generate_multiple_symbols

generate_multiple_symbols(
    symbols=["AAPL", "MSFT", "GOOGL"],
    n_ticks=100000,
    output_dir="data/mock_ticks",
)
```

Command line:
```bash
python tools/generate_mock_ticks.py --mode multi --symbols AAPL MSFT GOOGL --n-ticks 100000
```

### Benchmarking Performance

```bash
# Benchmark data loading
python tools/benchmark_loader.py data/AAPL_ticks.parquet --format parquet --agg

# Generate large dataset
python tools/generate_mock_ticks.py --mode large --n-days 365 --n-ticks 10000
```

## Complete Workflow Example

```python
import asyncio
import pandas as pd
from data_processing import TickToOHLCV, DataCleaner, Resampler
from backtesting import MeanReversionStrategy, BacktestEngine, Visualizer

# Step 1: Generate mock data (for testing)
from tools.generate_mock_ticks import generate_mock_ticks

df_ticks = generate_mock_ticks(symbol="AAPL", n_ticks=100000)
df_ticks.to_parquet("data/AAPL_ticks.parquet")

# Step 2: Clean data
cleaner = DataCleaner()
df_clean = cleaner.clean(df_ticks)

# Step 3: Convert to OHLCV
converter = TickToOHLCV()
df_ohlcv = converter.convert(df_clean, freq="1min")

# Step 4: Resample to daily
resampler = Resampler()
df_daily = resampler.resample(df_ohlcv, freq="1D")

# Step 5: Run backtest
strategy = MeanReversionStrategy(df_daily, window=20, num_std=2.0)
engine = BacktestEngine(strategy, initial_cash=100000)
summary = engine.run()

# Step 6: Visualize results
viz = Visualizer(engine.results)
viz.create_report(output_dir="results", strategy_name="Mean Reversion")

# Print summary
print("\nBacktest Summary:")
for key, value in summary.items():
    if isinstance(value, float):
        if "return" in key or "drawdown" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")
```

## Advanced Examples

### Event-Driven Backtest

```python
from backtesting.engine import EventDrivenEngine
from backtesting import BuyAndHoldStrategy
import pandas as pd

# Load data
df = pd.read_parquet("data/AAPL_ohlcv_1D.parquet")

# Create strategy
strategy = BuyAndHoldStrategy(df)

# Use event-driven engine for more realistic simulation
engine = EventDrivenEngine(strategy, initial_cash=100000)
results = engine.run()

print(results)
```

### Multi-Asset Portfolio

```python
from backtesting.portfolio import VectorizedPortfolio
import pandas as pd

# Load data for multiple assets
prices_df = pd.DataFrame({
    "AAPL": pd.read_parquet("data/AAPL_ohlcv_1D.parquet")["close"],
    "MSFT": pd.read_parquet("data/MSFT_ohlcv_1D.parquet")["close"],
    "GOOGL": pd.read_parquet("data/GOOGL_ohlcv_1D.parquet")["close"],
})

# Create signals (e.g., always long)
signals_df = pd.DataFrame(1, index=prices_df.index, columns=prices_df.columns)

# Run backtest
portfolio = VectorizedPortfolio(initial_cash=100000)
results = portfolio.backtest_multi_asset(prices_df, signals_df)

print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
```
