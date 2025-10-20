# API Reference

Complete API reference for the Market Data ETL & Strategy Backtesting Engine.

## ETL Module

### Config

```python
class Config(config_path: str = None)
```

Configuration loader and manager.

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file

**Methods:**
- `load()`: Load configuration from file
- `save()`: Save configuration to file
- `get(key, default=None)`: Get configuration value
- `set(key, value)`: Set configuration value

**Properties:**
- `websocket_url`: WebSocket URL
- `symbols`: List of symbols to track
- `batch_size`: Batch size for database writes
- `queue_size`: Queue size for backpressure

### LiveETL

```python
class LiveETL(
    ws_url: str,
    db_writer: DatabaseWriter,
    symbols: List[str],
    queue_size: int = 10000,
    reconnect_delay: int = 5,
    heartbeat_interval: int = 30,
    message_handler: Optional[Callable] = None
)
```

Real-time ETL client for WebSocket market data feeds.

**Parameters:**
- `ws_url`: WebSocket URL
- `db_writer`: DatabaseWriter instance
- `symbols`: List of symbols to subscribe to
- `queue_size`: Maximum queue size
- `reconnect_delay`: Delay between reconnection attempts (seconds)
- `heartbeat_interval`: Heartbeat interval (seconds)
- `message_handler`: Custom message handler function

**Methods:**
- `async start()`: Start the ETL pipeline
- `async stop()`: Stop the ETL pipeline

**Properties:**
- `stats`: Current statistics dictionary

### DatabaseWriter

```python
class DatabaseWriter(
    host: str = "localhost",
    port: int = 5432,
    database: str = "market_data",
    user: str = "postgres",
    password: str = "password",
    pool_size: int = 10,
    batch_size: int = 1000
)
```

Asynchronous database writer with connection pooling.

**Methods:**
- `async connect()`: Create connection pool
- `async disconnect()`: Close connection pool
- `async write(record)`: Write single record (buffered)
- `async write_batch(records)`: Write multiple records
- `async flush()`: Flush buffered records
- `async get_latest_trades(symbol, limit=100)`: Get latest trades

### normalize_trade

```python
def normalize_trade(raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]
```

Normalize raw trade data into standard format.

**Parameters:**
- `raw_data`: Raw trade data from WebSocket feed

**Returns:**
- Normalized trade record with keys: timestamp, symbol, price, volume
- `None` if data is invalid

### normalize_quote

```python
def normalize_quote(raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]
```

Normalize raw quote data into standard format.

**Parameters:**
- `raw_data`: Raw quote data from WebSocket feed

**Returns:**
- Normalized quote record with bid/ask prices and sizes
- `None` if data is invalid

## Data Processing Module

### DataCleaner

```python
class DataCleaner()
```

Clean and validate market data.

**Methods:**
- `remove_outliers_pandas(df, column="price", method="iqr", threshold=3.0)`: Remove outliers
- `deduplicate_pandas(df, subset=None, keep="first")`: Remove duplicates
- `fill_missing_timestamps_pandas(df, freq="1min", method="ffill")`: Fill missing timestamps
- `validate_prices_pandas(df, price_col="price", min_price=0.01, max_price=None)`: Validate prices
- `clean_pipeline_pandas(df, remove_outliers=True, deduplicate=True, validate_prices=True)`: Run complete pipeline
- `clean(df, **kwargs)`: Auto-detect library and clean

### TickToOHLCV

```python
class TickToOHLCV(use_polars: bool = True)
```

Convert tick-level data to OHLCV bars.

**Parameters:**
- `use_polars`: Use Polars for better performance

**Methods:**
- `convert_pandas(df, freq="1min", volume_col="volume")`: Convert using Pandas
- `convert_polars(df, freq="1m", volume_col="volume")`: Convert using Polars
- `convert(data, freq="1min", volume_col="volume")`: Auto-detect library
- `convert_file(input_path, output_path, freq="1min", chunk_size=None)`: Convert file
- `batch_convert(input_dir, output_dir, freq="1min", pattern="*.csv")`: Batch convert

### Resampler

```python
class Resampler(n_workers: Optional[int] = None)
```

Resample time-series market data to different frequencies.

**Parameters:**
- `n_workers`: Number of parallel workers (default: CPU count)

**Methods:**
- `resample_pandas(df, freq, agg_dict=None)`: Resample using Pandas
- `resample_polars(df, freq, agg_exprs=None)`: Resample using Polars
- `resample(df, freq, **kwargs)`: Auto-detect library
- `resample_file(input_path, output_path, freq)`: Resample file
- `batch_resample(input_dir, output_dir, freq, pattern="*.parquet", parallel=True)`: Batch resample
- `downsample_ohlcv(df, from_freq, to_freq)`: Downsample OHLCV data

## Backtesting Module

### BaseStrategy

```python
class BaseStrategy(data: pd.DataFrame, name: str = "Strategy")
```

Base class for trading strategies.

**Parameters:**
- `data`: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
- `name`: Strategy name

**Methods:**
- `generate_signals()`: Generate trading signals (abstract method)
- `get_signals()`: Get cached signals or generate new ones
- `calculate_positions(signals=None)`: Calculate position sizes from signals

**Attributes:**
- `data`: Strategy data DataFrame
- `name`: Strategy name

### BuyAndHoldStrategy

```python
class BuyAndHoldStrategy(data: pd.DataFrame, name: str = "BuyAndHold")
```

Simple buy and hold strategy.

### MeanReversionStrategy

```python
class MeanReversionStrategy(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    name: str = "MeanReversion"
)
```

Mean reversion strategy using z-score.

**Parameters:**
- `window`: Rolling window for mean/std calculation
- `num_std`: Number of standard deviations for entry signal

### MovingAverageCrossStrategy

```python
class MovingAverageCrossStrategy(
    data: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    name: str = "MA_Cross"
)
```

Moving average crossover strategy.

**Parameters:**
- `fast_window`: Fast MA window
- `slow_window`: Slow MA window

### Portfolio

```python
class Portfolio(
    initial_cash: float = 1_000_000,
    commission: float = 0.001,
    slippage: float = 0.0005
)
```

Portfolio manager for backtesting (event-driven).

**Methods:**
- `reset()`: Reset portfolio to initial state
- `execute_trade(symbol, quantity, price, timestamp=None)`: Execute a trade
- `get_position(symbol)`: Get current position
- `calculate_portfolio_value(prices)`: Calculate total portfolio value
- `get_history_df()`: Get trade history as DataFrame

### VectorizedPortfolio

```python
class VectorizedPortfolio(
    initial_cash: float = 1_000_000,
    commission: float = 0.001,
    slippage: float = 0.0005
)
```

Vectorized portfolio for fast backtesting.

**Methods:**
- `backtest_signals(prices, signals, position_size=1.0)`: Backtest trading signals
- `backtest_multi_asset(prices_df, signals_df, weights=None)`: Backtest multiple assets

### BacktestEngine

```python
class BacktestEngine(
    strategy: BaseStrategy,
    initial_cash: float = 1_000_000,
    commission: float = 0.001,
    slippage: float = 0.0005
)
```

High-performance vectorized backtesting engine.

**Methods:**
- `run()`: Run the backtest
- `get_summary()`: Get backtest summary
- `get_equity_curve()`: Get equity curve
- `get_returns()`: Get returns series
- `get_positions()`: Get positions over time

### PerformanceMetrics

```python
class PerformanceMetrics(
    results: pd.DataFrame,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
)
```

Calculate performance metrics for backtesting.

**Methods:**
- `total_return()`: Total return
- `cagr()`: Compound Annual Growth Rate
- `volatility(annualized=True)`: Volatility
- `sharpe_ratio(risk_free_rate=None)`: Sharpe Ratio
- `sortino_ratio(risk_free_rate=None)`: Sortino Ratio
- `max_drawdown()`: Maximum drawdown
- `calmar_ratio()`: Calmar Ratio
- `win_rate()`: Win rate
- `profit_factor()`: Profit factor
- `num_trades()`: Number of trades
- `max_consecutive_wins()`: Maximum consecutive wins
- `max_consecutive_losses()`: Maximum consecutive losses
- `exposure()`: Market exposure
- `average_win()`: Average winning return
- `average_loss()`: Average losing return
- `get_all_metrics()`: Get all metrics as dictionary
- `print_summary()`: Print metrics summary

### Visualizer

```python
class Visualizer(results: pd.DataFrame)
```

Create visualizations for backtesting results.

**Methods:**
- `plot_equity_curve(figsize=(12, 6), title="Equity Curve", show=True, save_path=None)`: Plot equity curve
- `plot_drawdown(figsize=(12, 6), title="Drawdown", show=True, save_path=None)`: Plot drawdown
- `plot_returns_distribution(figsize=(12, 6), bins=50, show=True, save_path=None)`: Plot returns distribution
- `plot_overview(figsize=(15, 10), show=True, save_path=None)`: Plot overview
- `plot_interactive(title="Backtest Results", save_path=None)`: Create interactive plot
- `create_report(output_dir="backtest_report", strategy_name="Strategy")`: Create comprehensive report

## Tools

### generate_mock_ticks

```python
def generate_mock_ticks(
    symbol: str = "AAPL",
    start_price: float = 150.0,
    n_ticks: int = 100000,
    start_time: datetime = None,
    tick_interval_ms: int = 100,
    volatility: float = 0.02,
    trend: float = 0.0001
) -> pd.DataFrame
```

Generate mock tick data for testing.

**Parameters:**
- `symbol`: Stock symbol
- `start_price`: Starting price
- `n_ticks`: Number of ticks to generate
- `start_time`: Start timestamp
- `tick_interval_ms`: Interval between ticks in milliseconds
- `volatility`: Price volatility
- `trend`: Price trend per tick

**Returns:**
- DataFrame with mock tick data

## Constants

### Frequency Mappings

**Pandas:**
- `"1min"` - 1 minute
- `"5min"` - 5 minutes
- `"15min"` - 15 minutes
- `"30min"` - 30 minutes
- `"1H"` - 1 hour
- `"1D"` - 1 day

**Polars:**
- `"1m"` - 1 minute
- `"5m"` - 5 minutes
- `"15m"` - 15 minutes
- `"30m"` - 30 minutes
- `"1h"` - 1 hour
- `"1d"` - 1 day

### Signal Values

- `1` - Long position
- `0` - Neutral (no position)
- `-1` - Short position

## Error Handling

All async methods may raise:
- `ConnectionError`: Database/WebSocket connection issues
- `TimeoutError`: Operation timeout
- `ValueError`: Invalid parameters

All data processing methods may raise:
- `ValueError`: Invalid data format
- `KeyError`: Missing required columns
- `FileNotFoundError`: Input file not found
