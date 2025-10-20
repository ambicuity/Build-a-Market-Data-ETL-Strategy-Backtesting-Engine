# Architecture Documentation

## Overview

The Market Data ETL & Strategy Backtesting Engine is designed as a high-performance, scalable system for ingesting, processing, and analyzing financial market data. The architecture follows a modular design with three main components:

1. **Real-Time ETL Pipeline** - Ingests live market data via WebSocket
2. **Data Processing** - Cleans, transforms, and aggregates historical data
3. **Backtesting Engine** - Tests trading strategies with vectorized operations

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Market Data Sources                          │
│              (WebSocket Feeds, Historical Files)                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ETL Pipeline Layer                           │
│  ┌───────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  WebSocket    │  │ Normalizer │  │  Database Writer     │  │
│  │  Client       │─▶│            │─▶│  (Async Batching)    │  │
│  │  (Async)      │  │            │  │                      │  │
│  └───────────────┘  └────────────┘  └──────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Time-Series Database                            │
│         (PostgreSQL/TimescaleDB, InfluxDB, QuestDB)             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                Data Processing Layer                             │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │  Cleaner    │  │ Resampler│  │  Tick-to-OHLCV           │  │
│  │             │─▶│          │─▶│  Converter               │  │
│  │ (Polars)    │  │ (Polars) │  │  (Parallel Processing)   │  │
│  └─────────────┘  └──────────┘  └──────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Backtesting Engine                              │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │  Strategy   │  │ Portfolio│  │  Performance Metrics     │  │
│  │  (Signals)  │─▶│ Manager  │─▶│  & Visualization         │  │
│  │             │  │          │  │                          │  │
│  └─────────────┘  └──────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ETL Pipeline

#### WebSocket Client (`etl/websocket_client.py`)
- **Purpose**: Real-time data ingestion from WebSocket feeds
- **Technology**: `asyncio`, `aiohttp`
- **Features**:
  - Automatic reconnection with exponential backoff
  - Heartbeat mechanism to keep connection alive
  - Backpressure handling with bounded queues
  - Concurrent message processing
  - Statistics tracking (messages/sec, errors, reconnects)

#### Normalizer (`etl/normalizer.py`)
- **Purpose**: Convert raw feed data into standardized format
- **Features**:
  - Support for multiple feed formats
  - Timestamp parsing (Unix, ISO 8601)
  - Error handling and logging
  - Trade and quote normalization

#### Database Writer (`etl/database_writer.py`)
- **Purpose**: Asynchronous database writes with batching
- **Technology**: `asyncpg` for PostgreSQL
- **Features**:
  - Connection pooling
  - Batch inserts for performance
  - Automatic table creation
  - Buffer management
  - Retry logic on failures

#### Configuration (`etl/config.py`)
- **Purpose**: Centralized configuration management
- **Features**:
  - YAML-based configuration
  - Default values
  - Dot notation for nested access
  - Environment-specific settings

### 2. Data Processing

#### Cleaner (`data_processing/cleaner.py`)
- **Purpose**: Data quality and validation
- **Features**:
  - Outlier detection (IQR, Z-score methods)
  - Duplicate removal
  - Missing timestamp handling
  - Price validation
  - Support for Pandas and Polars

#### Resampler (`data_processing/resampler.py`)
- **Purpose**: Time-series frequency conversion
- **Technology**: Polars for performance
- **Features**:
  - Multi-frequency support (1min, 5min, 1H, 1D, etc.)
  - OHLCV aggregation
  - Parallel processing with ProcessPoolExecutor
  - Batch file processing
  - Memory-efficient chunked operations

#### Tick-to-OHLCV Converter (`data_processing/tick_to_ohlcv.py`)
- **Purpose**: Convert tick data to OHLCV bars
- **Features**:
  - Vectorized operations
  - Compressed Parquet output
  - Batch processing for large datasets
  - Support for custom aggregation windows

### 3. Backtesting Engine

#### Strategy (`backtesting/strategy.py`)
- **Purpose**: Define trading logic
- **Base Classes**:
  - `BaseStrategy` - Abstract base for all strategies
  - `MeanReversionStrategy` - Mean reversion with z-score
  - `MovingAverageCrossStrategy` - MA crossover
  - `MomentumStrategy` - Momentum-based
  - `BuyAndHoldStrategy` - Benchmark strategy
- **Features**:
  - Signal generation (-1, 0, 1)
  - Position sizing
  - Extensible framework

#### Portfolio (`backtesting/portfolio.py`)
- **Two Implementations**:
  1. **Portfolio** - Event-driven for realistic simulation
  2. **VectorizedPortfolio** - Fast vectorized operations
- **Features**:
  - Commission and slippage modeling
  - Position tracking
  - Trade history
  - Multi-asset support
  - Risk management

#### Engine (`backtesting/engine.py`)
- **Purpose**: Execute backtests
- **Modes**:
  - **Vectorized** - Fast, uses NumPy operations
  - **Event-driven** - More realistic, bar-by-bar simulation
- **Features**:
  - Strategy execution
  - Performance metrics calculation
  - Results aggregation

#### Metrics (`backtesting/metrics.py`)
- **Purpose**: Calculate performance metrics
- **Metrics**:
  - Returns: Total, CAGR, Volatility
  - Risk-adjusted: Sharpe, Sortino, Calmar
  - Drawdown: Maximum, underwater periods
  - Trading: Win rate, profit factor, exposure
  - Advanced: Consecutive wins/losses, average win/loss
- **Features**:
  - Vectorized calculations
  - Annualization support
  - Risk-free rate adjustment

#### Visualization (`backtesting/visualization.py`)
- **Purpose**: Create charts and reports
- **Technologies**: Matplotlib, Plotly
- **Features**:
  - Equity curve plots
  - Drawdown charts
  - Returns distribution
  - Interactive plots (Plotly)
  - HTML report generation

## Data Flow

### Real-Time Ingestion Flow

```
WebSocket Feed → LiveETL → Queue → Normalizer → Buffer → DatabaseWriter → TimescaleDB
     │              │         │         │          │           │
     │              │         └─────────┴──────────┘           │
     │              │         Backpressure Control             │
     │              └──────────────────────────────────────────┘
     └─ Reconnect Logic, Heartbeat, Statistics
```

### Historical Processing Flow

```
Raw Tick Data → Cleaner → Resampler → Tick-to-OHLCV → Parquet Files
     │            │          │              │
     CSV/Parquet  │          │              └─ Compressed Storage
     500GB+       │          └─ Parallel Processing (Multi-core)
                  └─ Outlier Removal, Deduplication
```

### Backtesting Flow

```
Historical OHLCV → Strategy → Signals → Portfolio → Metrics → Visualization
     │               │          │          │          │           │
     Parquet Files   │          │          │          │           └─ Reports
     (Fast Load)     │          │          │          └─ Sharpe, Sortino, etc.
                     │          │          └─ Commission, Slippage
                     └─ User-defined Logic
```

## Performance Characteristics

### ETL Pipeline
- **Target**: 10,000+ ticks/second sustained
- **Latency**: <10ms per message
- **Memory**: Bounded by queue size
- **Scalability**: Horizontal (multiple symbols per worker)

### Data Processing
- **Throughput**: 500GB+ in parallel
- **Memory**: <60% system RAM with chunking
- **Parallelism**: Multi-process for CPU-bound operations

### Backtesting
- **Speed**: 50ms for vectorized backtest on 1 year of daily data
- **Accuracy**: Identical to event-driven for simple strategies
- **Scalability**: Multi-asset, multi-strategy support

## Technology Stack

### Core Libraries
- **Async I/O**: `asyncio`, `aiohttp`
- **Database**: `asyncpg`, `influxdb-client`
- **Data Processing**: `pandas`, `polars`, `numpy`
- **Parallel**: `multiprocessing`, `dask`
- **Visualization**: `matplotlib`, `plotly`
- **Testing**: `pytest`, `pytest-asyncio`

### Database Options
- **PostgreSQL/TimescaleDB** - Recommended for general use
- **InfluxDB** - Alternative for time-series
- **QuestDB** - High-performance time-series

## Deployment Considerations

### Requirements
- Python 3.9+
- 8GB+ RAM (16GB+ recommended for large datasets)
- Multi-core CPU for parallel processing
- SSD for fast I/O

### Scaling
- **Vertical**: More CPU cores for parallel processing
- **Horizontal**: Multiple ETL workers for different symbols
- **Database**: Partitioning by time for large datasets

### Monitoring
- ETL statistics (messages/sec, errors)
- Database connection pool metrics
- Memory usage tracking
- Processing time benchmarks

## Security

### Best Practices
- Configuration files should not contain credentials
- Use environment variables for sensitive data
- Database connection pooling with limits
- Input validation in normalizer
- Rate limiting for WebSocket connections

### Data Validation
- Price range validation
- Volume sanity checks
- Timestamp ordering
- Duplicate detection
