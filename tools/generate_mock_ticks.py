"""Generate mock tick data for testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def generate_mock_ticks(
    symbol: str = "AAPL",
    start_price: float = 150.0,
    n_ticks: int = 100000,
    start_time: datetime = None,
    tick_interval_ms: int = 100,
    volatility: float = 0.02,
    trend: float = 0.0001,
) -> pd.DataFrame:
    """Generate mock tick data.
    
    Args:
        symbol: Stock symbol
        start_price: Starting price
        n_ticks: Number of ticks to generate
        start_time: Start timestamp
        tick_interval_ms: Interval between ticks in milliseconds
        volatility: Price volatility
        trend: Price trend per tick
        
    Returns:
        DataFrame with mock tick data
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(days=30)
    
    # Generate timestamps
    timestamps = [
        start_time + timedelta(milliseconds=i * tick_interval_ms)
        for i in range(n_ticks)
    ]
    
    # Generate price series with random walk
    np.random.seed(42)
    price_changes = np.random.normal(trend, volatility, n_ticks)
    prices = start_price * np.exp(np.cumsum(price_changes))
    
    # Add some jumps for realism
    jump_indices = np.random.choice(n_ticks, size=n_ticks // 100, replace=False)
    jump_sizes = np.random.normal(0, volatility * 5, len(jump_indices))
    prices[jump_indices] *= (1 + jump_sizes)
    
    # Generate volumes (log-normal distribution)
    volumes = np.random.lognormal(mean=3, sigma=1, size=n_ticks)
    volumes = np.round(volumes * 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "price": prices,
        "volume": volumes,
    })
    
    return df


def generate_multiple_symbols(
    symbols: list,
    n_ticks: int = 100000,
    output_dir: str = "data/mock_ticks",
) -> None:
    """Generate mock tick data for multiple symbols.
    
    Args:
        symbols: List of symbols
        n_ticks: Number of ticks per symbol
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_prices = {
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOGL": 120.0,
        "AMZN": 130.0,
        "TSLA": 200.0,
    }
    
    for symbol in symbols:
        print(f"Generating mock ticks for {symbol}...")
        
        start_price = start_prices.get(symbol, 100.0)
        
        df = generate_mock_ticks(
            symbol=symbol,
            start_price=start_price,
            n_ticks=n_ticks,
        )
        
        # Save to CSV
        output_file = output_path / f"{symbol}_ticks.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} ticks to {output_file}")
        
        # Also save as Parquet for faster loading
        output_file_parquet = output_path / f"{symbol}_ticks.parquet"
        df.to_parquet(output_file_parquet, compression="zstd", index=False)
        print(f"Saved Parquet to {output_file_parquet}")


def generate_large_dataset(
    symbol: str = "AAPL",
    n_days: int = 365,
    ticks_per_day: int = 10000,
    output_file: str = "data/large_ticks.parquet",
) -> None:
    """Generate large tick dataset for performance testing.
    
    Args:
        symbol: Stock symbol
        n_days: Number of days
        ticks_per_day: Ticks per day
        output_file: Output file path
    """
    print(f"Generating large dataset: {n_days} days, {ticks_per_day} ticks/day...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    for day in range(n_days):
        start_time = datetime(2023, 1, 1) + timedelta(days=day)
        
        df = generate_mock_ticks(
            symbol=symbol,
            start_time=start_time,
            n_ticks=ticks_per_day,
            tick_interval_ms=50,
        )
        
        all_dfs.append(df)
        
        if (day + 1) % 30 == 0:
            print(f"Generated {day + 1} days...")
    
    # Combine all DataFrames
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save
    full_df.to_parquet(output_path, compression="zstd", index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Generated {len(full_df):,} ticks ({file_size_mb:.2f} MB)")
    print(f"Saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate mock tick data")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "large"],
        default="multi",
        help="Generation mode",
    )
    parser.add_argument("--symbol", default="AAPL", help="Symbol for single mode")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"], help="Symbols for multi mode")
    parser.add_argument("--n-ticks", type=int, default=100000, help="Number of ticks")
    parser.add_argument("--n-days", type=int, default=365, help="Number of days for large mode")
    parser.add_argument("--output", default="data/mock_ticks", help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        df = generate_mock_ticks(symbol=args.symbol, n_ticks=args.n_ticks)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{args.symbol}_ticks.parquet"
        df.to_parquet(output_file, compression="zstd", index=False)
        print(f"Generated {len(df)} ticks, saved to {output_file}")
    
    elif args.mode == "multi":
        generate_multiple_symbols(
            symbols=args.symbols,
            n_ticks=args.n_ticks,
            output_dir=args.output,
        )
    
    elif args.mode == "large":
        generate_large_dataset(
            symbol=args.symbol,
            n_days=args.n_days,
            ticks_per_day=args.n_ticks,
            output_file=f"{args.output}/large_ticks.parquet",
        )


if __name__ == "__main__":
    main()
