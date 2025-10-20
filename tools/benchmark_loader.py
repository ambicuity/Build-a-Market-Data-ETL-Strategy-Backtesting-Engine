"""Benchmark data loading performance."""

import time
import pandas as pd
import polars as pl
from pathlib import Path
import argparse
from typing import Callable
import psutil
import os


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_function(func: Callable, *args, **kwargs) -> dict:
    """Benchmark a function.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark results
    """
    mem_before = get_memory_usage()
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    mem_after = get_memory_usage()
    
    return {
        "execution_time": end_time - start_time,
        "memory_used": mem_after - mem_before,
        "result": result,
    }


def benchmark_csv_pandas(file_path: str) -> pd.DataFrame:
    """Load CSV using Pandas."""
    return pd.read_csv(file_path)


def benchmark_csv_polars(file_path: str) -> pl.DataFrame:
    """Load CSV using Polars."""
    return pl.read_csv(file_path)


def benchmark_parquet_pandas(file_path: str) -> pd.DataFrame:
    """Load Parquet using Pandas."""
    return pd.read_parquet(file_path)


def benchmark_parquet_polars(file_path: str) -> pl.DataFrame:
    """Load Parquet using Polars."""
    return pl.read_parquet(file_path)


def benchmark_chunked_pandas(file_path: str, chunk_size: int = 50000) -> pd.DataFrame:
    """Load CSV in chunks using Pandas."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def run_benchmarks(file_path: str, file_format: str = "parquet") -> None:
    """Run all benchmarks.
    
    Args:
        file_path: Path to data file
        file_format: File format ('csv' or 'parquet')
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {file_path}")
    print(f"{'='*70}\n")
    
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB\n")
    
    benchmarks = []
    
    if file_format == "csv":
        # CSV benchmarks
        print("1. Pandas CSV loading...")
        result = benchmark_function(benchmark_csv_pandas, file_path)
        benchmarks.append(("Pandas CSV", result))
        print(f"   Time: {result['execution_time']:.2f}s, Memory: {result['memory_used']:.2f} MB")
        
        print("\n2. Polars CSV loading...")
        result = benchmark_function(benchmark_csv_polars, file_path)
        benchmarks.append(("Polars CSV", result))
        print(f"   Time: {result['execution_time']:.2f}s, Memory: {result['memory_used']:.2f} MB")
        
        print("\n3. Pandas Chunked CSV loading...")
        result = benchmark_function(benchmark_chunked_pandas, file_path)
        benchmarks.append(("Pandas Chunked CSV", result))
        print(f"   Time: {result['execution_time']:.2f}s, Memory: {result['memory_used']:.2f} MB")
    
    else:
        # Parquet benchmarks
        print("1. Pandas Parquet loading...")
        result = benchmark_function(benchmark_parquet_pandas, file_path)
        benchmarks.append(("Pandas Parquet", result))
        print(f"   Time: {result['execution_time']:.2f}s, Memory: {result['memory_used']:.2f} MB")
        
        print("\n2. Polars Parquet loading...")
        result = benchmark_function(benchmark_parquet_polars, file_path)
        benchmarks.append(("Polars Parquet", result))
        print(f"   Time: {result['execution_time']:.2f}s, Memory: {result['memory_used']:.2f} MB")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Method':<25} {'Time (s)':<15} {'Memory (MB)':<15} {'Throughput'}")
    print(f"{'-'*70}")
    
    for name, result in benchmarks:
        throughput = file_size_mb / result['execution_time']
        print(f"{name:<25} {result['execution_time']:<15.2f} {result['memory_used']:<15.2f} {throughput:.2f} MB/s")
    
    # Find fastest
    fastest = min(benchmarks, key=lambda x: x[1]['execution_time'])
    print(f"\nFastest method: {fastest[0]} ({fastest[1]['execution_time']:.2f}s)")
    
    # Find most memory efficient
    most_efficient = min(benchmarks, key=lambda x: x[1]['memory_used'])
    print(f"Most memory efficient: {most_efficient[0]} ({most_efficient[1]['memory_used']:.2f} MB)")


def benchmark_aggregation(df: pd.DataFrame) -> None:
    """Benchmark aggregation operations.
    
    Args:
        df: DataFrame to benchmark
    """
    print(f"\n{'='*70}")
    print("AGGREGATION BENCHMARKS")
    print(f"{'='*70}\n")
    
    print("1. Calculating mean price...")
    result = benchmark_function(lambda: df["price"].mean())
    print(f"   Time: {result['execution_time']:.4f}s, Result: {result['result']:.2f}")
    
    print("\n2. Group by symbol and calculate mean...")
    result = benchmark_function(lambda: df.groupby("symbol")["price"].mean())
    print(f"   Time: {result['execution_time']:.4f}s")
    
    print("\n3. Resample to 1-minute bars...")
    df_copy = df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
    df_copy.set_index("timestamp", inplace=True)
    
    result = benchmark_function(
        lambda: df_copy.resample("1min").agg({
            "price": ["first", "max", "min", "last"],
            "volume": "sum"
        })
    )
    print(f"   Time: {result['execution_time']:.4f}s, Bars created: {len(result['result'])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark data loading")
    parser.add_argument("file", help="Path to data file")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet", help="File format")
    parser.add_argument("--agg", action="store_true", help="Run aggregation benchmarks")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return
    
    run_benchmarks(args.file, args.format)
    
    if args.agg:
        print("\n" + "="*70)
        print("Loading data for aggregation benchmarks...")
        
        if args.format == "parquet":
            df = pd.read_parquet(args.file)
        else:
            df = pd.read_csv(args.file)
        
        benchmark_aggregation(df)


if __name__ == "__main__":
    main()
