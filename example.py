"""Complete example demonstrating the Market Data ETL & Backtesting Engine."""

import pandas as pd
from pathlib import Path

# Import modules
from tools.generate_mock_ticks import generate_mock_ticks
from data_processing import TickToOHLCV, DataCleaner, Resampler
from backtesting import MeanReversionStrategy, BacktestEngine, Visualizer


def main():
    """Run complete workflow example."""
    print("="*70)
    print("Market Data ETL & Strategy Backtesting Engine")
    print("Complete Workflow Example")
    print("="*70)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate mock tick data (simulate 100 days of trading)
    print("\n[1/6] Generating mock tick data...")
    from datetime import datetime, timedelta
    
    # Generate data for a longer period
    all_ticks = []
    base_date = datetime(2023, 1, 1)
    
    for day in range(100):  # 100 trading days
        day_ticks = generate_mock_ticks(
            symbol="AAPL",
            start_price=150.0,
            n_ticks=1000,
            start_time=base_date + timedelta(days=day),
            tick_interval_ms=60000,  # 1 minute intervals
            volatility=0.02,
        )
        all_ticks.append(day_ticks)
    
    df_ticks = pd.concat(all_ticks, ignore_index=True)
    print(f"Generated {len(df_ticks):,} ticks over 100 days")
    
    # Step 2: Clean the data
    print("\n[2/6] Cleaning data...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_pipeline_pandas(
        df_ticks,
        remove_outliers=True,
        deduplicate=True,
        validate_prices=True,
    )
    print(f"After cleaning: {len(df_clean):,} rows")
    
    # Step 3: Convert to OHLCV (1-minute bars)
    print("\n[3/6] Converting ticks to OHLCV (1-minute bars)...")
    converter = TickToOHLCV(use_polars=False)
    df_ohlcv_1min = converter.convert_pandas(df_clean, freq="1min")
    print(f"Created {len(df_ohlcv_1min):,} 1-minute bars")
    
    # Step 4: Resample to daily bars
    print("\n[4/6] Resampling to daily bars...")
    resampler = Resampler()
    df_daily = resampler.resample_pandas(df_ohlcv_1min, freq="1D")
    print(f"Created {len(df_daily):,} daily bars")
    
    # Step 5: Run backtest with mean reversion strategy
    print("\n[5/6] Running backtest with Mean Reversion strategy...")
    strategy = MeanReversionStrategy(
        data=df_daily,
        window=20,      # 20-day rolling window
        num_std=2.0,    # 2 standard deviations
    )
    
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=100000,
        commission=0.001,
        slippage=0.0005,
    )
    
    summary = engine.run()
    
    # Print results
    print("\n[6/6] Backtest Results:")
    print("-" * 70)
    print(f"Strategy:            {summary['strategy']}")
    print(f"Initial Cash:        ${summary['initial_cash']:,.2f}")
    print(f"Final Equity:        ${summary['final_equity']:,.2f}")
    print(f"Total Return:        {summary['total_return']:.2%}")
    print(f"CAGR:                {summary['cagr']:.2%}")
    print(f"Sharpe Ratio:        {summary['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:       {summary['sortino_ratio']:.2f}")
    print(f"Max Drawdown:        {summary['max_drawdown']:.2%}")
    print(f"Calmar Ratio:        {summary['calmar_ratio']:.2f}")
    print(f"Volatility:          {summary['volatility']:.2%}")
    print(f"Win Rate:            {summary['win_rate']:.2%}")
    print(f"Profit Factor:       {summary['profit_factor']:.2f}")
    print(f"Number of Trades:    {summary['num_trades']}")
    print("-" * 70)
    
    # Optional: Create visualizations
    try:
        print("\nGenerating visualizations...")
        viz = Visualizer(engine.results)
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        viz.plot_equity_curve(
            title="Mean Reversion Strategy - Equity Curve",
            save_path="results/equity_curve.png",
            show=False,
        )
        
        viz.plot_drawdown(
            title="Mean Reversion Strategy - Drawdown",
            save_path="results/drawdown.png",
            show=False,
        )
        
        print("Visualizations saved to 'results/' directory")
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
