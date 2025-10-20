"""Data resampling utilities for time-series data."""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, Optional, List
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


class Resampler:
    """Resample time-series market data to different frequencies."""
    
    def __init__(self, n_workers: Optional[int] = None):
        """Initialize resampler.
        
        Args:
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.n_workers = n_workers or multiprocessing.cpu_count()
    
    def resample_pandas(
        self,
        df: pd.DataFrame,
        freq: str,
        agg_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Resample DataFrame using Pandas.
        
        Args:
            df: Input DataFrame with timestamp index or column
            freq: Target frequency (e.g., '5min', '1H', '1D')
            agg_dict: Custom aggregation dictionary
            
        Returns:
            Resampled DataFrame
        """
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # Default aggregation for OHLCV data
        if agg_dict is None:
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            # Only use columns that exist
            agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        resampled = df.resample(freq).agg(agg_dict)
        resampled = resampled.dropna()
        
        return resampled.reset_index()
    
    def resample_polars(
        self,
        df: pl.DataFrame,
        freq: str,
        agg_exprs: Optional[List] = None,
    ) -> pl.DataFrame:
        """Resample DataFrame using Polars.
        
        Args:
            df: Input Polars DataFrame
            freq: Target frequency (e.g., '5m', '1h', '1d')
            agg_exprs: Custom aggregation expressions
            
        Returns:
            Resampled DataFrame
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Ensure timestamp is datetime
        if df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
            )
        
        # Default aggregation for OHLCV data
        if agg_exprs is None:
            agg_exprs = []
            if "open" in df.columns:
                agg_exprs.append(pl.col("open").first())
            if "high" in df.columns:
                agg_exprs.append(pl.col("high").max())
            if "low" in df.columns:
                agg_exprs.append(pl.col("low").min())
            if "close" in df.columns:
                agg_exprs.append(pl.col("close").last())
            if "volume" in df.columns:
                agg_exprs.append(pl.col("volume").sum())
        
        resampled = (
            df.sort("timestamp")
            .group_by_dynamic("timestamp", every=freq)
            .agg(agg_exprs)
        )
        
        return resampled.drop_nulls()
    
    def resample(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        freq: str,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Resample data (auto-detect library).
        
        Args:
            df: Input DataFrame
            freq: Target frequency
            **kwargs: Additional arguments
            
        Returns:
            Resampled DataFrame
        """
        if isinstance(df, pl.DataFrame):
            # Convert frequency notation
            freq_map = {
                "1min": "1m", "5min": "5m", "15min": "15m",
                "30min": "30m", "1H": "1h", "1D": "1d"
            }
            polars_freq = freq_map.get(freq, freq)
            return self.resample_polars(df, polars_freq, **kwargs)
        else:
            return self.resample_pandas(df, freq, **kwargs)
    
    def resample_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        freq: str,
    ) -> None:
        """Resample data from file and save.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            freq: Target frequency
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        logger.info(f"Resampling {input_path} to {freq}...")
        
        # Read file
        if input_path.suffix == ".parquet":
            df = pl.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            df = pl.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Resample
        resampled = self.resample(df, freq)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(resampled, pl.DataFrame):
            resampled.write_parquet(output_path, compression="zstd")
        else:
            resampled.to_parquet(output_path, compression="zstd", index=False)
        
        logger.info(f"Resampled data saved to {output_path}")
    
    def _resample_file_worker(self, args):
        """Worker function for parallel resampling."""
        input_path, output_path, freq = args
        try:
            self.resample_file(input_path, output_path, freq)
            return True
        except Exception as e:
            logger.error(f"Error resampling {input_path}: {e}")
            return False
    
    def batch_resample(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        freq: str,
        pattern: str = "*.parquet",
        parallel: bool = True,
    ) -> None:
        """Resample multiple files in parallel.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            freq: Target frequency
            pattern: File pattern to match
            parallel: Use parallel processing
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(files)} files to resample")
        
        if not files:
            logger.warning("No files found to resample")
            return
        
        # Prepare work items
        work_items = [
            (
                input_file,
                output_dir / f"{input_file.stem}_resampled_{freq}.parquet",
                freq
            )
            for input_file in files
        ]
        
        if parallel and len(files) > 1:
            # Parallel processing
            logger.info(f"Using {self.n_workers} workers for parallel processing")
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(self._resample_file_worker, work_items))
            
            successful = sum(results)
            logger.info(f"Successfully resampled {successful}/{len(files)} files")
        else:
            # Sequential processing
            for work_item in work_items:
                self._resample_file_worker(work_item)
    
    def downsample_ohlcv(
        self,
        df: pd.DataFrame,
        from_freq: str,
        to_freq: str,
    ) -> pd.DataFrame:
        """Downsample OHLCV data from higher to lower frequency.
        
        Args:
            df: OHLCV DataFrame
            from_freq: Source frequency
            to_freq: Target frequency (must be lower)
            
        Returns:
            Downsampled OHLCV DataFrame
        """
        # Validate columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return self.resample_pandas(df, to_freq)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Resample time-series data")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--freq", required=True, help="Target frequency")
    parser.add_argument("--batch", action="store_true", help="Batch resample directory")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    resampler = Resampler(n_workers=args.workers)
    
    if args.batch:
        resampler.batch_resample(args.input, args.output, args.freq)
    else:
        resampler.resample_file(args.input, args.output, args.freq)
