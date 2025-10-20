"""Convert tick data to OHLCV bars."""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class TickToOHLCV:
    """Convert tick-level data to OHLCV bars."""
    
    def __init__(self, use_polars: bool = True):
        """Initialize converter.
        
        Args:
            use_polars: Use Polars for better performance (default: True)
        """
        self.use_polars = use_polars
    
    def convert_pandas(
        self,
        df: pd.DataFrame,
        freq: str = "1min",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """Convert tick data to OHLCV using Pandas.
        
        Args:
            df: DataFrame with columns: timestamp, price, volume
            freq: Resampling frequency (e.g., '1min', '5min', '1H')
            volume_col: Name of volume column
            
        Returns:
            OHLCV DataFrame
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Ensure timestamp is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        
        # Resample to OHLCV
        ohlcv = df.resample(freq).agg({
            "price": ["first", "max", "min", "last"],
            volume_col: "sum"
        })
        
        # Flatten column names
        ohlcv.columns = ["open", "high", "low", "close", "volume"]
        
        # Drop rows with no data
        ohlcv = ohlcv.dropna()
        
        # Reset index to make timestamp a column
        ohlcv.reset_index(inplace=True)
        
        return ohlcv
    
    def convert_polars(
        self,
        df: pl.DataFrame,
        freq: str = "1m",
        volume_col: str = "volume",
    ) -> pl.DataFrame:
        """Convert tick data to OHLCV using Polars.
        
        Args:
            df: Polars DataFrame with columns: timestamp, price, volume
            freq: Resampling frequency (e.g., '1m', '5m', '1h')
            volume_col: Name of volume column
            
        Returns:
            OHLCV DataFrame
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Ensure timestamp is datetime
        if df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
            )
        
        # Group by time period and aggregate
        ohlcv = (
            df.sort("timestamp")
            .group_by_dynamic("timestamp", every=freq)
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col(volume_col).sum().alias("volume"),
            ])
        )
        
        # Drop rows with null values
        ohlcv = ohlcv.drop_nulls()
        
        return ohlcv
    
    def convert(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        freq: str = "1min",
        volume_col: str = "volume",
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Convert tick data to OHLCV (auto-detect library).
        
        Args:
            data: DataFrame with tick data
            freq: Resampling frequency
            volume_col: Name of volume column
            
        Returns:
            OHLCV DataFrame
        """
        if isinstance(data, pl.DataFrame):
            # Convert frequency notation for Polars
            freq_map = {
                "1min": "1m", "5min": "5m", "15min": "15m",
                "30min": "30m", "1H": "1h", "1D": "1d"
            }
            polars_freq = freq_map.get(freq, freq)
            return self.convert_polars(data, polars_freq, volume_col)
        else:
            return self.convert_pandas(data, freq, volume_col)
    
    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        freq: str = "1min",
        chunk_size: Optional[int] = None,
    ) -> None:
        """Convert tick data file to OHLCV and save.
        
        Args:
            input_path: Path to input CSV/Parquet file
            output_path: Path to output Parquet file
            freq: Resampling frequency
            chunk_size: Process file in chunks (for large files)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        logger.info(f"Converting {input_path} to OHLCV...")
        
        if self.use_polars:
            # Use Polars for better performance
            df = pl.read_parquet(input_path) if input_path.suffix == ".parquet" else pl.read_csv(input_path)
            ohlcv = self.convert_polars(df, freq)
            ohlcv.write_parquet(output_path, compression="zstd")
        else:
            # Use Pandas with chunking for large files
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                    ohlcv_chunk = self.convert_pandas(chunk, freq)
                    chunks.append(ohlcv_chunk)
                
                ohlcv = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
                ohlcv = self.convert_pandas(df, freq)
            
            ohlcv.to_parquet(output_path, compression="zstd", index=False)
        
        logger.info(f"OHLCV data saved to {output_path}")
    
    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        freq: str = "1min",
        pattern: str = "*.csv",
    ) -> None:
        """Convert multiple tick data files to OHLCV.
        
        Args:
            input_dir: Input directory containing tick data files
            output_dir: Output directory for OHLCV files
            freq: Resampling frequency
            pattern: File pattern to match
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(files)} files to convert")
        
        for input_file in files:
            output_file = output_dir / f"{input_file.stem}_ohlcv.parquet"
            try:
                self.convert_file(input_file, output_file, freq)
            except Exception as e:
                logger.error(f"Error converting {input_file}: {e}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert tick data to OHLCV")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--freq", default="1min", help="Resampling frequency")
    parser.add_argument("--batch", action="store_true", help="Batch convert directory")
    
    args = parser.parse_args()
    
    converter = TickToOHLCV()
    
    if args.batch:
        converter.batch_convert(args.input, args.output, args.freq)
    else:
        converter.convert_file(args.input, args.output, args.freq)
