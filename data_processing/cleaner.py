"""Data cleaning utilities for market data."""

import pandas as pd
import polars as pl
import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and validate market data."""
    
    def __init__(self):
        """Initialize data cleaner."""
        pass
    
    def remove_outliers_pandas(
        self,
        df: pd.DataFrame,
        column: str = "price",
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Remove outliers using Pandas.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' for IQR method, 'zscore' for z-score method
            threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            mask = z_scores < threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed = len(df) - mask.sum()
        if removed > 0:
            logger.info(f"Removed {removed} outliers from {column}")
        
        return df[mask]
    
    def remove_outliers_polars(
        self,
        df: pl.DataFrame,
        column: str = "price",
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pl.DataFrame:
        """Remove outliers using Polars.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' for IQR method, 'zscore' for z-score method
            threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_clean = df.filter(
                (pl.col(column) >= lower_bound) & (pl.col(column) <= upper_bound)
            )
        
        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            
            df_clean = df.filter(
                ((pl.col(column) - mean).abs() / std) < threshold
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed = len(df) - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} outliers from {column}")
        
        return df_clean
    
    def deduplicate_pandas(
        self,
        df: pd.DataFrame,
        subset: Optional[list] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate rows using Pandas.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', or False for remove all)
            
        Returns:
            Deduplicated DataFrame
        """
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def deduplicate_polars(
        self,
        df: pl.DataFrame,
        subset: Optional[list] = None,
        keep: str = "first",
    ) -> pl.DataFrame:
        """Remove duplicate rows using Polars.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', or 'none')
            
        Returns:
            Deduplicated DataFrame
        """
        before = len(df)
        df = df.unique(subset=subset, keep=keep)
        removed = before - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def fill_missing_timestamps_pandas(
        self,
        df: pd.DataFrame,
        freq: str = "1min",
        method: str = "ffill",
    ) -> pd.DataFrame:
        """Fill missing timestamps using Pandas.
        
        Args:
            df: Input DataFrame with timestamp index or column
            freq: Frequency for regular time series
            method: Fill method ('ffill', 'bfill', 'interpolate')
            
        Returns:
            DataFrame with filled timestamps
        """
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        
        # Create complete timestamp range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Reindex to fill missing timestamps
        df = df.reindex(full_range)
        
        # Fill missing values
        if method == "ffill":
            df = df.fillna(method="ffill")
        elif method == "bfill":
            df = df.fillna(method="bfill")
        elif method == "interpolate":
            df = df.interpolate(method="time")
        
        df.index.name = "timestamp"
        return df.reset_index()
    
    def validate_prices_pandas(
        self,
        df: pd.DataFrame,
        price_col: str = "price",
        min_price: float = 0.01,
        max_price: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate price data using Pandas.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            min_price: Minimum valid price
            max_price: Maximum valid price (optional)
            
        Returns:
            Tuple of (valid_df, invalid_df)
        """
        mask = df[price_col] >= min_price
        
        if max_price is not None:
            mask &= df[price_col] <= max_price
        
        valid_df = df[mask]
        invalid_df = df[~mask]
        
        if len(invalid_df) > 0:
            logger.warning(f"Found {len(invalid_df)} invalid price records")
        
        return valid_df, invalid_df
    
    def clean_pipeline_pandas(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        deduplicate: bool = True,
        validate_prices: bool = True,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """Run complete cleaning pipeline on Pandas DataFrame.
        
        Args:
            df: Input DataFrame
            remove_outliers: Whether to remove outliers
            deduplicate: Whether to remove duplicates
            validate_prices: Whether to validate prices
            fill_missing: Whether to fill missing timestamps
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting cleaning pipeline. Initial rows: {len(df)}")
        
        if deduplicate:
            df = self.deduplicate_pandas(df)
        
        if validate_prices and "price" in df.columns:
            df, _ = self.validate_prices_pandas(df)
        
        if remove_outliers and "price" in df.columns:
            df = self.remove_outliers_pandas(df, column="price")
        
        if fill_missing and "timestamp" in df.columns:
            df = self.fill_missing_timestamps_pandas(df)
        
        logger.info(f"Cleaning complete. Final rows: {len(df)}")
        
        return df
    
    def clean(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Clean data (auto-detect library).
        
        Args:
            df: Input DataFrame
            **kwargs: Additional arguments for cleaning pipeline
            
        Returns:
            Cleaned DataFrame
        """
        if isinstance(df, pl.DataFrame):
            # Convert to Pandas for now (Polars pipeline can be added later)
            df_pd = df.to_pandas()
            cleaned = self.clean_pipeline_pandas(df_pd, **kwargs)
            return pl.from_pandas(cleaned)
        else:
            return self.clean_pipeline_pandas(df, **kwargs)
