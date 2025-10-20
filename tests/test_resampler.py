"""Tests for data resampler."""

import pytest
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from data_processing.resampler import Resampler


class TestResampler:
    """Test data resampler."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample OHLCV data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
        
        self.df_pandas = pd.DataFrame({
            "timestamp": dates,
            "open": 100 + pd.Series(range(100)) * 0.1,
            "high": 101 + pd.Series(range(100)) * 0.1,
            "low": 99 + pd.Series(range(100)) * 0.1,
            "close": 100.5 + pd.Series(range(100)) * 0.1,
            "volume": [100] * 100,
        })
        
        self.df_polars = pl.from_pandas(self.df_pandas)
        
        self.resampler = Resampler()
    
    def test_resample_pandas(self):
        """Test resampling with Pandas."""
        result = self.resampler.resample_pandas(self.df_pandas, freq="5min")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(self.df_pandas)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
    
    def test_resample_polars(self):
        """Test resampling with Polars."""
        result = self.resampler.resample_polars(self.df_polars, freq="5m")
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) < len(self.df_polars)
    
    def test_resample_auto_detect(self):
        """Test auto-detection of library."""
        # Test with Pandas DataFrame
        result_pandas = self.resampler.resample(self.df_pandas, freq="5min")
        assert isinstance(result_pandas, pd.DataFrame)
        
        # Test with Polars DataFrame
        result_polars = self.resampler.resample(self.df_polars, freq="5min")
        assert isinstance(result_polars, pl.DataFrame)
    
    def test_downsample_ohlcv(self):
        """Test downsampling OHLCV data."""
        result = self.resampler.downsample_ohlcv(
            self.df_pandas,
            from_freq="1min",
            to_freq="5min"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(self.df_pandas)
        
        # Check that high is the maximum of highs
        # Check that low is the minimum of lows
        # Check that volume is sum of volumes
    
    def test_resample_empty_dataframe(self):
        """Test resampling empty DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        empty_df["timestamp"] = pd.to_datetime(empty_df["timestamp"])
        
        result = self.resampler.resample_pandas(empty_df, freq="5min")
        
        assert len(result) == 0
    
    def test_resample_missing_columns(self):
        """Test resampling with missing columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="1min"),
            "price": [100] * 10,
        })
        
        # Should still work with available columns
        result = self.resampler.resample_pandas(df, freq="5min", agg_dict={"price": "mean"})
        
        assert len(result) < len(df)


def test_import_resampler():
    """Test that resampler can be imported."""
    from data_processing import Resampler
    
    assert Resampler is not None
    
    resampler = Resampler()
    assert resampler is not None
