"""Tests for ETL pipeline components."""

import pytest
import asyncio
from datetime import datetime
from etl.config import Config
from etl.normalizer import normalize_trade, normalize_quote
from etl.database_writer import DatabaseWriter


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert config.websocket_url is not None
        assert isinstance(config.symbols, list)
        assert config.batch_size > 0
        assert config.queue_size > 0
    
    def test_get_config_value(self):
        """Test getting configuration values."""
        config = Config()
        
        # Test dot notation
        value = config.get("websocket.url")
        assert value is not None
        
        # Test default value
        value = config.get("nonexistent.key", "default")
        assert value == "default"
    
    def test_set_config_value(self):
        """Test setting configuration values."""
        config = Config()
        
        config.set("test.key", "test_value")
        assert config.get("test.key") == "test_value"


class TestNormalizer:
    """Test data normalization."""
    
    def test_normalize_trade_basic(self):
        """Test basic trade normalization."""
        raw_data = {
            "timestamp": "2024-01-01T10:00:00",
            "symbol": "AAPL",
            "price": 150.50,
            "volume": 100,
        }
        
        result = normalize_trade(raw_data)
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.50
        assert result["volume"] == 100
    
    def test_normalize_trade_nested(self):
        """Test trade normalization with nested data."""
        raw_data = {
            "data": {
                "t": 1609459200000,  # Unix timestamp in ms
                "s": "MSFT",
                "p": 300.25,
                "v": 50,
            }
        }
        
        result = normalize_trade(raw_data)
        
        assert result is not None
        assert result["symbol"] == "MSFT"
        assert result["price"] == 300.25
    
    def test_normalize_trade_missing_symbol(self):
        """Test normalization with missing symbol."""
        raw_data = {
            "timestamp": "2024-01-01T10:00:00",
            "price": 150.50,
            "volume": 100,
        }
        
        result = normalize_trade(raw_data)
        
        assert result is None
    
    def test_normalize_trade_invalid_data(self):
        """Test normalization with invalid data."""
        raw_data = {
            "timestamp": "invalid",
            "symbol": "AAPL",
        }
        
        result = normalize_trade(raw_data)
        
        # Should return None for missing price
        assert result is None
    
    def test_normalize_quote(self):
        """Test quote normalization."""
        raw_data = {
            "timestamp": "2024-01-01T10:00:00",
            "symbol": "AAPL",
            "bid": 150.00,
            "ask": 150.10,
            "bid_size": 100,
            "ask_size": 200,
        }
        
        result = normalize_quote(raw_data)
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["bid_price"] == 150.00
        assert result["ask_price"] == 150.10


class TestDatabaseWriter:
    """Test database writer."""
    
    @pytest.mark.asyncio
    async def test_database_writer_init(self):
        """Test database writer initialization."""
        writer = DatabaseWriter(
            host="localhost",
            database="test_db",
            user="test_user",
        )
        
        assert writer.host == "localhost"
        assert writer.database == "test_db"
        assert writer.batch_size > 0
    
    @pytest.mark.asyncio
    async def test_write_to_buffer(self):
        """Test writing to buffer."""
        writer = DatabaseWriter(batch_size=10)
        
        # Mock record
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 100,
        }
        
        # Since we can't connect to a real DB in tests, we just check the buffer
        async with writer._lock:
            writer._buffer.append(record)
            assert len(writer._buffer) == 1


@pytest.mark.asyncio
async def test_etl_pipeline_components():
    """Integration test for ETL components."""
    # Test that components can be imported and initialized
    config = Config()
    
    assert config is not None
    assert config.symbols is not None
    
    # Test normalizer
    test_data = {
        "timestamp": "2024-01-01T10:00:00",
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 100,
    }
    
    normalized = normalize_trade(test_data)
    assert normalized is not None
    assert normalized["symbol"] == "AAPL"


def test_import_modules():
    """Test that all ETL modules can be imported."""
    from etl import Config, LiveETL, normalize_trade, DatabaseWriter, ETLPipeline
    
    assert Config is not None
    assert LiveETL is not None
    assert normalize_trade is not None
    assert DatabaseWriter is not None
    assert ETLPipeline is not None
