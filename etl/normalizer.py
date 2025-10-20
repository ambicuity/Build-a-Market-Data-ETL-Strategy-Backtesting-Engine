"""Data normalization module for raw market data."""

from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_trade(raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize raw trade data into standard format.
    
    Args:
        raw_data: Raw trade data from WebSocket feed
        
    Returns:
        Normalized trade record with keys: timestamp, symbol, price, volume
        Returns None if data is invalid
    """
    try:
        # Handle different feed formats
        if "data" in raw_data:
            data = raw_data["data"]
        else:
            data = raw_data
        
        # Extract timestamp
        timestamp = data.get("timestamp") or data.get("time") or data.get("t")
        if isinstance(timestamp, (int, float)):
            # Unix timestamp (milliseconds or seconds)
            if timestamp > 1e12:  # Milliseconds
                timestamp = timestamp / 1000.0
            timestamp = datetime.fromtimestamp(timestamp).isoformat()
        elif isinstance(timestamp, str):
            # ISO format string
            timestamp = timestamp
        else:
            timestamp = datetime.utcnow().isoformat()
        
        # Extract symbol
        symbol = data.get("symbol") or data.get("s") or data.get("ticker")
        if not symbol:
            logger.warning("Missing symbol in trade data")
            return None
        
        # Extract price
        price = data.get("price") or data.get("p") or data.get("last")
        if price is None:
            logger.warning(f"Missing price for {symbol}")
            return None
        price = float(price)
        
        # Extract volume
        volume = data.get("volume") or data.get("v") or data.get("size") or 0
        volume = float(volume)
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": price,
            "volume": volume,
        }
    
    except Exception as e:
        logger.error(f"Error normalizing trade data: {e}, data: {raw_data}")
        return None


def normalize_quote(raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize raw quote data into standard format.
    
    Args:
        raw_data: Raw quote data from WebSocket feed
        
    Returns:
        Normalized quote record with bid/ask prices and sizes
    """
    try:
        if "data" in raw_data:
            data = raw_data["data"]
        else:
            data = raw_data
        
        timestamp = data.get("timestamp", datetime.now().isoformat())
        symbol = data.get("symbol") or data.get("s")
        
        if not symbol:
            return None
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "bid_price": float(data.get("bid") or data.get("bp") or 0),
            "ask_price": float(data.get("ask") or data.get("ap") or 0),
            "bid_size": float(data.get("bid_size") or data.get("bs") or 0),
            "ask_size": float(data.get("ask_size") or data.get("as") or 0),
        }
    
    except Exception as e:
        logger.error(f"Error normalizing quote data: {e}")
        return None
