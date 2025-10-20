"""Configuration management for ETL pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Configuration loader and manager."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        
        if self.config_path.exists():
            self.load()
        else:
            self._config = self._default_config()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def save(self) -> None:
        """Save configuration to YAML file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "websocket": {
                "url": "wss://example.com/feed",
                "reconnect_delay": 5,
                "heartbeat_interval": 30,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "market_data",
                "user": "postgres",
                "password": "password",
            },
            "etl": {
                "batch_size": 1000,
                "queue_size": 10000,
                "symbols": ["AAPL", "MSFT", "GOOGL"],
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL."""
        return self.get("websocket.url", "wss://example.com/feed")
    
    @property
    def symbols(self) -> List[str]:
        """Get list of symbols to track."""
        return self.get("etl.symbols", ["AAPL", "MSFT", "GOOGL"])
    
    @property
    def batch_size(self) -> int:
        """Get batch size for database writes."""
        return self.get("etl.batch_size", 1000)
    
    @property
    def queue_size(self) -> int:
        """Get queue size for backpressure handling."""
        return self.get("etl.queue_size", 10000)
