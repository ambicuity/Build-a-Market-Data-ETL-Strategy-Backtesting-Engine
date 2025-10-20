"""ETL Pipeline orchestration."""

import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional
from .config import Config
from .websocket_client import LiveETL
from .database_writer import DatabaseWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ETLPipeline:
    """Orchestrates the ETL pipeline components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ETL pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.db_writer: Optional[DatabaseWriter] = None
        self.etl_client: Optional[LiveETL] = None
        self._shutdown_event = asyncio.Event()
    
    async def setup(self) -> None:
        """Setup pipeline components."""
        logger.info("Setting up ETL pipeline...")
        
        # Initialize database writer
        db_config = {
            "host": self.config.get("database.host", "localhost"),
            "port": self.config.get("database.port", 5432),
            "database": self.config.get("database.database", "market_data"),
            "user": self.config.get("database.user", "postgres"),
            "password": self.config.get("database.password", "password"),
            "batch_size": self.config.batch_size,
        }
        
        self.db_writer = DatabaseWriter(**db_config)
        await self.db_writer.connect()
        
        # Initialize ETL client
        self.etl_client = LiveETL(
            ws_url=self.config.websocket_url,
            db_writer=self.db_writer,
            symbols=self.config.symbols,
            queue_size=self.config.queue_size,
            reconnect_delay=self.config.get("websocket.reconnect_delay", 5),
            heartbeat_interval=self.config.get("websocket.heartbeat_interval", 30),
        )
        
        logger.info("ETL pipeline setup complete")
    
    async def run(self) -> None:
        """Run the ETL pipeline."""
        try:
            await self.setup()
            
            # Setup signal handlers
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            
            logger.info("Starting ETL pipeline...")
            
            # Run the ETL client
            await self.etl_client.start()
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise
        
        finally:
            await self.cleanup()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the pipeline."""
        logger.info("Shutting down ETL pipeline...")
        self._shutdown_event.set()
        
        if self.etl_client:
            await self.etl_client.stop()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        if self.db_writer:
            await self.db_writer.disconnect()
        
        logger.info("Cleanup complete")


async def main():
    """Main entry point for ETL pipeline."""
    # Look for config in standard location
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    
    pipeline = ETLPipeline(str(config_path) if config_path.exists() else None)
    
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
