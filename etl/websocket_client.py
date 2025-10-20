"""WebSocket client for real-time market data ingestion."""

import asyncio
import json
import logging
from typing import List, Optional, Callable
import aiohttp
from .normalizer import normalize_trade
from .database_writer import DatabaseWriter

logger = logging.getLogger(__name__)


class LiveETL:
    """Real-time ETL client for WebSocket market data feeds."""
    
    def __init__(
        self,
        ws_url: str,
        db_writer: DatabaseWriter,
        symbols: List[str],
        queue_size: int = 10000,
        reconnect_delay: int = 5,
        heartbeat_interval: int = 30,
        message_handler: Optional[Callable] = None,
    ):
        """Initialize WebSocket ETL client.
        
        Args:
            ws_url: WebSocket URL
            db_writer: Database writer instance
            symbols: List of symbols to subscribe to
            queue_size: Maximum queue size for backpressure
            reconnect_delay: Delay between reconnection attempts (seconds)
            heartbeat_interval: Heartbeat interval (seconds)
            message_handler: Custom message handler function
        """
        self.ws_url = ws_url
        self.db_writer = db_writer
        self.symbols = symbols
        self.queue_size = queue_size
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        self.message_handler = message_handler or normalize_trade
        
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.running = False
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "errors": 0,
            "reconnects": 0,
        }
    
    async def start(self) -> None:
        """Start the ETL pipeline."""
        self.running = True
        
        # Start worker tasks
        workers = [
            asyncio.create_task(self._stream()),
            asyncio.create_task(self._process_queue()),
            asyncio.create_task(self._heartbeat()),
            asyncio.create_task(self._stats_reporter()),
        ]
        
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            logger.info("ETL pipeline cancelled")
        finally:
            self.running = False
    
    async def stop(self) -> None:
        """Stop the ETL pipeline."""
        self.running = False
        await self.db_writer.flush()
        logger.info("ETL pipeline stopped")
    
    async def _stream(self) -> None:
        """Stream data from WebSocket with reconnection logic."""
        while self.running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._stats["reconnects"] += 1
                if self.running:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
    
    async def _connect_and_stream(self) -> None:
        """Establish WebSocket connection and stream data."""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url) as ws:
                logger.info(f"Connected to WebSocket: {self.ws_url}")
                
                # Subscribe to symbols
                subscribe_msg = {
                    "type": "subscribe",
                    "symbols": self.symbols,
                }
                await ws.send_json(subscribe_msg)
                logger.info(f"Subscribed to symbols: {self.symbols}")
                
                # Process messages
                async for msg in ws:
                    if not self.running:
                        break
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                            self._stats["errors"] += 1
                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket connection closed")
                        break
    
    async def _handle_message(self, data: dict) -> None:
        """Handle incoming message.
        
        Args:
            data: Raw message data
        """
        try:
            self._stats["messages_received"] += 1
            
            # Normalize the message
            record = self.message_handler(data)
            
            if record:
                # Add to queue with backpressure handling
                if self.queue.full():
                    logger.warning("Queue full, dropping message")
                    self._stats["errors"] += 1
                else:
                    await self.queue.put(record)
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._stats["errors"] += 1
    
    async def _process_queue(self) -> None:
        """Process messages from queue and write to database."""
        batch = []
        
        while self.running:
            try:
                # Get message with timeout to allow graceful shutdown
                try:
                    record = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    batch.append(record)
                except asyncio.TimeoutError:
                    pass
                
                # Write batch if it's large enough or queue is empty
                if batch and (len(batch) >= 100 or self.queue.empty()):
                    await self.db_writer.write_batch(batch)
                    self._stats["messages_processed"] += len(batch)
                    batch = []
            
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                self._stats["errors"] += 1
        
        # Flush remaining batch
        if batch:
            await self.db_writer.write_batch(batch)
            self._stats["messages_processed"] += len(batch)
    
    async def _heartbeat(self) -> None:
        """Send periodic heartbeat to keep connection alive."""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)
            logger.debug("Heartbeat")
    
    async def _stats_reporter(self) -> None:
        """Report statistics periodically."""
        while self.running:
            await asyncio.sleep(60)
            logger.info(
                f"Stats: received={self._stats['messages_received']}, "
                f"processed={self._stats['messages_processed']}, "
                f"errors={self._stats['errors']}, "
                f"reconnects={self._stats['reconnects']}, "
                f"queue_size={self.queue.qsize()}"
            )
    
    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return {
            **self._stats,
            "queue_size": self.queue.qsize(),
            "running": self.running,
        }
