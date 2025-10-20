"""Asynchronous database writer for market data."""

import asyncio
import asyncpg
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseWriter:
    """Asynchronous database writer with connection pooling."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "market_data",
        user: str = "postgres",
        password: str = "password",
        pool_size: int = 10,
        batch_size: int = 1000,
    ):
        """Initialize database writer.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            pool_size: Connection pool size
            batch_size: Batch size for bulk inserts
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.batch_size = batch_size
        
        self.pool: Optional[asyncpg.Pool] = None
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Create connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=self.pool_size,
            )
            logger.info(f"Connected to database: {self.database}")
            await self._create_tables()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.flush()
            await self.pool.close()
            logger.info("Database connection closed")
    
    async def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            price NUMERIC(20, 6) NOT NULL,
            volume NUMERIC(20, 6) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
        ON trades(symbol, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
        ON trades(timestamp DESC);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            logger.info("Tables created/verified")
    
    async def write(self, record: Dict[str, Any]) -> None:
        """Write a single record (buffered).
        
        Args:
            record: Trade record to write
        """
        if not record:
            return
        
        async with self._lock:
            self._buffer.append(record)
            
            if len(self._buffer) >= self.batch_size:
                await self._flush_buffer()
    
    async def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple records at once.
        
        Args:
            records: List of trade records
        """
        if not records:
            return
        
        async with self._lock:
            self._buffer.extend(records)
            
            if len(self._buffer) >= self.batch_size:
                await self._flush_buffer()
    
    async def flush(self) -> None:
        """Flush buffered records to database."""
        async with self._lock:
            await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Internal method to flush buffer (must be called with lock held)."""
        if not self._buffer or not self.pool:
            return
        
        try:
            records = self._buffer[:]
            self._buffer.clear()
            
            # Prepare values for bulk insert
            values = [
                (
                    record["timestamp"],
                    record["symbol"],
                    record["price"],
                    record["volume"],
                )
                for record in records
            ]
            
            insert_sql = """
            INSERT INTO trades (timestamp, symbol, price, volume)
            VALUES ($1, $2, $3, $4)
            """
            
            async with self.pool.acquire() as conn:
                await conn.executemany(insert_sql, values)
            
            logger.debug(f"Flushed {len(records)} records to database")
        
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            # Re-add records to buffer for retry
            self._buffer.extend(records)
    
    async def get_latest_trades(
        self, symbol: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get latest trades for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Number of trades to retrieve
            
        Returns:
            List of trade records
        """
        if not self.pool:
            return []
        
        query = """
        SELECT timestamp, symbol, price, volume
        FROM trades
        WHERE symbol = $1
        ORDER BY timestamp DESC
        LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, limit)
            return [dict(row) for row in rows]
