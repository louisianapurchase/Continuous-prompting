"""Data sources for trading data."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import random
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Get the next data point.
        
        Returns:
            Dictionary containing trading data, or None if exhausted
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the data source to the beginning."""
        pass


class SampleDataSource(DataSource):
    """
    Generates sample trading data with realistic price movements.
    
    This is useful for testing without needing real market data.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        price_volatility: float = 0.0001,
        volume_range: tuple = (1000, 10000),
        max_points: Optional[int] = None,
    ):
        """
        Initialize sample data source.

        Args:
            symbols: List of stock symbols to generate data for
            price_volatility: Maximum price change percentage per update (default: 0.0001 = 0.01%)
            volume_range: Tuple of (min, max) volume
            max_points: Maximum number of data points to generate (None for infinite)
        """
        self.symbols = symbols or ["AAPL", "GOOGL", "MSFT", "TSLA"]
        self.price_volatility = price_volatility
        self.volume_range = volume_range
        self.max_points = max_points

        # Initialize prices for each symbol with CURRENT realistic starting prices
        # Updated December 2025 - GOOGL had 20:1 split in July 2022
        fixed_prices = {
            "AAPL": 175.00,   # Current price ~$175
            "GOOGL": 140.00,  # Current price ~$140 (after 20:1 split)
            "MSFT": 380.00,   # Current price ~$380
            "TSLA": 240.00,   # Current price ~$240
        }
        self.current_prices = {
            symbol: fixed_prices.get(symbol, 150.00) for symbol in self.symbols
        }

        # Track starting prices to calculate total change from start
        self.starting_prices = self.current_prices.copy()

        self.start_time = datetime.now()
        self.current_index = 0

        logger.info(
            f"Sample data source initialized with symbols: {self.symbols}"
        )
    
    def get_next(self) -> Optional[Dict[str, Any]]:
        """Generate next sample data point - returns data for ALL symbols at once."""
        if self.max_points and self.current_index >= self.max_points:
            return None

        # Calculate timestamp
        timestamp = self.start_time + timedelta(seconds=self.current_index)

        # Generate data for ALL symbols
        stocks_data = []
        for symbol in self.symbols:
            # Update price with random walk
            current_price = self.current_prices[symbol]
            price_change = random.uniform(
                -self.price_volatility, self.price_volatility
            )
            new_price = current_price * (1 + price_change)
            self.current_prices[symbol] = new_price

            # Generate volume
            volume = random.randint(*self.volume_range)

            # Calculate total change from starting price (not just recent change)
            starting_price = self.starting_prices[symbol]
            total_change_pct = ((new_price - starting_price) / starting_price) * 100

            stock_data = {
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'price': round(new_price, 2),
                'volume': volume,
                'change': round(total_change_pct, 2),  # total % change from start
                'high': round(new_price * 1.001, 2),
                'low': round(new_price * 0.999, 2),
            }
            stocks_data.append(stock_data)

        self.current_index += 1

        # Return a batch containing all stocks
        return {
            'type': 'batch',
            'timestamp': timestamp.isoformat(),
            'stocks': stocks_data
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        fixed_prices = {
            "AAPL": 175.00,   # Current price ~$175
            "GOOGL": 140.00,  # Current price ~$140 (after 20:1 split)
            "MSFT": 380.00,   # Current price ~$380
            "TSLA": 240.00,   # Current price ~$240
        }
        self.current_prices = {
            symbol: fixed_prices.get(symbol, 150.00) for symbol in self.symbols
        }
        self.starting_prices = self.current_prices.copy()
        self.start_time = datetime.now()
        self.current_index = 0
        logger.info("Sample data source reset.")


class CSVDataSource(DataSource):
    """
    Loads trading data from a CSV file and returns batches of all symbols.

    Expected CSV format:
    timestamp,Symbol,price,volume,change,high,low,open

    The CSV should contain data for multiple symbols with the same timestamps.
    This source groups data by timestamp and returns all symbols together.
    """

    def __init__(self, csv_path: str, symbols: List[str] = None):
        """
        Initialize CSV data source.

        Args:
            csv_path: Path to CSV file
            symbols: List of symbols to filter (None = use all symbols in CSV)
        """
        self.csv_path = csv_path
        self.symbols = symbols
        self.data_by_timestamp = {}
        self.timestamps = []
        self.current_index = 0
        self.last_file_mtime = None
        self._load_data()

    def _load_data(self) -> None:
        """Load data from CSV file and group by timestamp."""
        try:
            import pandas as pd
            import os
            from zoneinfo import ZoneInfo

            # Track file modification time
            if os.path.exists(self.csv_path):
                self.last_file_mtime = os.path.getmtime(self.csv_path)

            df = pd.read_csv(self.csv_path)

            # Filter by symbols if specified
            if self.symbols:
                df = df[df['Symbol'].isin(self.symbols)]

            # Convert timestamp column to datetime for filtering
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter out future timestamps (Yahoo Finance sometimes includes incomplete bars)
            try:
                et_tz = ZoneInfo('America/New_York')
                now_et = datetime.now(et_tz)
            except Exception:
                now_et = datetime.now()

            # Make now_et timezone-naive if df timestamps are timezone-naive
            if df['timestamp'].dt.tz is None and now_et.tzinfo is not None:
                now_et = now_et.replace(tzinfo=None)
            # Make df timestamps timezone-naive if now_et is timezone-naive
            elif df['timestamp'].dt.tz is not None and now_et.tzinfo is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            # Filter out future data
            original_count = len(df)
            df = df[df['timestamp'] <= now_et]
            filtered_count = original_count - len(df)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} future data points")

            # Group data by timestamp
            self.data_by_timestamp = {}  # Clear existing data
            for timestamp, group in df.groupby('timestamp'):
                stocks_data = []
                for _, row in group.iterrows():
                    stock_data = {
                        'timestamp': str(row['timestamp']),
                        'symbol': row['Symbol'],
                        'price': float(row['price']),
                        'volume': int(row['volume']),
                        'change': float(row.get('change', 0)),
                        'high': float(row.get('high', row['price'])),
                        'low': float(row.get('low', row['price'])),
                    }
                    stocks_data.append(stock_data)

                self.data_by_timestamp[timestamp] = stocks_data

            self.timestamps = sorted(self.data_by_timestamp.keys())

            logger.info(
                f"Loaded {len(self.timestamps)} timestamps with {len(df)} total data points from {self.csv_path}"
            )
            if self.timestamps:
                logger.info(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")

        except FileNotFoundError:
            logger.warning(
                f"CSV file not found: {self.csv_path}. Using empty dataset."
            )
            self.data_by_timestamp = {}
            self.timestamps = []

    def reload_if_modified(self) -> bool:
        """
        Check if CSV file has been modified and reload if needed.

        Returns:
            True if data was reloaded, False otherwise
        """
        try:
            import os
            if not os.path.exists(self.csv_path):
                return False

            current_mtime = os.path.getmtime(self.csv_path)
            if self.last_file_mtime is None or current_mtime > self.last_file_mtime:
                logger.info(f"CSV file modified. Reloading data from {self.csv_path}")
                old_index = self.current_index
                self._load_data()
                # Keep current position if possible
                self.current_index = min(old_index, len(self.timestamps) - 1)
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking CSV modification: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading CSV: {e}", exc_info=True)
            self.data_by_timestamp = {}
            self.timestamps = []

    def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Get next batch of data (all symbols at current timestamp).

        If we've reached the end, check if CSV has been updated with new data.
        """
        # Check if we've reached the end
        if self.current_index >= len(self.timestamps):
            # Try to reload if CSV has been modified
            if self.reload_if_modified():
                # If we got new data and we're still at the end, we're caught up
                if self.current_index >= len(self.timestamps):
                    return None
            else:
                return None

        timestamp = self.timestamps[self.current_index]
        stocks_data = self.data_by_timestamp[timestamp]
        self.current_index += 1

        # Return batch format (same as SampleDataSource)
        return {
            'type': 'batch',
            'timestamp': timestamp,
            'stocks': stocks_data
        }

    def reset(self) -> None:
        """Reset to beginning of CSV."""
        self.current_index = 0
        logger.info("CSV data source reset.")
