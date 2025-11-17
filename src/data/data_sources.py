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

        # Initialize prices for each symbol with FIXED realistic starting prices
        # This prevents massive jumps when simulator restarts
        fixed_prices = {
            "AAPL": 150.00,
            "GOOGL": 2800.00,
            "MSFT": 380.00,
            "TSLA": 250.00,
        }
        self.current_prices = {
            symbol: fixed_prices.get(symbol, 150.00) for symbol in self.symbols
        }
        
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

            stock_data = {
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'price': round(new_price, 2),
                'volume': volume,
                'change': round(price_change * 100, 2),  # percentage
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
        self.current_prices = {
            symbol: random.uniform(100, 500) for symbol in self.symbols
        }
        self.start_time = datetime.now()
        self.current_index = 0
        logger.info("Sample data source reset.")


class CSVDataSource(DataSource):
    """
    Loads trading data from a CSV file.
    
    Expected CSV format:
    timestamp,symbol,price,volume,high,low
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize CSV data source.
        
        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = csv_path
        self.data = []
        self.current_index = 0
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            self.data = df.to_dict('records')
            logger.info(
                f"Loaded {len(self.data)} data points from {self.csv_path}"
            )
        except FileNotFoundError:
            logger.warning(
                f"CSV file not found: {self.csv_path}. Using empty dataset."
            )
            self.data = []
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            self.data = []
    
    def get_next(self) -> Optional[Dict[str, Any]]:
        """Get next data point from CSV."""
        if self.current_index >= len(self.data):
            return None
        
        data_point = self.data[self.current_index]
        self.current_index += 1
        return data_point
    
    def reset(self) -> None:
        """Reset to beginning of CSV."""
        self.current_index = 0
        logger.info("CSV data source reset.")


class RandomDataSource(DataSource):
    """
    Generates completely random trading data.
    
    Useful for stress testing and experimentation.
    """
    
    def __init__(self, symbols: List[str] = None, max_points: Optional[int] = None):
        """
        Initialize random data source.
        
        Args:
            symbols: List of stock symbols
            max_points: Maximum number of data points (None for infinite)
        """
        self.symbols = symbols or ["STOCK_A", "STOCK_B", "STOCK_C"]
        self.max_points = max_points
        self.current_index = 0
    
    def get_next(self) -> Optional[Dict[str, Any]]:
        """Generate random data point."""
        if self.max_points and self.current_index >= self.max_points:
            return None
        
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'symbol': random.choice(self.symbols),
            'price': round(random.uniform(10, 1000), 2),
            'volume': random.randint(100, 100000),
            'change': round(random.uniform(-10, 10), 2),
            'high': round(random.uniform(10, 1000), 2),
            'low': round(random.uniform(10, 1000), 2),
        }
        
        self.current_index += 1
        return data_point
    
    def reset(self) -> None:
        """Reset counter."""
        self.current_index = 0

