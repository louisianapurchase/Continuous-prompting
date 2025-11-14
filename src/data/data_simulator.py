"""Trading data simulator for streaming market data."""

import time
from typing import Generator, Dict, Any, Optional
from datetime import datetime
import logging

from .data_sources import DataSource
from .news_generator import NewsGenerator

logger = logging.getLogger(__name__)


class TradingDataSimulator:
    """
    Simulates live trading data by streaming data points at regular intervals.
    
    This class takes a data source and streams it row-by-row at a specified
    interval to simulate real-time market data.
    """
    
    def __init__(
        self,
        data_source: DataSource,
        update_interval: float = 1.0,
        news_generator: Optional[NewsGenerator] = None,
    ):
        """
        Initialize the trading data simulator.

        Args:
            data_source: Source of trading data
            update_interval: Time in seconds between data updates
            news_generator: Optional news event generator
        """
        self.data_source = data_source
        self.update_interval = update_interval
        self.news_generator = news_generator
        self.is_running = False
        self._current_index = 0
        
    def start(self) -> None:
        """Start the data simulator."""
        self.is_running = True
        self._current_index = 0
        logger.info(
            f"Data simulator started. Update interval: {self.update_interval}s"
        )
    
    def stop(self) -> None:
        """Stop the data simulator."""
        self.is_running = False
        logger.info("Data simulator stopped.")
    
    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream trading data points at regular intervals.

        Yields:
            Dictionary containing trading data for a single time point
            OR news event dictionary if news is generated
        """
        self.start()

        try:
            while self.is_running:
                current_time = datetime.now()

                # Check if we should generate a news event
                if self.news_generator:
                    news_event = self.news_generator.generate_event(current_time)
                    if news_event:
                        # Yield news event as a data point
                        news_data = news_event.to_dict()
                        news_data['stream_timestamp'] = current_time.isoformat()
                        news_data['index'] = self._current_index

                        logger.info(f"News event injected: {news_event}")
                        yield news_data

                        self._current_index += 1
                        time.sleep(self.update_interval)
                        continue

                # Get next regular data point
                data_point = self.data_source.get_next()

                if data_point is None:
                    logger.info("Data source exhausted. Stopping stream.")
                    break

                # Add metadata
                data_point['stream_timestamp'] = current_time.isoformat()
                data_point['index'] = self._current_index

                yield data_point

                self._current_index += 1

                # Wait for next update
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            logger.info("Stream interrupted by user.")
            self.stop()
        except Exception as e:
            logger.error(f"Error in data stream: {e}")
            self.stop()
            raise
    
    def get_current_index(self) -> int:
        """Get the current data point index."""
        return self._current_index
    
    def reset(self) -> None:
        """Reset the simulator to the beginning."""
        self._current_index = 0
        self.data_source.reset()
        if self.news_generator:
            self.news_generator.reset()
        logger.info("Data simulator reset.")

