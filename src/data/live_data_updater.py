"""
Live data updater for real-time stock data during market hours.

This module automatically downloads and updates stock data from Yahoo Finance.
During market hours, it updates the CSV incrementally. After hours, it downloads once.
"""

import os
import logging
from datetime import datetime, time
from pathlib import Path
import threading
import time as time_module
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Real-time data updates disabled.")


def is_market_hours() -> bool:
    """
    Check if current time is during market hours (9:30 AM - 4:00 PM ET, Mon-Fri).

    Returns:
        True if market is open, False otherwise
    """
    # Get current time in Eastern Time (US stock market timezone)
    try:
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz)
    except Exception:
        # Fallback to local time if timezone not available
        logger.warning("Could not get Eastern Time. Using local time.")
        now_et = datetime.now()

    # Check if weekend
    if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    # Note: This is a simplified check (doesn't account for holidays)
    market_open = time(9, 30)
    market_close = time(16, 0)

    current_time = now_et.time()
    is_open = market_open <= current_time <= market_close

    logger.debug(f"Market hours check: ET time={current_time}, is_open={is_open}")
    return is_open


def download_stock_data(symbols, interval='1m', period='1d', csv_path='data/raw/real_trading_data_1m_1d.csv', append_mode=False):
    """
    Download stock data from Yahoo Finance and save to CSV.

    Args:
        symbols: List of stock symbols
        interval: Data interval (1m, 5m, etc.)
        period: Time period (1d, 5d, etc.)
        csv_path: Path to save CSV file
        append_mode: If True, append new data to existing CSV instead of replacing

    Returns:
        True if successful, False otherwise
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not installed. Cannot download data.")
        return False

    try:
        logger.info(f"Downloading {interval} data for {symbols} (append={append_mode})...")

        all_data = []

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                continue
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Reset index to make timestamp a column
            df.reset_index(inplace=True)
            
            # Rename columns
            df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'price',
                'Volume': 'volume'
            }, inplace=True)
            
            # Calculate percentage change from first price
            first_price = df['price'].iloc[0]
            df['change'] = ((df['price'] - first_price) / first_price) * 100
            
            # Select columns
            df = df[['timestamp', 'Symbol', 'price', 'volume', 'change', 'high', 'low', 'open']]
            
            all_data.append(df)
        
        if not all_data:
            logger.error("No data downloaded for any symbol")
            return False

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by timestamp
        combined_df.sort_values('timestamp', inplace=True)

        # Create directory if needed
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Handle append mode
        if append_mode and os.path.exists(csv_path):
            logger.info(f"Appending to existing CSV: {csv_path}")
            # Load existing data
            existing_df = pd.read_csv(csv_path)

            # Combine with new data
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)

            # Remove duplicates (same timestamp + symbol)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'Symbol'], keep='last')

            # Sort by timestamp
            combined_df.sort_values('timestamp', inplace=True)

            logger.info(f"After merging: {len(combined_df)} total data points")

        # Save to CSV
        combined_df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(combined_df)} data points to {csv_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}", exc_info=True)
        return False


def get_csv_last_timestamp(csv_path):
    """
    Get the last timestamp from the CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        datetime object of last timestamp, or None if error
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        # Get the last timestamp
        last_timestamp_str = df['timestamp'].iloc[-1]
        last_timestamp = pd.to_datetime(last_timestamp_str)
        return last_timestamp
    except Exception as e:
        logger.error(f"Error reading CSV timestamp: {e}")
        return None


def is_csv_up_to_date(csv_path, tolerance_minutes=2):
    """
    Check if CSV data is up to date (within tolerance of current time).

    Args:
        csv_path: Path to CSV file
        tolerance_minutes: How many minutes behind is acceptable

    Returns:
        True if CSV is up to date, False otherwise
    """
    last_timestamp = get_csv_last_timestamp(csv_path)
    if not last_timestamp:
        return False

    # Get current ET time
    try:
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz)
    except Exception:
        now_et = datetime.now()

    # Make last_timestamp timezone-aware if it isn't
    if last_timestamp.tzinfo is None:
        try:
            et_tz = ZoneInfo("America/New_York")
            last_timestamp = last_timestamp.replace(tzinfo=et_tz)
        except Exception:
            pass

    # Calculate time difference
    time_diff = now_et - last_timestamp
    minutes_behind = time_diff.total_seconds() / 60

    logger.info(f"CSV last timestamp: {last_timestamp}, Current ET: {now_et}, Minutes behind: {minutes_behind:.1f}")

    return minutes_behind <= tolerance_minutes


def ensure_data_available(symbols, csv_path='data/raw/real_trading_data_1m_1d.csv'):
    """
    Ensure CSV data is available. Download if missing or outdated.

    For multi-day data:
    - If CSV doesn't exist, download today's data
    - If CSV exists but doesn't have today's data, append today's data
    - If CSV has today's data, use it as-is

    Args:
        symbols: List of stock symbols
        csv_path: Path to CSV file

    Returns:
        True if data is available, False otherwise
    """
    csv_file = Path(csv_path)

    # If file doesn't exist, download it
    if not csv_file.exists():
        logger.info(f"CSV file not found. Downloading data...")
        return download_stock_data(symbols, csv_path=csv_path, append_mode=False)

    # Check if CSV has today's data
    try:
        last_timestamp = get_csv_last_timestamp(csv_path)
        if last_timestamp is None:
            logger.warning("Could not read CSV timestamp. Re-downloading...")
            return download_stock_data(symbols, csv_path=csv_path, append_mode=False)

        # Convert to Eastern Time for comparison
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')

        # Make last_timestamp timezone-aware if it isn't
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=et_tz)

        # Get current time in ET
        now_et = datetime.now(et_tz)

        # Check if last data is from today
        if last_timestamp.date() == now_et.date():
            logger.info(f"CSV has today's data (last: {last_timestamp}). Using existing file.")
            return True
        else:
            logger.info(f"CSV last data is from {last_timestamp.date()}. Appending today's data...")
            return download_stock_data(symbols, csv_path=csv_path, append_mode=True)

    except Exception as e:
        logger.error(f"Error checking CSV data: {e}", exc_info=True)
        logger.info("Re-downloading data...")
        return download_stock_data(symbols, csv_path=csv_path, append_mode=False)


class LiveDataUpdater:
    """
    Background thread that updates CSV data during market hours.

    During market hours (9:30 AM - 4:00 PM ET):
    - Updates CSV every 60 seconds with new data
    - Appends new data points without replacing the file

    After market hours:
    - Does nothing (data is already complete)
    """

    def __init__(self, symbols, csv_path='data/raw/real_trading_data_1m_1d.csv', update_interval=60):
        """
        Initialize live data updater.

        Args:
            symbols: List of stock symbols
            csv_path: Path to CSV file
            update_interval: Seconds between updates (default: 60)
        """
        self.symbols = symbols
        self.csv_path = csv_path
        self.update_interval = update_interval
        self.running = False
        self.thread = None

    def start(self):
        """Start the background update thread."""
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available. Live updates disabled.")
            return

        if self.running:
            logger.warning("Live updater already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("Live data updater started")

    def stop(self):
        """Stop the background update thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Live data updater stopped")

    def _update_loop(self):
        """Background loop that updates data during market hours."""
        while self.running:
            try:
                # Only update during market hours AND if CSV is behind
                if is_market_hours():
                    # Check if CSV needs updating (more than 2 minutes behind)
                    if not is_csv_up_to_date(self.csv_path, tolerance_minutes=2):
                        logger.info("Market is open and CSV is behind. Updating data...")
                        download_stock_data(self.symbols, csv_path=self.csv_path, append_mode=True)
                    else:
                        logger.debug("CSV is up to date. No update needed.")
                else:
                    logger.debug("Market is closed. Skipping update.")

                # Wait before next update
                time_module.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in live update loop: {e}", exc_info=True)
                time_module.sleep(self.update_interval)

