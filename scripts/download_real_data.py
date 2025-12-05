"""
Download real intraday stock data using yfinance.

This script downloads 1-minute or 5-minute interval data for the last 7 days
(maximum available from Yahoo Finance for free) and saves it to CSV.

Usage:
    pip install yfinance pandas
    python scripts/download_real_data.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Stocks to download
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

# Interval options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d'
# Note: 1m data is only available for last 7 days
INTERVAL = '1m'  # 1-minute intervals (RECOMMENDED - realistic trading data)

# Period options: '1d', '5d', '7d', '1mo', '3mo', '1y', '2y', '5y', 'max'
# For 1m interval, max is 7d
PERIOD = '1d'  # Last 1 day (gives ~390 data points per stock for a full trading day)

def download_stock_data(symbol, interval='1m', period='1d'):
    """
    Download intraday data for a single stock.
    
    Args:
        symbol: Stock ticker symbol
        interval: Data interval (1m, 5m, etc.)
        period: Time period (1d, 5d, 7d, etc.)
    
    Returns:
        DataFrame with stock data
    """
    print(f"Downloading {symbol} data ({interval} intervals, {period} period)...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        print(f"  WARNING: No data returned for {symbol}")
        return None
    
    # Add symbol column
    df['Symbol'] = symbol
    
    # Reset index to make timestamp a column
    df.reset_index(inplace=True)
    
    # Rename columns to match our format
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
    
    # Select only the columns we need
    df = df[['timestamp', 'Symbol', 'price', 'volume', 'change', 'high', 'low', 'open']]
    
    print(f"  Downloaded {len(df)} data points for {symbol}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    return df

def main(append_mode=False):
    """
    Download data for all symbols and save to CSV.

    Args:
        append_mode: If True, append new data to existing CSV instead of replacing
    """
    print("=" * 60)
    print("Real Stock Data Downloader")
    print("=" * 60)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Interval: {INTERVAL}")
    print(f"Period: {PERIOD}")
    print(f"Mode: {'APPEND' if append_mode else 'REPLACE'}")
    print("=" * 60)
    print()

    all_data = []

    for symbol in SYMBOLS:
        df = download_stock_data(symbol, interval=INTERVAL, period=PERIOD)
        if df is not None:
            all_data.append(df)
        print()

    if not all_data:
        print("ERROR: No data downloaded for any symbol!")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp
    combined_df.sort_values('timestamp', inplace=True)

    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)

    # Save to CSV
    output_file = f'data/raw/real_trading_data_{INTERVAL}_{PERIOD}.csv'

    if append_mode and os.path.exists(output_file):
        # Load existing data
        print(f"Loading existing data from {output_file}...")
        existing_df = pd.read_csv(output_file)
        print(f"  Existing data points: {len(existing_df)}")

        # Combine with new data
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)

        # Remove duplicates (same timestamp + symbol)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.drop_duplicates(subset=['timestamp', 'Symbol'], keep='last')

        # Sort by timestamp
        combined_df.sort_values('timestamp', inplace=True)

        print(f"  After merging: {len(combined_df)} total data points")

    combined_df.to_csv(output_file, index=False)

    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Total data points: {len(combined_df)}")
    print(f"Saved to: {output_file}")
    if len(combined_df) > 0:
        print(f"Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print()
    print("To use this data, update config.yaml:")
    print("  data:")
    print("    source: 'csv'")
    print(f"    csv_path: '{output_file}'")
    print("=" * 60)

if __name__ == '__main__':
    main()

