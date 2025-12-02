# Download Real Stock Data

This script downloads real intraday stock data from Yahoo Finance using the `yfinance` library.

## Quick Start

### 1. Install Required Package

```bash
pip install yfinance
```

### 2. Download Data

```bash
python scripts/download_real_data.py
```

This will download **1-minute interval data for the last 1 day** for AAPL, GOOGL, MSFT, and TSLA.

### 3. Use the Data

Update `config.yaml`:

```yaml
data:
  source: "csv"
  csv_path: "data/raw/real_trading_data_1m_1d.csv"
```

Then run the app:

```bash
python run_flask.py
```

## Customization

Edit `scripts/download_real_data.py` to change:

### Interval Options
- `'1m'` - 1 minute (only available for last 7 days)
- `'5m'` - 5 minutes
- `'15m'` - 15 minutes
- `'30m'` - 30 minutes
- `'1h'` - 1 hour

### Period Options
- `'1d'` - Last 1 day
- `'5d'` - Last 5 days
- `'7d'` - Last 7 days (max for 1m interval)
- `'1mo'` - Last 1 month
- `'3mo'` - Last 3 months

### Example: Download 5-minute data for 5 days

```python
INTERVAL = '5m'
PERIOD = '5d'
```

## Data Format

The downloaded CSV will have these columns:
- `timestamp` - Date and time of the data point
- `Symbol` - Stock ticker (AAPL, GOOGL, MSFT, TSLA)
- `price` - Closing price at that interval
- `volume` - Trading volume
- `change` - Percentage change from first price
- `high` - Highest price in that interval
- `low` - Lowest price in that interval
- `open` - Opening price in that interval

## Notes

- **Free data**: Yahoo Finance provides free data, no API key required
- **1-minute data**: Only available for the last 7 days
- **Market hours**: Data is only available during market hours (9:30 AM - 4:00 PM ET)
- **Weekends**: No data on weekends or market holidays
- **Rate limits**: Yahoo Finance may rate limit if you make too many requests

## Troubleshooting

### "No data returned"
- Check if the market was open during the requested period
- Try a longer period (e.g., '5d' instead of '1d')
- Try a larger interval (e.g., '5m' instead of '1m')

### "Module not found: yfinance"
```bash
pip install yfinance
```

### "Empty CSV file"
- The market may have been closed during the requested period
- Try downloading data for a weekday during market hours

