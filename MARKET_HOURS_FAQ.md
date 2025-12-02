# Market Hours FAQ

## Will this only work during market hours?

**Short answer: No! The app works 24/7.**

### How It Works

**With Simulated Data (default):**
- Works **anytime, anywhere**
- Generates realistic random walk price data
- No dependency on market hours
- No dependency on internet connection (except for Ollama)
- Perfect for testing and development

**With Real Historical Data (CSV):**
- Works **anytime, anywhere**
- Plays back previously downloaded historical data
- No dependency on market hours
- No internet connection needed (after data is downloaded)
- Data is from actual trading days, but you can replay it anytime

### When Market Hours Matter

Market hours **only matter** when you're **downloading new data**:

**Yahoo Finance Data Availability:**
- **Market hours**: 9:30 AM - 4:00 PM ET (Monday-Friday)
- **Weekends**: No trading data available
- **Holidays**: No trading data on market holidays
- **After hours**: Limited data availability

**What this means:**
- If you run `python scripts/download_real_data.py` on a **Saturday**, it will download Friday's data
- If you run it on a **Monday at 2 PM ET**, it will download Monday's data up to 2 PM
- If you run it on a **Monday at 8 AM ET**, it will download Friday's data (Monday hasn't started yet)

### Best Practices

**For Development/Testing:**
```yaml
# config.yaml
data:
  source: "sample"  # Use simulated data - works 24/7
```

**For Realistic Testing:**
1. Download data once during or after market hours:
   ```bash
   python scripts/download_real_data.py
   ```

2. Use the downloaded data anytime:
   ```yaml
   # config.yaml
   data:
     source: "csv"
     csv_path: "data/raw/real_trading_data_1m_1d.csv"
   ```

3. The app will replay the data at your configured speed (default: 0.5 seconds per data point)

### Data Playback Speed

The `update_interval` controls how fast data plays back, **not** real-time market speed:

```yaml
data:
  update_interval: 0.5  # 0.5 seconds between data points
```

**Example:**
- Downloaded data: 390 data points (1 trading day at 1-minute intervals)
- Playback speed: 0.5 seconds per data point
- Total playback time: 390 × 0.5 = 195 seconds = **3.25 minutes**

So a full trading day (6.5 hours) plays back in just 3.25 minutes!

### Downloading Data for Different Days

**Get the most recent trading day:**
```python
# scripts/download_real_data.py
PERIOD = '1d'  # Last 1 day
```

**Get a full week of data:**
```python
PERIOD = '5d'  # Last 5 trading days
```

**Get a volatile day (e.g., earnings day):**
1. Find the date you want (e.g., "2024-01-15")
2. Modify the download script to use specific dates:
   ```python
   df = ticker.history(start='2024-01-15', end='2024-01-16', interval='1m')
   ```

### Summary

✅ **App runs 24/7** - No dependency on market hours for running the app
✅ **Simulated data** - Always available, works offline
✅ **Historical data** - Download once, replay anytime
❌ **Live data** - Not supported (this is a simulation/backtesting framework)

**Market hours only matter when downloading new data from Yahoo Finance.**

### Want Live Data?

If you need actual live market data, you would need to:
1. Use a real-time data provider (e.g., Alpaca, Interactive Brokers, Polygon.io)
2. Modify the `CSVDataSource` to connect to their API
3. Handle market hours, rate limits, and API keys
4. Pay for the data service (most are paid)

For this experimental framework, **historical data is recommended** because:
- Free and unlimited
- Reproducible (same data every time)
- Faster than real-time (can speed up playback)
- No API keys or rate limits
- Perfect for testing and development

