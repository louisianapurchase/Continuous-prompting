# Using Real Stock Data

This guide shows you how to replace the simulated data with real historical stock data.

## Quick Setup (3 Steps)

### Step 1: Install yfinance

```bash
pip install yfinance
```

### Step 2: Download Real Data

```bash
python scripts/download_real_data.py
```

**What this does:**
- Downloads 1-minute interval data for AAPL, GOOGL, MSFT, TSLA
- Gets the last 1 day of trading data (most recent market day)
- Saves to `data/raw/real_trading_data_1m_1d.csv`
- Shows you the time range and number of data points

**Example output:**
```
Downloading AAPL data (1m intervals, 1d period)...
  Downloaded 390 data points for AAPL
  Time range: 2024-12-02 09:30:00 to 2024-12-02 16:00:00
  Price range: $149.50 - $152.30

Total data points: 1560
Saved to: data/raw/real_trading_data_1m_1d.csv
```

### Step 3: Update config.yaml

Change the data source from `sample` to `csv`:

```yaml
data:
  update_interval: 0.5
  source: "csv"  # Changed from "sample"
  
  # CSV source settings
  csv_path: "data/raw/real_trading_data_1m_1d.csv"
```

### Step 4: Run the App

```bash
python run_flask.py
```

Now you'll see **real historical stock data** playing back at 0.5 second intervals!

## What Changed

### Before (Simulated Data)
- Random walk price generation
- Fake volume numbers
- Unrealistic price movements
- Infinite data stream

### After (Real Data)
- Actual historical prices from Yahoo Finance
- Real trading volumes
- Realistic market movements
- Finite data (replays when it reaches the end)

## Customization

### Download Different Time Periods

Edit `scripts/download_real_data.py`:

**5-minute intervals for 5 days:**
```python
INTERVAL = '5m'
PERIOD = '5d'
```

**15-minute intervals for 1 month:**
```python
INTERVAL = '15m'
PERIOD = '1mo'
```

### Download Different Stocks

Edit `scripts/download_real_data.py`:

```python
SYMBOLS = ['AAPL', 'TSLA', 'NVDA', 'AMD']
```

Then update `config.yaml`:

```yaml
data:
  sample:
    symbols: ['AAPL', 'TSLA', 'NVDA', 'AMD']
```

## Data Playback Speed

The `update_interval` in `config.yaml` controls how fast the data plays back:

```yaml
data:
  update_interval: 0.5  # 0.5 seconds between data points
```

**Examples:**
- `0.1` - Very fast (10 data points per second)
- `0.5` - Fast (2 data points per second) - **recommended**
- `1.0` - Normal (1 data point per second)
- `2.0` - Slow (1 data point every 2 seconds)

**Note:** If you downloaded 1-minute interval data, setting `update_interval: 0.5` means each real minute of trading will play back in 0.5 seconds (120x speed).

## Limitations

### Yahoo Finance Free Data Limits

- **1-minute data**: Only last 7 days available
- **5-minute data**: Several months available
- **15-minute+ data**: Years of data available
- **Market hours only**: No data outside 9:30 AM - 4:00 PM ET
- **No weekends**: No data on weekends or holidays

### What Happens When Data Runs Out

The CSV data source will reach the end of the file and stop. You can:

1. **Download more data** with a longer period
2. **Loop the data** (modify CSVDataSource to restart from beginning)
3. **Switch back to simulated data** in config.yaml

## Troubleshooting

### "No data downloaded"
- Market may be closed (weekends, holidays, after hours)
- Try downloading data for a previous weekday
- Try a longer period: `PERIOD = '5d'`

### "CSV file not found"
- Make sure you ran `python scripts/download_real_data.py` first
- Check that the file exists: `data/raw/real_trading_data_1m_1d.csv`
- Check the path in `config.yaml` matches the downloaded file

### "Data plays back too fast/slow"
- Adjust `update_interval` in `config.yaml`
- Smaller = faster, larger = slower

## Benefits of Real Data

✅ **Realistic testing** - See how your LLM responds to real market conditions
✅ **Reproducible** - Same data every time you run it
✅ **Historical events** - Includes real news events, earnings, etc.
✅ **Accurate patterns** - Real support/resistance levels, trends, volatility
✅ **Better training** - Helps tune your LLM prompts and triggers

Enjoy trading with real data! 📈

