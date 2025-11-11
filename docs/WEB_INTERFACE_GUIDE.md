# 🌐 Web Interface Guide

The Streamlit web interface provides a beautiful, interactive way to run and monitor your continuous prompting experiments.

## 🚀 Quick Start

### Launch the Web Interface

**Windows:**
```bash
run_web.bat
```

**Linux/Mac:**
```bash
chmod +x run_web.sh
./run_web.sh
```

**Or manually:**
```bash
streamlit run streamlit_app.py
```

Your browser will automatically open to `http://localhost:8501`

## 📱 Interface Overview

### Sidebar - Configuration Panel

The left sidebar contains all configuration options:

#### ⚙️ LLM Settings
- **Model**: Choose your Ollama model (mistral, llama2, phi, etc.)
- **Temperature**: Control randomness (0.0 = deterministic, 1.0 = creative)
- **Max Tokens**: Maximum response length

#### 📊 Data Settings
- **Data Source**: 
  - `sample` - Generate realistic trading data
  - `csv` - Load from CSV file
- **Update Interval**: How often data updates (in seconds)
- **Symbols**: Which stocks to track (for sample data)
- **Price Volatility**: How much prices fluctuate

#### 🎯 Strategy Settings
- **Strategy Type**:
  - `continuous` - LLM sees every data point (or batches)
  - `event_driven` - LLM only responds to significant events
- **Batch Size**: How many data points to batch together (continuous mode)

#### 💬 Prompt Settings
- **System Prompt**: Customize the LLM's role and behavior

#### 🎮 Control Buttons
- **▶️ Start**: Begin the experiment
- **⏹️ Stop**: Pause the experiment
- **🔄 Reset**: Clear all data and start fresh

### Main Dashboard

#### Top Metrics Bar
Real-time statistics displayed in 4 cards:
1. **Data Points**: Total number of data points processed
2. **LLM Responses**: Number of responses generated
3. **Elapsed Time**: How long the experiment has been running
4. **Response Rate**: Percentage of data points that triggered responses

#### Tab 1: 📊 Live Data
- Shows the **current data point** being processed
- Displays:
  - Symbol (stock ticker)
  - Current price
  - Price change percentage
  - Volume
  - Timestamp
- Visual indicator: 📈 for price up, 📉 for price down

#### Tab 2: 💬 LLM Responses
- Shows all LLM responses in chronological order (newest first)
- Each response includes:
  - Response number and iteration
  - Timestamp
  - Associated data point (symbol, price, change)
  - Full LLM response text
- Clean, formatted boxes for easy reading
- Automatically scrolls to show latest responses

#### Tab 3: 📈 Charts
- **Interactive price charts** for all symbols
- Separate subplot for each stock
- Features:
  - Zoom in/out
  - Pan across time
  - Hover for exact values
  - Auto-updates as new data arrives

#### Tab 4: 📋 History
- **Full data table** with all processed data points
- Columns: symbol, price, change, volume, timestamp
- Sortable and searchable
- **📥 Download CSV** button to export data for analysis

## 🎨 Features

### Real-Time Updates
- Dashboard auto-refreshes every 0.5 seconds when running
- No need to manually refresh
- Smooth, non-flickering updates

### Clean Visual Design
- Color-coded price changes (green = up, red = down)
- Organized tabs to prevent information overload
- Professional styling with custom CSS
- Responsive layout that works on different screen sizes

### User-Friendly Controls
- All settings in one place (sidebar)
- One-click start/stop
- No command-line arguments needed
- Visual feedback for all actions

### Data Export
- Download complete data history as CSV
- Timestamped filenames
- Ready for external analysis (Excel, Python, R, etc.)

## 💡 Usage Tips

### For Quick Experiments
1. Keep default settings
2. Click **▶️ Start**
3. Watch the **Live Data** tab for streaming data
4. Switch to **LLM Responses** tab to see AI analysis
5. Click **⏹️ Stop** when done

### For Detailed Analysis
1. Configure your preferred settings in sidebar
2. Start the experiment
3. Monitor the **Charts** tab to see price trends
4. Review responses in **LLM Responses** tab
5. Download data from **History** tab for offline analysis

### For Comparing Strategies
1. Run experiment with `continuous` strategy
2. Note the response rate and quality
3. Click **🔄 Reset**
4. Switch to `event_driven` strategy
5. Run again and compare results

### For Testing Different Models
1. Start with a small model (e.g., `phi`)
2. Run for 20-30 data points
3. **🔄 Reset**
4. Change model to `mistral` or `llama2`
5. Compare response quality and speed

## 🔧 Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check that Ollama is on `http://localhost:11434`

### "Model not found" error
- Pull the model first: `ollama pull mistral`
- Check available models: `ollama list`

### Web page won't load
- Check if port 8501 is available
- Try: `streamlit run streamlit_app.py --server.port 8502`

### Data not updating
- Click **🔄 Reset** and try again
- Check browser console for errors (F12)
- Restart Streamlit

### Slow performance
- Reduce update interval (increase to 2-3 seconds)
- Use a smaller model
- Reduce batch size
- Close other browser tabs

## 🎯 Best Practices

### Starting Out
- Use `--max-iterations` in CLI or manually stop after 20-30 points
- Start with `continuous` strategy to see how it works
- Use default prompts first, customize later

### Experimenting
- Change ONE setting at a time
- Take notes on what works well
- Download data for each experiment
- Compare metrics (response rate, quality)

### Performance
- For long experiments, use `event_driven` strategy
- Increase update interval for slower, more thoughtful responses
- Monitor the metrics bar to track progress

## 📊 Understanding the Output

### Response Rate
- **100%**: LLM responds to every data point (continuous with batch_size=1)
- **10-30%**: Typical for event-driven with moderate triggers
- **<5%**: Very selective event-driven strategy

### Good Response Indicators
- Relevant to the data shown
- Mentions specific prices/changes
- Provides actionable insights
- Consistent quality across responses

### Signs to Adjust Settings
- Responses too generic → Adjust system prompt, reduce batch size
- Too many responses → Switch to event-driven or increase batch size
- Responses too slow → Use smaller model, increase update interval
- Not enough responses → Lower event thresholds, use continuous mode

## 🚀 Advanced Features

### Custom Data Sources
1. Prepare CSV file with columns: `symbol`, `price`, `volume`, `timestamp`
2. Place in `data/raw/` folder
3. Select `csv` as data source
4. Enter filename in CSV Path field

### Custom Prompts
- Edit the System Prompt in sidebar
- Examples:
  - "You are a risk-averse investor focused on capital preservation"
  - "You are a day trader looking for quick profit opportunities"
  - "You are a technical analyst. Use only price action in your analysis"

### Combining with CLI
- Run web interface for visual monitoring
- Use CLI with `--display minimal` for production runs
- Compare results from both interfaces

## 📚 Next Steps

After getting comfortable with the web interface:
1. Read the full [README.md](README.md) for architecture details
2. Check [QUICKSTART.md](QUICKSTART.md) for CLI usage
3. Explore the code in `src/` to understand how it works
4. Create custom strategies by extending `BaseStrategy`
5. Build custom data sources by implementing `DataSource` interface

---

**Enjoy experimenting! 🎉**

If you find issues or have suggestions, feel free to open an issue or contribute!

