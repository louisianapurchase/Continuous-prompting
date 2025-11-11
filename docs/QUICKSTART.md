# Quick Start Guide

Get up and running with the Continuous Prompting Framework in minutes!

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running ([Download here](https://ollama.ai/))

## Installation

### Option 1: Automated Setup (Linux/Mac)

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup (All Platforms)

1. **Create a virtual environment:**
```bash
python -m venv venv
```

2. **Activate the virtual environment:**

On Windows:
```bash
venv\Scripts\activate
```

On Linux/Mac:
```bash
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Pull an Ollama model:**
```bash
ollama pull mistral
```

Other good options: `llama2:7b`, `phi`, `tinyllama`

## Running Your First Experiment

### Option A: Web Interface (Easiest!) 🌐

**Launch the Streamlit web app:**

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically with a beautiful dashboard!

**What you can do:**
- ⚙️ Configure everything in the sidebar (model, data source, strategy)
- ▶️ Click "Start" to begin the experiment
- 📊 Watch live data streaming in the "Live Data" tab
- 💬 See LLM responses in the "LLM Responses" tab
- 📈 View interactive price charts in the "Charts" tab
- 📋 Download data history as CSV
- ⏹️ Stop/Reset anytime

**Perfect for:**
- First-time users
- Visual learners
- Experimenting with different settings
- Analyzing results in real-time

---

### Option B: Command Line Interface

**1. Start with the default configuration**

```bash
python main.py
```

This will:
- Generate sample trading data (AAPL, GOOGL, MSFT, TSLA)
- Stream data at 1-second intervals
- Use continuous prompting strategy
- Send data to the Mistral model

**2. Limit the experiment duration**

```bash
python main.py --max-iterations 20
```

This will process only 20 data points and then stop.

**3. Choose display mode**

```bash
# Full mode - shows data stream + responses
python main.py --display full

# Compact mode - only LLM responses
python main.py --display compact

# Minimal mode - quiet, logs only
python main.py --display minimal
```

**4. Try event-driven strategy**

Edit `config.yaml` and change:
```yaml
strategy:
  type: "event_driven"  # Changed from "continuous"
```

Then run:
```bash
python main.py --max-iterations 50
```

The LLM will only respond when significant events occur (price changes, volume spikes, etc.)

## Configuration

All settings are in `config.yaml`. Key sections:

### LLM Settings
```yaml
llm:
  model: "mistral"      # Change to your preferred model
  temperature: 0.7      # Creativity (0.0 = deterministic, 1.0 = creative)
```

### Data Settings
```yaml
data:
  update_interval: 1.0  # Seconds between updates
  source: "sample"      # Options: sample, csv
```

### Strategy Settings
```yaml
strategy:
  type: "continuous"    # Options: continuous, event_driven
```

## Experiment Ideas

### 1. Test Different Models

```bash
# Pull different models
ollama pull llama2:7b
ollama pull phi
ollama pull tinyllama

# Edit config.yaml to change model
# Then run experiments
```

### 2. Adjust Update Frequency

In `config.yaml`:
```yaml
data:
  update_interval: 0.5  # Faster updates (2 per second)
  # or
  update_interval: 5.0  # Slower updates (1 every 5 seconds)
```

### 3. Modify Event Triggers

In `config.yaml`:
```yaml
strategy:
  type: "event_driven"
  event_driven:
    triggers:
      - type: "price_change"
        threshold: 0.03  # 3% price change
      - type: "volume_spike"
        threshold: 3.0   # 3x average volume
```

### 4. Customize Prompts

In `config.yaml`:
```yaml
prompts:
  system_prompt: |
    You are an expert day trader. Focus on short-term opportunities
    and risk management. Be extremely concise.
```

## Output and Logs

- **Console**: Real-time LLM responses
- **Logs**: `logs/continuous_prompting_*.log`
- **Metrics**: `logs/metrics/*.json`
- **Conversations**: Saved if enabled in config

## Troubleshooting

### "ollama-python package not installed"
```bash
pip install ollama-python
```

### "Model 'mistral' not found"
```bash
ollama pull mistral
```

### "Connection refused" to Ollama
Make sure Ollama is running:
```bash
ollama serve
```

### Import errors
Make sure you're in the virtual environment:
```bash
# Check if (venv) appears in your prompt
# If not, activate it:
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## Next Steps

1. **Read the full README.md** for architecture details
2. **Experiment with different strategies** in `config.yaml`
3. **Create custom data sources** by extending `DataSource` class
4. **Build your own strategy** by extending `BaseStrategy` class
5. **Analyze metrics** in `logs/metrics/` to compare experiments

## Tips for Experimentation

- Start with `--max-iterations 10` to test quickly
- Use `event_driven` strategy to reduce LLM calls
- Monitor token usage and response times
- Try different temperature settings (0.3 for focused, 0.9 for creative)
- Experiment with batch sizes in continuous strategy

Happy experimenting! 🚀

