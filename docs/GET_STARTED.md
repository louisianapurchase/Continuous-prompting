# 🚀 Get Started in 5 Minutes

The fastest way to start experimenting with continuous prompting!

## Step 1: Install Dependencies (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

## Step 2: Get Ollama Ready (2 minutes)

```bash
# Pull a model (if you haven't already)
ollama pull mistral

# Verify it's available
ollama list
```

## Step 3: Launch! (1 minute)

### 🌐 Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

**Your browser will open automatically!**

Then:
1. Click **▶️ Start** in the sidebar
2. Watch the magic happen! ✨

### 💻 Command Line (Alternative)

```bash
python main.py --max-iterations 20
```

## What You'll See

### Web Interface
- 📊 **Live Data Tab**: Real-time trading data streaming
- 💬 **LLM Responses Tab**: AI analysis as it happens
- 📈 **Charts Tab**: Interactive price visualizations
- 📋 **History Tab**: Full data table + CSV download

### Command Line
- Streaming data updates (one per second)
- LLM responses in formatted boxes
- Clean, color-coded output
- Final statistics summary

## Quick Experiments

### Experiment 1: Continuous Prompting (Default)
```bash
streamlit run streamlit_app.py
# Click Start - LLM sees every data point
```

### Experiment 2: Event-Driven
```bash
# In web interface:
# 1. Change Strategy to "event_driven"
# 2. Click Start
# LLM only responds to significant events
```

### Experiment 3: Different Models
```bash
# In web interface:
# 1. Change Model to "llama2" or "phi"
# 2. Click Start
# Compare response quality
```

### Experiment 4: Adjust Frequency
```bash
# In web interface:
# 1. Set Update Interval to 0.5 (faster)
# 2. Or set to 5.0 (slower, more thoughtful)
# 3. Click Start
```

## Troubleshooting

### "Connection refused"
```bash
# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull mistral
```

### "Module not found"
```bash
# Install dependencies
pip install -r requirements.txt
```

### Web interface won't load
```bash
# Test dependencies first
python test_streamlit.py

# If all good, try different port
streamlit run streamlit_app.py --server.port 8502
```

## Next Steps

1. ✅ **Got it working?** Great! Now read [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md)
2. 📖 **Want more examples?** Check [QUICKSTART.md](QUICKSTART.md)
3. 🏗️ **Understand the architecture?** See [README.md](README.md)
4. 🎨 **Curious about features?** Read [STREAMLIT_FEATURES.md](STREAMLIT_FEATURES.md)

## Tips for Success

### Start Simple
- Use default settings first
- Run for 20-30 data points
- Observe what happens
- Then experiment!

### Learn by Doing
- Try different strategies
- Change the prompts
- Test various models
- Compare results

### Export and Analyze
- Download CSV from History tab
- Open in Excel or Python
- Look for patterns
- Iterate on your approach

## Common Questions

**Q: How long should I run experiments?**
A: Start with 20-50 data points. For serious experiments, 100-500 points.

**Q: Which strategy is better?**
A: Depends on your goal! Continuous sees everything, event-driven is selective.

**Q: What's a good response rate?**
A: 100% for continuous (batch_size=1), 10-30% for event-driven.

**Q: Can I use real trading data?**
A: Yes! Prepare a CSV with columns: symbol, price, volume, timestamp.

**Q: How do I customize the prompts?**
A: Edit in the web interface sidebar or in config.yaml.

**Q: Can I run multiple experiments?**
A: Yes! Use different terminal windows or browser tabs (different ports).

## Resources

- 📖 [README.md](README.md) - Full documentation
- 🚀 [QUICKSTART.md](QUICKSTART.md) - Detailed quick start
- 🌐 [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md) - Web UI guide
- 🎨 [STREAMLIT_FEATURES.md](STREAMLIT_FEATURES.md) - Feature overview
- ⚙️ [config.yaml](config.yaml) - Configuration file

## Have Fun! 🎉

This framework is all about experimentation. There are no wrong answers - try things, break things, learn things!

**Happy prompting!** 🤖✨

