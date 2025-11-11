# Continuous Prompting Framework

Experimental framework for continuous LLM prompting with streaming trading data.

## Quick Start

```bash
pip install -r requirements.txt
ollama pull mistral
python run_web.py  # Web interface
# OR
python run.py --max-iterations 20  # CLI
```

## What is This?

Explores **continuous prompting** - feeding streaming data to LLMs in real-time.

**Key Question:** How do you effectively prompt an LLM with continuously streaming data?

## Features

- Web Interface (Streamlit) - Live charts, interactive controls
- CLI Interface - Clean terminal output
- Real-time Data Streaming
- Multiple Strategies (Continuous, Event-Driven, Custom)
- Interactive Charts
- CSV Export

## Structure

```
├── src/app/          # CLI (cli.py) & Web (web.py)
├── src/data/         # Data streaming
├── src/llm/          # Ollama integration
├── src/strategies/   # Prompting strategies
├── examples/         # Example scripts
├── docs/             # Documentation
├── run.py            # CLI launcher
└── run_web.py        # Web launcher
```

## Usage

**Web Interface:**
```bash
python run_web.py
```

**CLI:**
```bash
python run.py                      # Default
python run.py --max-iterations 50  # Limit iterations
python run.py --display compact    # Compact mode
```

## Configuration

Edit `config.yaml`:
```yaml
llm:
  model: "mistral"
  temperature: 0.7

data:
  update_interval: 1.0  # seconds
  source: "sample"

strategy:
  type: "continuous"  # or "event_driven"
```

## The Continuous Prompting Challenge

**Current Implementation Limitation:**

This framework doesn't truly solve continuous prompting - it's a starting point for experimentation.

**What it currently does:**
- Creates a NEW prompt for each data point (or batch)
- Manually injects context via prompt history
- Essentially "rapid one-shot prompting"

**Why this is problematic:**
- Token limits constrain how much history you can include
- LLM has no persistent memory between prompts
- No true state maintenance
- Inefficient for long-running streams

**Potential Solutions to Explore:**

1. **Continuous Fine-tuning**
   - Fine-tune model on incoming data in real-time
   - Model actually "learns" from the stream
   - Requires: Training infrastructure, checkpointing

2. **Vector Database + RAG**
   - Store all data points in vector DB
   - Retrieve relevant context for each prompt
   - Maintains "memory" without token limits
   - Tools: ChromaDB, Pinecone, Weaviate

3. **Stateful LLM Wrapper**
   - Maintain conversation state server-side
   - Use summarization to compress history
   - Implement sliding window of context
   - Periodically summarize old data

4. **Hybrid Approach**
   - Batch data into "epochs" (e.g., every 100 points)
   - Fine-tune on each epoch
   - Use fine-tuned model for next epoch
   - Combines benefits of both approaches

**This is the core experiment:** Finding effective methods for continuous LLM prompting!

## Current Strategies

### 1. Continuous Strategy (`src/strategies/continuous_strategy.py`)
- Prompts LLM with every data point (or batches)
- Includes limited history in prompt
- High token usage
- Never misses data

### 2. Event-Driven Strategy (`src/strategies/event_driven_strategy.py`)
- Only prompts on significant events
- Triggers: price changes, volume spikes, time intervals
- Lower token usage
- More selective

### 3. Custom Strategy
- Extend `BaseStrategy` class
- Implement your own logic
- See `src/strategies/base_strategy.py`

## Documentation

- **[docs/GET_STARTED.md](docs/GET_STARTED.md)** - Detailed setup guide
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Quick reference
- **[docs/WEB_INTERFACE_GUIDE.md](docs/WEB_INTERFACE_GUIDE.md)** - Web UI documentation
- **[docs/STRATEGY_EXPLANATION.md](docs/STRATEGY_EXPLANATION.md)** - Deep dive into continuous prompting strategies

## Experimentation Ideas

1. Implement vector DB storage for data history
2. Add summarization layer for context compression
3. Experiment with different batch sizes
4. Try fine-tuning approaches
5. Build adaptive trigger mechanisms
6. Compare token usage across strategies

## Development

```bash
# Run example
python examples/example_simple.py

# View strategy implementation
cat src/strategies/continuous_strategy.py
```

## License

MIT

---

**Note:** This framework provides infrastructure for experimentation. The continuous prompting strategy is intentionally basic - it's up to you to explore better approaches!

