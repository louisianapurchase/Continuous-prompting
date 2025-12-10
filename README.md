# Continuous Prompting Framework

An experimental framework for exploring continuous LLM prompting with streaming data. This project investigates methods for effectively prompting Large Language Models with continuously streaming trading data, addressing the core challenge of maintaining context and memory across thousands of data points.

## Recent Updates (December 2025)

**MAJOR UPDATE - Autonomous LLM Strategy (December 6):**
- **Autonomous LLM activation** - LLM decides when to activate itself (no hardcoded triggers!)
- **Chain-of-Thought reasoning** - LLM explains its thinking step-by-step before making decisions
- **Self-reflection** - LLM critiques its own decisions to catch mistakes
- **Confidence scoring** - LLM only acts when confident (configurable threshold)
- **ReAct pattern** - Modern AI technique combining reasoning and acting
- **Adaptive behavior** - LLM learns what market patterns are significant
- **Removed reactive strategy** - Eliminated all hardcoded trigger logic
- **Cleaner codebase** - Single autonomous strategy, no legacy code

**Why This Matters:**
Previous versions used hardcoded rules (e.g., "activate if price changes > 1.5%"). The new autonomous strategy lets the LLM decide what's important. This is a fundamental shift from rule-based to truly intelligent behavior.

**Multi-Day Trading Support (December 5):**
- **Automatic multi-day data** - CSV appends new trading days automatically
- **Live data updates** - Downloads fresh data every 60 seconds during market hours
- **Market hours awareness** - Waits for next trading day when market closes
- **Interactive charts** - Zoom, pan, and explore all historical data
- **Continuous operation** - Can run indefinitely across multiple trading days

**Real Data Integration (December 2):**
- **Real stock data support** - Download actual historical intraday data from Yahoo Finance
- **Fixed portfolio tracking** - Positions display correctly with accurate P/L calculations
- **Improved LLM context** - Portfolio status appears at TOP of prompts (critical info first)
- **Better percentage tracking** - Live prices show total change from start
- **1-minute interval data** - Realistic minute-by-minute trading data

**Major Frontend Overhaul (November):**
- **Replaced Streamlit with Flask** - No more page reloads!
- **Real-time updates via SSE** - True live streaming
- **Modern UI with Tailwind CSS** - Clean, professional design
- **Live-updating charts** - Smooth animations, no flickering
- **Portfolio tracking** - Virtual trading with LLM-driven decisions
- **Batch data processing** - LLM sees all 4 stocks simultaneously

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Memory Management](#memory-management)
- [Prompting Strategies](#prompting-strategies)
- [Web Interface](#web-interface)
- [CLI Interface](#cli-interface)
- [Configuration](#configuration)
- [The Continuous Prompting Challenge](#the-continuous-prompting-challenge)
- [Current Progress](#current-progress)
- [Future Exploration](#future-exploration)
- [Development](#development)

## Overview

### What is This Project?

This framework explores **continuous prompting** - the challenge of feeding streaming data to LLMs in real-time while maintaining relevant context and managing token limits. Unlike traditional one-shot prompting, continuous prompting requires the LLM to process an ongoing stream of data points, remember relevant history, and provide contextual analysis.

### The Core Problem

When streaming data continuously to an LLM:
- Token limits constrain how much history can be included
- Context windows fill up quickly
- LLMs have no inherent memory between prompts
- Repeating context in every prompt is inefficient
- Important historical patterns may be lost

### Our Approach

We've implemented two memory management strategies to address these challenges:
1. **Sliding Window + Summarization** - Keeps recent data in full detail, summarizes older data
2. **ChromaDB Vector Storage + RAG** - Stores all data as embeddings, retrieves relevant context semantically

### Key Features

- **Real stock data support** - Download actual historical intraday data from Yahoo Finance (1-minute intervals)
- **Simulated data option** - Realistic random walk price generation for testing
- **Real-time data streaming** - Stream data at configurable intervals (default: 0.5 seconds)
- **News event injection** - Simulated news events (1-2 per day)
- **Reactive strategy** - LLM only responds to significant events (RECOMMENDED)
- **Built-in memory management** - Vector DB + sliding window for context retrieval
- **Ollama LLM integration** - Local, private, free (supports GPU acceleration)
- **Modern Flask web interface** - Real-time updates via Server-Sent Events (no page reloads!)
- **Live-updating charts** - Chart.js with tabbed interface for each stock
- **Portfolio tracking** - Virtual trading with LLM-driven BUY/SELL/HOLD decisions
- **Accurate P/L tracking** - Real-time profit/loss calculations with position details
- **CLI interface** - Terminal-based experiments for headless environments
- **Configurable everything** - Prompts, parameters, triggers, data sources

## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running ([Download](https://ollama.ai/))

### Setup Steps

1. **Clone or download this repository**

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Pull an Ollama model:**
```bash
ollama pull mistral
```

Other recommended models: `llama2:7b`, `phi`, `tinyllama`

### Optional: ChromaDB for Vector Storage

For the ChromaDB memory manager (recommended for production):
```bash
pip install chromadb sentence-transformers
```

## Quick Start

### Web Interface (Recommended)

```bash
python run.py
```

Open your browser to `http://localhost:5000`

**What you'll see:**
- **Live Stock Prices**: Real-time price updates with color-coded changes (no page reloads!)
- **Price Movement Chart**: Live-updating chart showing all 4 stocks
- **Portfolio Performance**: Cash, portfolio value, profit/loss, trade statistics
- **LLM Responses**: AI analysis as it happens, streamed in real-time

**Quick experiment:**
1. Click "Start" button
2. Watch live data stream and charts update automatically
3. See LLM responses appear in real-time
4. Monitor portfolio performance
5. Click "Stop" when done

**Automatic Real Data Download:**
- If you set `source: "csv"` in config.yaml, the app **automatically downloads** real stock data on startup
- **During market hours (9:30 AM - 4:00 PM ET)**: Updates CSV every 60 seconds with new data
- **After market hours**: Uses existing data (no updates needed)
- No manual download required!

**To use real data:**
```yaml
# config.yaml
data:
  source: "csv"  # Automatically downloads and updates during market hours
  csv_path: "data/raw/real_trading_data_1m_1d.csv"
```

**To use simulated data:**
```yaml
# config.yaml
data:
  source: "sample"  # Fast, works offline, no downloads
```

**Why Flask instead of Streamlit?**
- **No page reloads** - True real-time updates via Server-Sent Events
- **Smooth chart updates** - Charts update live without flickering
- **Better performance** - Lightweight and fast
- **Modern UI** - Clean design with Tailwind CSS

### Command Line Interface

```bash
python run.py --cli                      # Run CLI interface
python run.py --cli --max-iterations 50  # Limit to 50 data points
python run.py --cli --display compact    # Compact output mode
python run.py --cli --display minimal    # Minimal output (logs only)
```

**What you'll see:**
- Streaming data updates (one per second)
- LLM responses in formatted boxes
- Clean, color-coded terminal output
- Final statistics summary

## Architecture

### Project Structure

```
continuous-prompting/
├── frontend/             # Flask web interface
│   ├── app.py            # Flask application with SSE
│   └── templates/
│       └── index.html    # Modern UI with Tailwind CSS
├── src/
│   ├── app/              # User interfaces
│   │   └── cli.py        # Command-line interface
│   ├── data/             # Data streaming layer
│   │   ├── data_sources.py
│   │   ├── data_simulator.py
│   │   └── news_generator.py
│   ├── llm/              # LLM integration
│   │   └── ollama_client.py
│   ├── strategies/       # Prompting strategies
│   │   ├── base_strategy.py
│   │   ├── reactive_strategy.py  # RECOMMENDED
│   │   └── continuous_strategy.py
│   ├── memory/           # Memory management
│   │   ├── base_memory.py
│   │   ├── chromadb_memory.py
│   │   └── sliding_window.py
│   ├── portfolio/        # Portfolio management
│   │   └── portfolio_manager.py
│   ├── prompts/          # Prompt management
│   │   └── prompt_manager.py
│   └── utils/            # Utilities
│       ├── config_loader.py
│       └── display.py
├── config.yaml           # Configuration file
├── run.py                # CLI launcher
└── requirements.txt      # Dependencies
```

### Core Components

**1. Data Layer (`src/data/`)**
- `DataSimulator`: Orchestrates data streaming
- `SampleDataSource`: Generates realistic trading data
- Streams data at configurable intervals (default: 1 second)

**2. LLM Layer (`src/llm/`)**
- `OllamaClient`: Interfaces with Ollama API
- `PromptManager`: Manages system and user prompts
- Handles conversation history and streaming responses

**3. Strategy Layer (`src/strategies/`)**
- `BaseStrategy`: Abstract base class for all strategies
- `ContinuousStrategy`: Processes every data point (or batches)
- `EventDrivenStrategy`: Only responds to significant events

**4. Memory Layer (`src/memory/`)**
- `BaseMemoryManager`: Abstract interface for memory management
- `SlidingWindowMemoryManager`: Recent data + statistical summaries
- `ChromaMemoryManager`: Vector database with semantic search

**5. Interface Layer (`frontend/` and `src/app/`)**
- `frontend/app.py`: Flask web application with Server-Sent Events
- `frontend/templates/index.html`: Modern UI with real-time updates
- `src/app/cli.py`: Terminal-based interface

**6. Portfolio Layer (`src/portfolio/`)**
- `PortfolioManager`: Manages virtual trading portfolio
- Tracks cash, positions, trades, profit/loss
- Executes LLM trading decisions (BUY/SELL/HOLD)

## Data Pipeline

### How Data Flows Through the System

1. **Data Source** (Choose one)
   - **Real Data**: `CSVDataSource` loads historical intraday data from Yahoo Finance
     - Download with `python scripts/download_real_data.py`
     - 1-minute interval data for AAPL, GOOGL, MSFT, TSLA
     - Actual historical prices, volumes, and movements
   - **Simulated Data**: `SampleDataSource` generates realistic trading data
     - Random walk price simulation with 0.1% volatility
     - Configurable starting prices and volume ranges
     - Infinite data stream for testing

2. **Data Streaming**
   - `DataSimulator` orchestrates the stream
   - Yields batches of all 4 stocks per interval (default: 0.5 seconds)
   - Runs in background thread for non-blocking operation
   - Each batch includes: timestamp, symbol, price, change %, volume, high, low

3. **Strategy Processing**
   - Strategy receives each data point
   - Decides whether to prompt the LLM (continuous vs event-driven)
   - Retrieves relevant context from memory manager

4. **Memory Management**
   - Stores data point and LLM response
   - Manages context window to stay within token limits
   - Retrieves relevant historical context for next prompt

5. **LLM Prompting**
   - Builds prompt with current data + relevant context
   - Sends to Ollama via API
   - Receives streaming response

6. **Response Display**
   - Web interface: Updates tabs in real-time
   - CLI: Prints formatted response to terminal
   - Both: Logs to file for later analysis

### Data Point Structure

Each data point contains:
```python
{
    'symbol': 'AAPL',           # Stock ticker
    'price': 150.25,            # Current price
    'change': 1.5,              # Percentage change
    'volume': 1000000,          # Trading volume
    'timestamp': '2025-11-14T10:30:00'  # ISO format timestamp
}
```

### Data Generation Logic

The `SampleDataSource` simulates realistic market behavior:
- **Price Movement**: Random walk with configurable volatility
- **Volume**: Varies randomly around a base level
- **Trends**: Can simulate uptrends, downtrends, or sideways movement
- **Volatility**: Configurable per-symbol (default: 0.02 = 2%)

### News Event Injection

The `NewsGenerator` injects realistic news events into the data stream:
- **Frequency**: Configurable (default: 1.5 events per day = 1-2 events)
- **Categories**: Earnings, product launches, regulatory, market, executive changes
- **Sentiment**: Positive, negative, or neutral
- **Impact Levels**: Low, medium, or high
- **Examples**:
  - "AAPL beats earnings expectations by 15%"
  - "TSLA delays major product launch"
  - "GOOGL faces regulatory investigation"

News events are treated as special data points that can trigger LLM responses, especially useful with the reactive strategy.

## Memory Management

The framework includes two built-in memory management strategies to handle continuous data streams efficiently.

### 1. Sliding Window + Summarization (Default)

**How it works:**
- Keeps last N data points in full detail (default: 20)
- When window fills, oldest items move to pending summary queue
- Every M items (default: 10), creates a statistical summary
- Summaries include: avg price, min/max, avg change, count per symbol
- Keeps up to K summaries (default: 10)

**Best for:**
- Quick experiments
- Development and testing
- No database setup needed
- Predictable token usage

**Configuration:**
```yaml
memory:
  type: "sliding_window"
  sliding_window:
    window_size: 20           # Recent items in full detail
    summary_batch_size: 10    # Items per summary batch
    max_summaries: 10         # Max summaries to keep
    enable_llm_summary: false # Use LLM for summaries (experimental)
```

**Example context provided to LLM:**
```
Recent Data (detailed):
1. AAPL: $150.00 (+1.50%) | Vol: 1,000,000 | 2025-11-14T10:30:00
   Analysis: Positive momentum detected...
2. GOOGL: $2800.00 (-0.50%) | Vol: 2,000,000 | 2025-11-14T10:30:01
   Analysis: Slight pullback...
...

Historical Summaries:
Batch 1 (10 items, 2025-11-14T10:29:00 to 10:29:10):
  AAPL: avg $148.50, range $147.00-$150.00, avg change +0.8%
  GOOGL: avg $2795.00, range $2790.00-$2800.00, avg change -0.2%
```

### 2. ChromaDB Vector Storage + RAG

**How it works:**
- Stores ALL data points as embeddings in ChromaDB
- Uses sentence-transformers for embedding generation (all-MiniLM-L6-v2)
- Retrieves top-K most semantically similar items for each prompt
- Persistent storage across sessions
- Supports semantic search (e.g., "similar price movements for AAPL")

**Best for:**
- Production systems
- Long-running experiments
- Pattern recognition across large datasets
- Unlimited history with efficient token usage

**Configuration:**
```yaml
memory:
  type: "chromadb"
  chromadb:
    collection_name: "trading_data"
    persist_directory: "./data/chroma"
    top_k: 5                  # Number of similar items to retrieve
    embedding_model: "all-MiniLM-L6-v2"
```

**Installation:**
```bash
pip install chromadb sentence-transformers
```

**Example context provided to LLM:**
```
Most Relevant Historical Context (5 items):

1. AAPL: $149.50 (+1.20%) | Vol: 950,000 | 2025-11-14T10:25:00
   LLM Analysis: Strong upward momentum...

2. AAPL: $148.00 (+0.80%) | Vol: 1,100,000 | 2025-11-14T10:20:00
   LLM Analysis: Consistent buying pressure...

[Items selected based on semantic similarity to current data point]
```

### 3. No Memory Management

Set `memory.type: "none"` to disable memory management and use basic conversation history. This will hit token limits quickly but is useful for baseline comparisons.

### Memory Manager Interface

All memory managers implement the same interface:

```python
class BaseMemoryManager:
    def add_data_point(self, data: Dict, response: Optional[str] = None):
        """Store a data point and its LLM response"""

    def get_context(self, current_data: Dict, max_tokens: int = 2000) -> str:
        """Retrieve relevant context for the current data point"""

    def clear(self):
        """Clear all stored data"""

    def get_stats(self) -> Dict:
        """Get memory statistics (total items, memory type, etc.)"""
```

This allows easy swapping between memory strategies without changing strategy code.

## Prompting Strategies

### Autonomous Strategy (Current Implementation)

**The Problem We Solved:**

Traditional approaches use **hardcoded rules** to decide when the LLM should respond:
- "Activate if price changes > 1.5%"
- "Activate if volume > 2x average"
- "Activate every 60 seconds"

**The issue?** These rules are arbitrary. What's "significant" changes based on:
- Market conditions (volatile vs stable)
- Stock characteristics (tech vs utilities)
- Time of day (market open vs close)
- Recent context (already trending vs sudden move)

**Our Solution: Let the LLM Decide**

Instead of hardcoded rules, we ask the LLM itself: **"Is this market state worth analyzing?"**

The LLM considers:
- Current price movements across all stocks
- Recent market activity (last 10 data points)
- Current market volatility
- Whether this is just noise or a real signal

This is **fundamentally different** from rule-based systems. The LLM learns what's important through its training, not through our arbitrary thresholds.

---

### How It Works

#### 1. **Self-Activation (Tool Use Pattern)**

Every 5 data points, we ask the LLM:

```
You are monitoring live market data. Should you activate and analyze?

CURRENT MARKET STATE:
AAPL: $278.50 (+2.1%) Vol: 77,980
GOOGL: $320.40 (+0.5%) Vol: 45,230
MSFT: $481.85 (+0.1%) Vol: 32,100
TSLA: $454.66 (-0.1%) Vol: 51,200

RECENT ACTIVITY (last 10 data points):
[9:35] AAPL: $275.20 (+0.8%), GOOGL: $320.10 (+0.3%)...
[9:36] AAPL: $276.40 (+1.2%), GOOGL: $320.25 (+0.4%)...
...

MARKET VOLATILITY: 0.0142

Should you activate? Respond in JSON:
{
  "should_activate": true/false,
  "reason": "brief explanation",
  "confidence": 0.0-1.0
}
```

**Why this works:**
- LLM sees the full context, not just isolated metrics
- LLM can recognize complex patterns (e.g., "AAPL momentum with volume confirmation")
- LLM adapts to market conditions (what's significant in volatile vs calm markets)
- Confidence scoring prevents false positives

**Example LLM response:**
```json
{
  "should_activate": true,
  "reason": "AAPL showing strong upward momentum (+2.1%) with 2.3x normal volume - potential breakout",
  "confidence": 0.88
}
```

Only activates if `confidence >= 0.7` (configurable).

---

#### 2. **Chain-of-Thought Reasoning (CoT)**

**Why we use it:**
Research shows LLMs make significantly better decisions when they "think out loud" step-by-step, rather than jumping directly to conclusions.

When activated, the LLM follows a structured reasoning process:

```
STEP 1 - OBSERVATION:
What do you observe in the current market data? List specific facts.

STEP 2 - PATTERN RECOGNITION:
Are there any patterns, trends, or anomalies?

STEP 3 - OPPORTUNITY IDENTIFICATION:
Based on your observations, are there trading opportunities?

STEP 4 - RISK ASSESSMENT:
What are the risks of each potential action?

STEP 5 - DECISION:
Make your final trading decision(s).
```

**Example LLM response:**
```
OBSERVATION:
- AAPL: $278.50 (+2.1%) with 77,980 volume (2.3x average)
- Other stocks relatively flat (+0.5% or less)
- AAPL has been climbing steadily for last 6 data points

PATTERNS:
- AAPL showing sustained upward momentum, not just a spike
- Volume confirms genuine buying interest, not just price noise
- Other stocks stable - AAPL-specific catalyst likely

OPPORTUNITIES:
- BUY AAPL - Momentum + volume suggests continuation
- HOLD others - No clear signals

RISKS:
- Already up 2.1% - might be late to the move
- Could be false breakout if volume drops
- Confidence: 75%

DECISION: BUY AAPL $500
```

**Why this is better than direct prompting:**
- Forces systematic analysis (prevents impulsive decisions)
- Makes reasoning transparent (you can see WHY it decided)
- Catches logical errors (if reasoning is flawed, decision likely is too)
- Improves decision quality (proven by research: [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903))

---

#### 3. **Self-Reflection (Verification)**

**Why we use it:**
Even with CoT, LLMs can make mistakes. Self-reflection adds a verification layer where the LLM critiques its own decision.

After making a decision, we ask:

```
You just decided to BUY AAPL $500. Now critique your own decision:

1. SANITY CHECK:
   - Do you have enough cash? (Current: $1,000)
   - Did you verify the stock is actually moving up?
   - Is this based on solid reasoning or emotion?

2. RISK ASSESSMENT:
   - What's the worst-case scenario?
   - Is the reward worth the risk?

3. ALTERNATIVE PERSPECTIVES:
   - What would a contrarian investor say?
   - Are you missing any information?

4. FINAL VERDICT:
   - Stand by your decision? (YES/NO)
   - If NO, what should you do instead?
```

**Example LLM response:**
```
REFLECTION:

SANITY CHECK:
- Cash available: $1,000
- $500 < $1,000 - can afford
- AAPL is up 2.1% with high volume - solid technical signal
- Decision based on momentum + volume confirmation

RISK ASSESSMENT:
- Worst case: AAPL reverses, lose ~$50 (10% drop)
- Best case: Continues up, gain ~$100 (20% gain)
- Risk/reward ratio: 1:2 - acceptable

ALTERNATIVE PERSPECTIVES:
- Contrarian view: "Already up 2.1%, too late to enter"
- Counter-argument: Volume confirms this is early in the move, not late
- Missing info: None critical - have price, volume, recent history

FINAL VERDICT: CONFIRMED
CONFIDENCE: 75%
```

**Why this works:**
- Catches mistakes before execution (e.g., trying to buy without cash)
- Provides second opinion (reduces overconfidence)
- Can revise decisions if flaws found
- Improves decision quality (proven by research: [Self-Consistency](https://arxiv.org/abs/2203.11171))

---

#### 4. **ReAct Pattern (Reasoning + Acting)**

The strategy follows the ReAct framework ([paper](https://arxiv.org/abs/2210.03629)):

1. **Observe** → Monitor market data silently
2. **Reason** → Decide if activation needed (self-activation)
3. **Act** → Perform analysis if activated (Chain-of-Thought)
4. **Reflect** → Verify decision quality (self-reflection)
5. **Execute** → Make trades only if verified

This creates a **reasoning loop** where the LLM continuously evaluates and adapts.

---

### Configuration

```yaml
strategy:
  type: "autonomous"

  autonomous:
    # How often to ask LLM if it wants to activate
    # Lower = more frequent checks, Higher = more selective
    activation_check_interval: 5

    # Minimum confidence (0.0-1.0) required to activate
    # Higher = more selective, Lower = more responsive
    min_confidence_threshold: 0.7

    # Enable Chain-of-Thought reasoning
    # Dramatically improves decision quality
    enable_chain_of_thought: true

    # Enable self-reflection
    # LLM critiques its own decisions
    enable_self_reflection: true

    # Enable trading
    enable_trading: true
```

**Tuning Guide:**

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| `activation_check_interval` | 10 | 5 | 3 |
| `min_confidence_threshold` | 0.8 | 0.7 | 0.6 |
| `enable_chain_of_thought` | true | true | true |
| `enable_self_reflection` | true | true | false* |

*Aggressive mode can skip reflection for faster execution

---

### Why These Techniques Matter

**1. Self-Activation (vs Hardcoded Triggers)**
- Hardcoded: "Activate if price > 1.5%" → Misses context
- Autonomous: LLM considers full market state → Adapts to conditions

**2. Chain-of-Thought (vs Direct Prompting)**
- Direct: "Should I buy AAPL?" → Impulsive decision
- CoT: Step-by-step reasoning → Systematic analysis

**3. Self-Reflection (vs Single-Pass)**
- Single-pass: Make decision, execute → No verification
- Reflection: Make decision, critique, then execute → Catches mistakes

**These aren't just "nice to have" features** - they're proven techniques from AI research that measurably improve decision quality.

---

### Example: Full Autonomous Flow

```
[Data Point 1-4] → Stored silently
[Data Point 5] → Self-Activation Check

LLM: {
  "should_activate": true,
  "reason": "AAPL up 2.1% with high volume",
  "confidence": 0.88
}

Confidence 0.88 >= 0.70 → ACTIVATE

--- CHAIN-OF-THOUGHT ANALYSIS ---

OBSERVATION:
- AAPL: $278.50 (+2.1%) Vol: 77,980
- Volume 2.3x higher than average
- Sustained climb over last 6 data points

PATTERNS:
- Strong upward momentum with volume confirmation
- Other stocks flat - AAPL-specific move

OPPORTUNITIES:
- BUY AAPL - Momentum likely to continue

RISKS:
- Could reverse if volume drops
- Already up 2.1% - might be late

CONFIDENCE: 75%
DECISION: BUY AAPL $500

--- SELF-REFLECTION ---

SANITY CHECK:
- Have $1,000 cash, buying $500 - OK
- Based on solid technical signals

RISK ASSESSMENT:
- Risk/reward: 1:2 - acceptable

FINAL VERDICT: CONFIRMED

Trade Executed: BUY $500 AAPL @ $278.50
```

**This entire reasoning process appears in the frontend** so you can see exactly why the LLM made each decision.

---

### Legacy Strategies (Removed)

**Reactive Strategy** (removed December 6, 2025):
- Used hardcoded triggers (price change > 1.5%, volume > 2x, etc.)
- Could not adapt to market conditions
- No reasoning transparency
- No self-verification

**Why we removed it:**
- Autonomous strategy is strictly superior
- Simpler codebase (one strategy vs multiple)
- Better decision quality
- More transparent reasoning
- Truly intelligent behavior vs rule-following

If you need the old reactive strategy, check git history before December 6, 2025
---

### Custom Strategy Development

You can extend `BaseStrategy` to implement your own logic:

```python
from src.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def process_data_point(self, data):
        # Your custom logic here
        if self.should_prompt(data):
            context = self.memory_manager.get_context(data)
            prompt = self.build_custom_prompt(data, context)
            response = self.llm_client.chat(prompt)
            self.memory_manager.add_data_point(data, response)
            return response
        return None
```

**Note:** The autonomous strategy is highly configurable and should handle most use cases. Only create a custom strategy if you need fundamentally different behavior.

## Web Interface

### Launching

```bash
python frontend/app.py
```

The server will start on `http://localhost:5000`. Open this URL in your browser.

### Interface Overview

**Header:**
- Status indicator (Running/Stopped with animated dot)
- Control buttons: Start, Stop, Reset
- Real-time status updates

**Top Metrics Bar (4 cards):**
- **Data Points**: Total data points processed
- **LLM Responses**: Total LLM responses generated
- **Cash Available**: Current cash in portfolio
- **Portfolio Value**: Total portfolio value (stocks + cash)

**Left Column:**

**Live Stock Prices Card:**
- Real-time price updates for all 4 stocks (AAPL, GOOGL, MSFT, TSLA)
- Color-coded price changes (green = up, red = down)
- Timestamp for each update
- **Updates live without page reload!**

**Price Movement Chart:**
- Live-updating line chart showing all 4 stocks
- Color-coded lines (AAPL=blue, GOOGL=red, MSFT=green, TSLA=orange)
- Last 50 data points visible
- Smooth updates without redrawing entire chart

**Right Column:**

**Portfolio Performance Card:**
- Total Trades, Winning Trades, Win Rate
- Profit/Loss with percentage
- Color-coded (green = profit, red = loss)
- Updates in real-time as trades execute

**LLM Responses Card:**
- Scrollable list of all LLM responses
- Newest responses appear at top
- Each response shows:
  - Timestamp
  - Stock prices at time of response
  - Full LLM analysis
- Auto-scrolls as new responses arrive

### How It Works (Technical)

**Server-Sent Events (SSE):**
- Flask backend streams events to frontend via `/stream` endpoint
- Frontend connects with `EventSource` API
- Events pushed in real-time: data updates, LLM responses, portfolio updates
- No polling, no page reloads - true push-based updates

**Event Types:**
- `data`: New stock price data (updates prices and chart)
- `response`: New LLM response (adds to response list)
- `portfolio`: Portfolio update (updates metrics)
- `heartbeat`: Keep-alive ping every 5 seconds

**API Endpoints:**
- `GET /`: Main page
- `POST /api/start`: Start simulation
- `POST /api/stop`: Stop simulation
- `POST /api/reset`: Reset simulation
- `GET /api/status`: Get current status
- `GET /api/portfolio`: Get portfolio summary
- `GET /stream`: SSE stream for real-time updates

### Usage Tips

**For quick experiments:**
1. Click "Start" button
2. Watch live stock prices update
3. See chart update in real-time
4. Monitor LLM responses as they appear
5. Click "Stop" when done

**For portfolio tracking:**
1. Start simulation
2. Watch portfolio metrics update
3. See LLM make trading decisions (BUY/SELL/HOLD)
4. Monitor profit/loss in real-time
5. Check win rate and trade statistics

**For comparing strategies:**
1. Edit `config.yaml` to change strategy
2. Click "Reset" to clear data
3. Click "Start" to run with new strategy
4. Compare response frequency and quality



## CLI Interface

### Launching

```bash
python run.py
```

### Command-Line Options

```bash
python run.py --max-iterations 50   # Limit to 50 data points
python run.py --display full        # Full mode (data + responses)
python run.py --display compact     # Compact mode (responses only)
python run.py --display minimal     # Minimal mode (logs only)
```

### Output Modes

**Full Mode (default):**
- Shows streaming data updates
- Shows LLM responses in formatted boxes
- Color-coded output
- Progress indicators

**Compact Mode:**
- Only shows LLM responses
- Cleaner output for focusing on AI analysis
- Still logs all data

**Minimal Mode:**
- Quiet console output
- Everything logged to files
- Good for production runs

### What You'll See

```
=== Data Stream ===
[2025-11-14 10:30:00] AAPL: $150.25 (+1.5%) | Vol: 1,000,000

=== LLM Response #1 ===
Strong upward momentum detected for AAPL. Price has increased 1.5%
with healthy volume. This suggests continued buying pressure...

=== Statistics ===
Data Points: 50
LLM Responses: 50
Response Rate: 100%
Elapsed Time: 50.2s
```

## Configuration

All settings are in `config.yaml`. The file is well-commented and organized into sections.

### LLM Settings

```yaml
llm:
  model: "mistral"      # Ollama model name
  temperature: 0.7      # Creativity (0.0 = deterministic, 1.0 = creative)
  max_tokens: 500       # Maximum response length
  base_url: "http://localhost:11434"  # Ollama API URL
```

**Available models:** mistral, llama2:7b, phi, tinyllama, codellama, etc.
**Temperature guide:** 0.3 (focused), 0.7 (balanced), 0.9 (creative)

### Data Settings

```yaml
data:
  update_interval: 1.0  # Seconds between data points
  source: "sample"      # Options: sample, csv

  # Sample data source settings
  sample:
    symbols: ["AAPL", "GOOGL", "MSFT", "TSLA"]
    volatility: 0.02    # Price volatility (2%)
    base_prices:
      AAPL: 150.0
      GOOGL: 2800.0
      MSFT: 380.0
      TSLA: 250.0

  # News injection settings
  news:
    enabled: true       # Enable news event injection
    events_per_day: 1.5 # Average number of news events per day (1-2 events)
```

**News events** are injected into the data stream at realistic intervals. They include:
- Earnings reports (beats/misses expectations)
- Product launches (successful/delayed)
- Regulatory news (investigations, approvals)
- Market events (analyst upgrades/downgrades)
- Executive changes (CEO appointments, departures)

### Strategy Settings

```yaml
strategy:
  type: "reactive"  # Options: reactive, continuous, event_driven (reactive recommended)

  # Reactive strategy settings (RECOMMENDED)
  reactive:
    check_interval: 1        # Check triggers every N data points
    alert_cooldown: 60       # Minimum seconds between alerts for same symbol
    triggers:
      - type: "price_change"
        threshold: 0.03      # 3% price change triggers alert
      - type: "volume_spike"
        threshold: 2.5       # 2.5x average volume triggers alert
      - type: "news_event"   # Any news event triggers alert
      - type: "pattern"
        pattern_type: "consecutive_moves"
        count: 3             # 3 consecutive moves in same direction
      - type: "time_interval"
        interval: 300        # Periodic check-in every 300 data points

  # Continuous strategy settings
  continuous:
    batch_size: 1     # Process every N data points

  # Event-driven strategy settings
  event_driven:
    triggers:
      - type: "price_change"
        threshold: 0.02      # 2% price change
      - type: "volume_spike"
        threshold: 2.0       # 2x average volume
      - type: "time_interval"
        seconds: 60          # Every 60 seconds
```

### Memory Settings

```yaml
memory:
  type: "sliding_window"  # Options: sliding_window, chromadb, none

  # Sliding Window Memory Settings
  sliding_window:
    window_size: 20           # Number of recent items to keep in full detail
    summary_batch_size: 10    # Items per summary batch
    max_summaries: 10         # Maximum number of summaries to keep
    enable_llm_summary: false # Use LLM for summaries (experimental)

  # ChromaDB Vector Memory Settings
  chromadb:
    collection_name: "trading_data"
    persist_directory: "./data/chroma"
    top_k: 5                  # Number of similar items to retrieve
    embedding_model: "all-MiniLM-L6-v2"
```

### Prompt Settings

```yaml
prompts:
  system_prompt: |
    You are an expert financial analyst monitoring real-time trading data.
    Provide concise, actionable insights based on price movements, volume,
    and historical patterns. Focus on identifying trends and anomalies.

  user_prompt_template: |
    Current Data: {current_data}

    Analyze this data point and provide insights.
```

## The Continuous Prompting Challenge

### The Problem

Traditional LLM prompting is one-shot: you send a prompt, get a response, done. But what happens when you have a continuous stream of data?

**Challenges:**
1. **Token Limits**: LLMs have finite context windows (typically 2k-32k tokens)
2. **No Persistent Memory**: Each prompt is independent unless you manually manage history
3. **Inefficiency**: Repeating context in every prompt wastes tokens
4. **Context Loss**: Can't include all historical data, must choose what to keep
5. **Pattern Recognition**: LLM needs historical context to identify patterns

**Example scenario:**
- Data streams at 1 point/second
- Each data point is ~50 tokens
- After 1 hour: 3,600 data points = 180,000 tokens
- Most models: 4k-8k token limit
- Problem: Can only include last ~80 data points in context

### Our Solutions (Implemented)

**1. Sliding Window + Summarization**
- Keeps recent data in full detail (last 20 points)
- Summarizes older data into compact statistics
- Predictable token usage
- No external dependencies

**Pros:**
- Simple, no database needed
- Predictable memory usage
- Fast retrieval
- Good for short-term patterns

**Cons:**
- Loses fine-grained historical detail
- Summary quality depends on aggregation method
- May miss long-term patterns

**2. Vector Database + RAG**
- Stores ALL data as embeddings
- Retrieves semantically similar items
- Unlimited history
- Efficient token usage

**Pros:**
- Unlimited history storage
- Semantic search finds relevant context
- Scales to millions of data points
- Persistent across sessions

**Cons:**
- Requires ChromaDB installation
- Embedding generation overhead
- More complex setup

### Future Exploration Ideas

**3. Continuous Fine-Tuning**
- Fine-tune model on incoming data in real-time
- Model actually "learns" from the stream
- Requires: Training infrastructure, checkpointing
- Most complex but potentially most powerful

**4. Hybrid Approach**
- Combine memory management with periodic fine-tuning
- Use RAG for short-term context
- Fine-tune for long-term learning
- Best of both worlds

**5. Adaptive Context Selection**
- Use LLM to decide what historical context is relevant
- Dynamic context window based on current data
- Meta-prompting for context selection

## Current Progress

### What We've Built

**Core Infrastructure:**
- Data streaming layer with realistic simulation
- News event injection system (1-2 events per day)
- Ollama LLM integration with streaming responses
- Strategy pattern for different prompting approaches
- Web and CLI interfaces for experimentation
- Comprehensive configuration system

**Memory Management:**
- Sliding window with automatic statistical summarization
- ChromaDB vector storage with semantic search
- Pluggable memory manager interface
- Configurable via YAML and web UI

**Prompting Strategies:**
- Reactive strategy with configurable triggers (RECOMMENDED)
- Continuous strategy with batching support
- Event-driven strategy with multiple trigger types
- Memory-aware context retrieval
- Conversation history management

**News Event System:**
- Realistic news generation (earnings, products, regulatory, market, executive)
- Configurable frequency (events per day)
- Sentiment analysis (positive, negative, neutral)
- Impact levels (low, medium, high)
- Integration with reactive strategy for event-driven alerts

**User Interfaces:**
- Streamlit web dashboard with live charts
- News event display with color-coded sentiment
- CLI with multiple display modes
- Real-time metrics and statistics
- CSV export for offline analysis

### What Works Well

- Data streaming is smooth and configurable
- News injection adds realistic market events
- Reactive strategy solves the continuous prompting problem elegantly
- Memory managers effectively manage token limits
- Web interface provides excellent visibility
- Ollama integration is fast and reliable
- Configuration system is flexible
- Pattern detection catches important sequences
- Alert cooldown prevents spam

### Known Limitations

- LLM-based summarization not implemented (only statistical summaries)
- No support for custom data sources via web UI (only CLI/config)
- No built-in evaluation metrics for response quality
- No A/B testing framework for comparing strategies

## Future Exploration

### Immediate Next Steps

1. Integrate memory managers with event-driven strategy
2. Implement LLM-based summarization option
3. Add response quality metrics
4. Build evaluation framework for comparing strategies
5. Add support for real CSV data in web UI

### Research Directions

1. **Memory Management:**
   - Experiment with different window sizes and summary intervals
   - Compare ChromaDB vs sliding window performance
   - Implement hybrid memory strategies
   - Test with different embedding models

2. **Prompting Strategies:**
   - Adaptive batch sizing based on data volatility
   - Multi-level event triggers (minor, major, critical)
   - Predictive prompting (prompt before events occur)
   - Collaborative strategies (multiple LLMs)

3. **Fine-Tuning:**
   - Periodic fine-tuning on accumulated data
   - Transfer learning from one symbol to another
   - Continuous learning with catastrophic forgetting prevention

4. **Evaluation:**
   - Response quality metrics (relevance, accuracy, actionability)
   - Token usage efficiency
   - Latency and throughput benchmarks
   - A/B testing framework

5. **Production Readiness:**
   - Error handling and recovery
   - Monitoring and alerting
   - Scalability testing
   - Multi-user support

## Development

### Project Structure Details

**src/app/cli.py:**
- Command-line argument parsing
- Terminal display formatting
- Progress tracking
- Statistics reporting

**frontend/app.py:**
- Flask application with Server-Sent Events
- Real-time data streaming via SSE
- Background thread for simulation
- API endpoints for control (start/stop/reset)

**frontend/templates/index.html:**
- Modern UI with Tailwind CSS
- Chart.js for live-updating charts
- JavaScript EventSource for SSE connection
- Real-time DOM updates without page reloads

**src/data/simulator.py:**
- Main orchestrator for data streaming
- Manages data source lifecycle
- Yields data points at configured intervals
- Thread-safe operation

**src/data/sample_source.py:**
- Generates realistic trading data
- Random walk price simulation
- Volume variation
- Configurable volatility per symbol

**src/llm/ollama_client.py:**
- Ollama API integration
- Streaming response handling
- Conversation history management
- Model verification and error handling

**src/llm/prompt_manager.py:**
- System and user prompt management
- Template rendering
- Prompt formatting utilities

**src/strategies/base_strategy.py:**
- Abstract base class for all strategies
- Defines interface: `process_data_point()`, `get_stats()`, `reset()`
- Common utilities for strategies

**src/strategies/continuous_strategy.py:**
- Implements continuous prompting
- Batching logic
- Memory manager integration
- Context retrieval and prompt building

**src/strategies/event_driven_strategy.py:**
- Implements event-driven prompting
- Trigger evaluation (price change, volume spike, time interval)
- Event detection logic

**src/memory/base_memory.py:**
- Abstract interface for memory managers
- Defines: `add_data_point()`, `get_context()`, `clear()`, `get_stats()`

**src/memory/sliding_window_memory.py:**
- Sliding window implementation
- Statistical summarization
- Deque-based efficient window management

**src/memory/chroma_memory.py:**
- ChromaDB integration
- Embedding generation
- Semantic similarity search
- Persistent storage

### Adding a Custom Strategy

1. Create a new file in `src/strategies/`:

```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, llm_client, prompt_manager, config=None, memory_manager=None):
        super().__init__(llm_client, prompt_manager, config)
        self.memory_manager = memory_manager
        # Your initialization

    def process_data_point(self, data):
        # Your logic to decide when to prompt
        if self.should_prompt(data):
            # Get context from memory
            context = self.memory_manager.get_context(data)

            # Build prompt
            prompt = f"Context: {context}\nCurrent: {data}\nAnalyze:"

            # Get LLM response
            response = self.llm_client.chat(prompt, maintain_history=True)

            # Store in memory
            self.memory_manager.add_data_point(data, response)

            return response
        return None

    def should_prompt(self, data):
        # Your custom logic
        return True
```

2. Register in `config.yaml`:

```yaml
strategy:
  type: "my_strategy"
```

3. Update strategy factory in `src/app/cli.py` and `src/app/web.py`

### Adding a Custom Memory Manager

1. Create a new file in `src/memory/`:

```python
from src.memory.base_memory import BaseMemoryManager
from typing import Dict, Optional, Any

class MyMemoryManager(BaseMemoryManager):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Your initialization

    def add_data_point(self, data: Dict[str, Any], response: Optional[str] = None):
        # Store data and response
        pass

    def get_context(self, current_data: Dict[str, Any], max_tokens: int = 2000) -> str:
        # Retrieve relevant context
        return "context string"

    def clear(self):
        # Clear all stored data
        pass

    def get_stats(self) -> Dict[str, Any]:
        # Return statistics
        return {"total_items": 0, "memory_type": "My Memory"}
```

2. Register in `config.yaml`:

```yaml
memory:
  type: "my_memory"
  my_memory:
    # Your config options
```

3. Update memory factory in strategies

### Running Tests

```bash
# Test memory managers
python test_memory.py

# Test with different configurations
python run.py --max-iterations 10 --display full
```

### Troubleshooting

**"Connection refused" to Ollama:**
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

**"Model not found":**
```bash
# List available models
ollama list

# Pull a model
ollama pull mistral
```

**Import errors:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**Web interface won't load:**
```bash
# Check if Flask is installed
pip install flask

# Try a different port (edit frontend/app.py, change port=5000 to port=5001)
python frontend/app.py

# Check if port 5000 is already in use
# Windows: netstat -ano | findstr :5000
# Linux/Mac: lsof -i :5000
```

**ChromaDB errors:**
```bash
# Install ChromaDB dependencies
pip install chromadb sentence-transformers

# Clear ChromaDB data if corrupted
rm -rf data/chroma
```

### Contributing

This is an experimental framework for research and learning. Feel free to:
- Experiment with different approaches
- Implement new strategies
- Add new memory managers
- Improve the UI
- Add evaluation metrics
- Share your findings

### Project Philosophy

**This framework is intentionally:**
- **Experimental**: Built for exploration, not production
- **Modular**: Easy to swap components and try new approaches
- **Educational**: Code is readable and well-commented
- **Flexible**: Highly configurable for different experiments
- **Local-first**: Uses Ollama for privacy and cost-free experimentation

**This framework is NOT:**
- A production-ready trading system
- Financial advice or recommendations
- Optimized for performance
- A complete solution (it's a starting point)

### License

This project is provided as-is for educational and research purposes.

### Acknowledgments

- Ollama for local LLM inference
- Flask for the web framework
- Chart.js for live-updating charts
- Tailwind CSS for modern UI styling
- ChromaDB for vector storage
- The open-source LLM community

---

## Using Real Stock Data

### Quick Setup (Automatic!)

**Step 1: Update config.yaml**
```yaml
data:
  source: "csv"
  csv_path: "data/raw/real_trading_data_1m_1d.csv"
```

**Step 2: Run the App**
```bash
python run.py
```

**That's it!** The app will:
- Automatically download data if the CSV doesn't exist
- Automatically download fresh data if the CSV is from a previous day
- **During market hours**: Update the CSV every 60 seconds with new data
- **After market hours**: Use the complete day's data (no updates needed)

**Manual Download (Optional):**

If you want to download data manually:
```bash
python scripts/download_real_data.py
```

### What You Get

**Real Data:**
- Actual historical prices from Yahoo Finance
- Real trading volumes
- Realistic market movements
- Accurate percentage changes

**Example:**
- AAPL: 390 data points from 9:30 AM to 4:00 PM ET
- GOOGL: 390 data points from 9:30 AM to 4:00 PM ET
- MSFT: 390 data points from 9:30 AM to 4:00 PM ET
- TSLA: 390 data points from 9:30 AM to 4:00 PM ET

**Total: ~1,560 data points** (one full trading day)

### Customization

**Download Different Time Periods:**

Edit `scripts/download_real_data.py`:

```python
# 5-minute intervals for 5 days
INTERVAL = '5m'
PERIOD = '5d'

# 15-minute intervals for 1 month
INTERVAL = '15m'
PERIOD = '1mo'
```

**Download Different Stocks:**

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

### Data Playback Speed

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

**Note:** If you downloaded 1-minute interval data, setting `update_interval: 0.5` means each real minute of trading will play back in 0.5 seconds (120x speed). A full trading day (6.5 hours) plays back in ~3 minutes!

### Market Hours and Data Availability

**Does the app only work during market hours?**

**No! The app works 24/7.**

**With Simulated Data (default):**
- Works anytime, anywhere
- No dependency on market hours or internet
- Perfect for testing and development

**With Real Historical Data (CSV):**
- Works anytime, anywhere
- Plays back previously downloaded historical data
- No dependency on market hours (after data is downloaded)
- Data is from actual trading days, but you can replay it anytime

**Market hours only matter when downloading new data:**

**Yahoo Finance Data Availability:**
- **Market hours**: 9:30 AM - 4:00 PM ET (Monday-Friday)
- **Weekends**: No trading data available
- **Holidays**: No trading data on market holidays
- **After hours**: Limited data availability

**What this means:**
- If you run the download script on a **Saturday**, it will download Friday's data
- If you run it on a **Monday at 2 PM ET**, it will download Monday's data up to 2 PM
- If you run it on a **Monday at 8 AM ET**, it will download Friday's data (Monday hasn't started yet)

**How the CSV changes throughout the day:**
- **During market hours**: Each time you download, you get more data points
  - 10:00 AM: ~30 data points (9:30-10:00 AM)
  - 12:00 PM: ~150 data points (9:30 AM-12:00 PM)
  - 4:00 PM: ~390 data points (full trading day)
- **After market close**: You get the complete day's data (~390 points)
- **Next day**: The download gets the new day's data

**Best practice:** Download data after market close (after 4:00 PM ET) to get a complete trading day.

### Yahoo Finance Limitations

- **1-minute data**: Only last 7 days available
- **5-minute data**: Several months available
- **15-minute+ data**: Years of data available
- **Market hours only**: No data outside 9:30 AM - 4:00 PM ET
- **No weekends**: No data on weekends or holidays
- **Free and unlimited**: No API key required

### Troubleshooting

**"No data downloaded"**
- Market may be closed (weekends, holidays, after hours)
- Try downloading data for a previous weekday
- Try a longer period: `PERIOD = '5d'`

**"CSV file not found"**
- Make sure you ran `python scripts/download_real_data.py` first
- Check that the file exists: `data/raw/real_trading_data_1m_1d.csv`
- Check the path in `config.yaml` matches the downloaded file

**"Data plays back too fast/slow"**
- Adjust `update_interval` in `config.yaml`
- Smaller = faster, larger = slower

### Benefits of Real Data

- **Realistic testing** - See how your LLM responds to real market conditions
- **Reproducible** - Same data every time you run it
- **Historical events** - Includes real news events, earnings, etc.
- **Accurate patterns** - Real support/resistance levels, trends, volatility
- **Better training** - Helps tune your LLM prompts and triggers

---

## Summary

This framework provides a solid foundation for experimenting with continuous LLM prompting. We've addressed the core challenge of managing context and memory through two built-in strategies: sliding window with summarization and vector database with RAG.

The data pipeline is straightforward: generate or load data, stream it at regular intervals, process through a strategy that decides when to prompt, retrieve relevant context from memory, send to LLM, and display results.

Memory management is the key innovation here. Instead of naively including all history (impossible due to token limits) or only recent data (losing important patterns), we use intelligent strategies to maintain relevant context efficiently.

The web interface provides excellent visibility into what's happening, while the CLI is perfect for automated experiments. Both use the same underlying components, so you can switch between them freely.

Current limitations are documented, and there's plenty of room for exploration. The modular architecture makes it easy to implement new strategies, memory managers, or data sources.

This is a research tool, not a production system. Use it to learn, experiment, and explore the fascinating challenge of continuous prompting. The code is yours to modify, extend, and improve.

Happy experimenting!




