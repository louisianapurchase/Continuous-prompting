"""
Streamlit Web Interface for Continuous Prompting Framework

A beautiful, interactive web UI to control and visualize the continuous prompting experiments.
"""

import streamlit as st
import yaml
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import TradingDataSimulator, SampleDataSource, CSVDataSource
from src.data.news_generator import NewsGenerator
from src.llm import OllamaClient, PromptManager
from src.strategies import ContinuousStrategy, EventDrivenStrategy
from src.strategies.reactive_strategy import ReactiveStrategy
from src.memory import ChromaMemoryManager, SlidingWindowMemoryManager
from src.portfolio import PortfolioManager

logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Continuous Prompting Framework",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .response-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    .data-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9rem;
        color: #1a1a1a;
    }
    .news-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    .news-positive {
        border-left-color: #4caf50;
        background-color: #e8f5e9;
    }
    .news-negative {
        border-left-color: #f44336;
        background-color: #ffebee;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'response_history' not in st.session_state:
    st.session_state.response_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'strategy' not in st.session_state:
    st.session_state.strategy = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = None


def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def create_components(config, custom_settings):
    """Create data source, simulator, LLM client, strategy, and portfolio manager."""
    # Create data source
    if custom_settings['data_source'] == 'sample':
        data_source = SampleDataSource(
            symbols=custom_settings['symbols'],
            price_volatility=custom_settings['volatility'],
        )
    else:
        data_source = CSVDataSource(custom_settings['csv_path'])

    # Create news generator if enabled
    news_generator = None
    if custom_settings.get('news_enabled', False):
        news_generator = NewsGenerator(
            symbols=custom_settings['symbols'],
            events_per_day=custom_settings.get('news_events_per_day', 1.5),
            enabled=True
        )

    # Create simulator
    simulator = TradingDataSimulator(
        data_source=data_source,
        update_interval=custom_settings['update_interval'],
        news_generator=news_generator,
    )

    # Create LLM client
    llm_client = OllamaClient(
        model=custom_settings['model'],
        temperature=custom_settings['temperature'],
        max_tokens=custom_settings['max_tokens'],
    )

    # Create prompt manager
    prompt_manager = PromptManager(
        system_prompt=custom_settings['system_prompt'],
    )

    # Create memory manager
    memory_type = custom_settings.get('memory_type', 'sliding_window')
    memory_manager = None

    if memory_type == 'chromadb':
        memory_config = config.get('memory', {}).get('chromadb', {})
        memory_manager = ChromaMemoryManager(memory_config)
    elif memory_type == 'sliding_window':
        memory_config = config.get('memory', {}).get('sliding_window', {})
        memory_manager = SlidingWindowMemoryManager(memory_config)
    # else: memory_manager stays None (no memory management)

    # Create portfolio manager if trading is enabled
    portfolio_manager = None
    if custom_settings.get('enable_trading', True):
        portfolio_manager = PortfolioManager(
            symbols=custom_settings['symbols'],
            initial_cash_per_symbol=custom_settings.get('initial_cash_per_symbol', 1000.0)
        )

    # Create strategy
    strategy_type = custom_settings['strategy']
    if strategy_type == 'continuous':
        strategy = ContinuousStrategy(
            llm_client,
            prompt_manager,
            {'batch_size': custom_settings['batch_size'], 'memory': config.get('memory', {})},
            memory_manager=memory_manager
        )
    elif strategy_type == 'reactive':
        reactive_config = config['strategy'].get('reactive', {})
        reactive_config['memory'] = config.get('memory', {})
        reactive_config['enable_trading'] = custom_settings.get('enable_trading', True)
        strategy = ReactiveStrategy(
            llm_client,
            prompt_manager,
            reactive_config,
            memory_manager=memory_manager,
            portfolio_manager=portfolio_manager
        )
    else:  # event_driven
        strategy = EventDrivenStrategy(
            llm_client,
            prompt_manager,
            config['strategy'].get('event_driven', {})
        )

    return simulator, strategy, portfolio_manager


def process_stream(simulator, strategy, state_container):
    """
    Process the data stream in the background.

    Note: We pass objects directly instead of accessing session_state
    because session_state is not thread-safe.
    """
    try:
        for data_point in simulator.stream():
            if not state_container['running']:
                break

            # Update current data
            state_container['current_data'] = data_point
            state_container['iteration'] += 1

            # Add to history (keep last 100)
            state_container['data_history'].append(data_point)
            if len(state_container['data_history']) > 100:
                state_container['data_history'].pop(0)

            # Process with strategy
            response = strategy.process_data_point(data_point)

            if response:
                state_container['response_history'].append({
                    'iteration': state_container['iteration'],
                    'timestamp': datetime.now().isoformat(),
                    'data': data_point,
                    'response': response,
                })

        simulator.stop()
        state_container['running'] = False
    except Exception as e:
        logger.error(f"Error in process_stream: {e}")
        state_container['running'] = False
        simulator.stop()


# Header
st.markdown('<div class="main-header">Continuous Prompting Framework</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Load default config
    config = load_config()
    
    st.subheader("LLM Settings")
    model = st.text_input("Model", value=config['llm']['model'])
    temperature = st.slider("Temperature", 0.0, 1.0, config['llm']['temperature'], 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 4096, config['llm']['max_tokens'], 100)
    
    st.subheader("Data Settings")
    data_source = st.selectbox("Data Source", ["sample", "csv"])
    update_interval = st.slider("Update Interval (seconds)", 0.5, 10.0,
                                config['data']['update_interval'], 0.5)

    if data_source == "sample":
        symbols_input = st.text_input("Symbols (comma-separated)", "AAPL,GOOGL,MSFT,TSLA")
        symbols = [s.strip() for s in symbols_input.split(',')]
        volatility = st.slider("Price Volatility", 0.01, 0.10, 0.02, 0.01)
        csv_path = None
    else:
        symbols = []
        volatility = 0.02
        csv_path = st.text_input("CSV Path", "data/raw/trading_data.csv")

    # News injection settings
    st.subheader("News Events")
    news_enabled = st.checkbox("Enable News Injection",
                               value=config['data'].get('news', {}).get('enabled', True),
                               help="Inject generated news events 1-2 times per day")
    if news_enabled:
        news_events_per_day = st.slider("Events Per Day", 0.5, 5.0, 1.5, 0.5,
                                        help="Average number of news events per day")
    else:
        news_events_per_day = 0

    st.subheader("Strategy Settings")
    strategy_type = st.selectbox("Strategy", ["reactive", "continuous", "event_driven"],
                                 help="Reactive: LLM only responds to important events (RECOMMENDED)")

    if strategy_type == "continuous":
        batch_size = st.number_input("Batch Size", 1, 10, 1)
    else:
        batch_size = 1

    st.subheader("Memory Management")
    memory_type = st.selectbox(
        "Memory Type",
        ["sliding_window", "chromadb", "none"],
        help="Choose how to manage LLM context and history"
    )

    if memory_type == "sliding_window":
        st.caption("Keeps recent data + summarizes old data")
        window_size = st.number_input("Window Size", 5, 100, 20,
                                      help="Number of recent items to keep in full detail")
    elif memory_type == "chromadb":
        st.caption("Vector database with semantic search")
        top_k = st.number_input("Top K Retrieval", 1, 20, 5,
                               help="Number of similar items to retrieve")

    st.subheader("Trading Settings")
    enable_trading = st.checkbox("Enable Portfolio Trading",
                                 value=True,
                                 help="Allow LLM to make buy/sell decisions with virtual portfolio")
    if enable_trading:
        initial_cash_per_symbol = st.number_input("Initial Cash Per Symbol", 100, 10000, 1000, 100,
                                                  help="Starting cash allocation per stock (default: $1000)")
    else:
        initial_cash_per_symbol = 1000

    st.subheader("Prompt Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value=config['prompts']['system_prompt'],
        height=150
    )

    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start", disabled=st.session_state.running):
            try:
                # Create custom settings
                custom_settings = {
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'data_source': data_source,
                    'update_interval': update_interval,
                    'symbols': symbols,
                    'volatility': volatility,
                    'csv_path': csv_path,
                    'strategy': strategy_type,
                    'batch_size': batch_size,
                    'system_prompt': system_prompt,
                    'memory_type': memory_type,
                    'news_enabled': news_enabled,
                    'news_events_per_day': news_events_per_day,
                    'enable_trading': enable_trading,
                    'initial_cash_per_symbol': initial_cash_per_symbol,
                }

                # Create components
                simulator, strategy, portfolio_manager = create_components(config, custom_settings)

                # Initialize session state
                st.session_state.simulator = simulator
                st.session_state.strategy = strategy
                st.session_state.portfolio_manager = portfolio_manager
                st.session_state.running = True
                st.session_state.start_time = datetime.now()
                st.session_state.iteration = 0
                st.session_state.data_history = []
                st.session_state.response_history = []
                st.session_state.current_data = None

                # Create a shared state dict that the thread can safely modify
                if 'shared_state' not in st.session_state:
                    st.session_state.shared_state = {}

                st.session_state.shared_state = {
                    'running': True,
                    'current_data': None,
                    'iteration': 0,
                    'data_history': [],
                    'response_history': []
                }

                # Start processing in background thread
                thread = threading.Thread(
                    target=process_stream,
                    args=(simulator, strategy, st.session_state.shared_state),
                    daemon=True
                )
                thread.start()
                st.session_state.thread = thread

                st.success("Started!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col2:
        if st.button("Stop", disabled=not st.session_state.running):
            st.session_state.running = False
            if 'shared_state' in st.session_state:
                st.session_state.shared_state['running'] = False
            if 'simulator' in st.session_state and st.session_state.simulator:
                st.session_state.simulator.stop()
            st.success("Stopped!")
            st.rerun()

    if st.button("Reset"):
        st.session_state.running = False
        st.session_state.data_history = []
        st.session_state.response_history = []
        st.session_state.current_data = None
        st.session_state.iteration = 0
        if 'shared_state' in st.session_state:
            st.session_state.shared_state = {
                'running': False,
                'current_data': None,
                'iteration': 0,
                'data_history': [],
                'response_history': []
            }
        if 'simulator' in st.session_state and st.session_state.simulator:
            st.session_state.simulator.stop()
        st.success("Reset!")
        st.rerun()

# Sync shared state to session state for display
if 'shared_state' in st.session_state and st.session_state.shared_state:
    st.session_state.current_data = st.session_state.shared_state.get('current_data')
    st.session_state.iteration = st.session_state.shared_state.get('iteration', 0)
    st.session_state.data_history = st.session_state.shared_state.get('data_history', [])
    st.session_state.response_history = st.session_state.shared_state.get('response_history', [])
    st.session_state.running = st.session_state.shared_state.get('running', False)

# Main content area
if st.session_state.running:
    status_text = "**Running**"
    status_color = "green"
else:
    status_text = "**Stopped**"
    status_color = "red"

st.markdown(f"**Status:** {status_text}")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Data Points", st.session_state.iteration)

with col2:
    st.metric("LLM Responses", len(st.session_state.response_history))

with col3:
    if st.session_state.start_time and st.session_state.running:
        elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
        st.metric("Elapsed Time", f"{elapsed:.1f}s")
    else:
        st.metric("Elapsed Time", "0.0s")

with col4:
    if st.session_state.iteration > 0:
        response_rate = len(st.session_state.response_history) / st.session_state.iteration * 100
        st.metric("Response Rate", f"{response_rate:.1f}%")
    else:
        st.metric("Response Rate", "0%")

st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Data", "LLM Responses", "Portfolio", "Charts", "History"])

with tab1:
    st.subheader("Current Data Point")

    if st.session_state.current_data:
        data = st.session_state.current_data

        # Check if this is a news event
        if data.get('type') == 'news':
            # Display news event
            sentiment = data.get('sentiment', 'neutral')
            news_class = f"news-box news-{sentiment}"

            st.markdown(f"""
            <div class="{news_class}">
            <h3>NEWS EVENT</h3>
            <strong>Symbol:</strong> {data.get('symbol', 'N/A')}<br>
            <strong>Headline:</strong> {data.get('headline', 'No headline')}<br>
            <strong>Sentiment:</strong> {sentiment.upper()}<br>
            <strong>Impact:</strong> {data.get('impact', 'unknown').upper()}<br>
            <strong>Category:</strong> {data.get('category', 'unknown')}<br>
            <strong>Timestamp:</strong> {data.get('timestamp', 'N/A')[:19]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display regular trading data
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div class="data-box">
                <strong>Symbol:</strong> {data.get('symbol', 'N/A')}<br>
                <strong>Price:</strong> ${data.get('price', 0):.2f}<br>
                <strong>Change:</strong> {data.get('change', 0):+.2f}%<br>
                <strong>Volume:</strong> {data.get('volume', 0):,}<br>
                <strong>Timestamp:</strong> {data.get('timestamp', 'N/A')[:19]}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Price indicator
                change = data.get('change', 0)
                if change >= 0:
                    st.success(f"Price UP {change:.2f}%")
                else:
                    st.error(f"Price DOWN {change:.2f}%")
    else:
        st.info("Waiting for data...")

with tab2:
    st.subheader("LLM Responses")
    
    if st.session_state.response_history:
        # Show most recent responses first
        for resp in reversed(st.session_state.response_history[-10:]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Response #{resp['iteration']}**")
                with col2:
                    st.caption(resp['timestamp'][:19])
                
                data = resp['data']
                st.caption(f"{data['symbol']}: ${data['price']:.2f} ({data['change']:+.2f}%)")
                
                st.markdown(f"""
                <div class="response-box">
                {resp['response']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
    else:
        st.info("⏳ No responses yet. Waiting for LLM output...")

with tab3:
    st.subheader("Portfolio Performance")

    if st.session_state.portfolio_manager:
        portfolio = st.session_state.portfolio_manager
        summary = portfolio.get_summary()

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolio Value", f"${summary['total_value']:.2f}",
                     delta=f"{summary['total_return_pct']:.2f}%")

        with col2:
            st.metric("Cash Available", f"${summary['cash']:.2f}")

        with col3:
            st.metric("Buy & Hold Value", f"${summary['buy_hold_value']:.2f}",
                     delta=f"{summary['buy_hold_return_pct']:.2f}%")

        with col4:
            outperf = summary['outperformance']
            st.metric("Outperformance", f"{outperf:.2f}%",
                     delta=f"{'Beating' if outperf > 0 else 'Trailing'} market")

        st.markdown("---")

        # Trading statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Trades", summary['total_trades'])

        with col2:
            st.metric("Winning Trades", summary['winning_trades'])

        with col3:
            st.metric("Win Rate", f"{summary['win_rate']:.1f}%")

        st.markdown("---")

        # Current positions
        st.subheader("Current Positions")

        if summary['positions']:
            for pos in summary['positions']:
                with st.expander(f"{pos['symbol']} - {pos['shares']:.4f} shares"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Average Cost:** ${pos['avg_cost']:.2f}")
                        st.write(f"**Current Price:** ${pos['current_price']:.2f}")
                        st.write(f"**Position Value:** ${pos['value']:.2f}")

                    with col2:
                        profit_loss = pos['profit_loss']
                        profit_loss_pct = pos['profit_loss_pct']

                        if profit_loss >= 0:
                            st.success(f"**Profit:** ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                        else:
                            st.error(f"**Loss:** ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
        else:
            st.info("No open positions")

        # Recent trades
        if portfolio.trades:
            st.markdown("---")
            st.subheader("Recent Trades (Last 10)")

            recent_trades = portfolio.trades[-10:]
            for trade in reversed(recent_trades):
                trade_dict = trade.to_dict()
                action_color = "🟢" if trade_dict['action'] == 'buy' else "🔴"
                st.caption(
                    f"{action_color} {trade_dict['action'].upper()} {trade_dict['shares']:.4f} shares of "
                    f"{trade_dict['symbol']} @ ${trade_dict['price']:.2f} = ${trade_dict['total']:.2f} "
                    f"({trade_dict['timestamp'][:19]})"
                )
    else:
        st.info("Portfolio trading is disabled. Enable it in the sidebar to track performance.")

with tab4:
    st.subheader("Price Charts")

    if len(st.session_state.data_history) > 1:
        # Create DataFrame from history
        df = pd.DataFrame(st.session_state.data_history)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Define fixed symbol order and colors
        # Get all unique symbols and sort them alphabetically for consistency
        all_symbols = sorted(df['symbol'].unique())

        # Define a fixed color palette
        color_palette = {
            0: '#1f77b4',  # Blue
            1: '#ff7f0e',  # Orange
            2: '#2ca02c',  # Green
            3: '#d62728',  # Red
            4: '#9467bd',  # Purple
            5: '#8c564b',  # Brown
            6: '#e377c2',  # Pink
            7: '#7f7f7f',  # Gray
        }

        # Assign fixed colors to symbols
        symbol_colors = {symbol: color_palette[i % len(color_palette)]
                        for i, symbol in enumerate(all_symbols)}

        # Create subplots with fixed order
        fig = make_subplots(
            rows=len(all_symbols), cols=1,
            subplot_titles=[f"{symbol} Price Movement" for symbol in all_symbols],
            vertical_spacing=0.1
        )

        # Add traces in fixed order
        for idx, symbol in enumerate(all_symbols, 1):
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')

            if len(symbol_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['timestamp'],
                        y=symbol_data['price'],
                        mode='lines+markers',
                        name=symbol,
                        line=dict(width=2, color=symbol_colors[symbol]),
                        marker=dict(size=4, color=symbol_colors[symbol]),
                    ),
                    row=idx, col=1
                )

        fig.update_layout(height=300*len(all_symbols), showlegend=True)
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Price ($)")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Collecting data for charts...")

with tab4:
    st.subheader("Data History")
    
    if st.session_state.data_history:
        df = pd.DataFrame(st.session_state.data_history)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("⏳ No data history yet...")

# Auto-refresh when running
if st.session_state.running:
    time.sleep(0.5)
    st.rerun()

