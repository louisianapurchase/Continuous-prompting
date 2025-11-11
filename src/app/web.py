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
from src.llm import OllamaClient, PromptManager
from src.strategies import ContinuousStrategy, EventDrivenStrategy

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
    }
    .data-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9rem;
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


def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def create_components(config, custom_settings):
    """Create data source, simulator, LLM client, and strategy."""
    # Create data source
    if custom_settings['data_source'] == 'sample':
        data_source = SampleDataSource(
            symbols=custom_settings['symbols'],
            price_volatility=custom_settings['volatility'],
        )
    else:
        data_source = CSVDataSource(custom_settings['csv_path'])
    
    # Create simulator
    simulator = TradingDataSimulator(
        data_source=data_source,
        update_interval=custom_settings['update_interval'],
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
    
    # Create strategy
    if custom_settings['strategy'] == 'continuous':
        strategy = ContinuousStrategy(
            llm_client,
            prompt_manager,
            {'batch_size': custom_settings['batch_size']}
        )
    else:
        strategy = EventDrivenStrategy(
            llm_client,
            prompt_manager,
            config['strategy'].get('event_driven', {})
        )
    
    return simulator, strategy


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
    
    st.subheader("Strategy Settings")
    strategy_type = st.selectbox("Strategy", ["continuous", "event_driven"])
    
    if strategy_type == "continuous":
        batch_size = st.number_input("Batch Size", 1, 10, 1)
    else:
        batch_size = 1
    
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
                }
                
                # Create components
                simulator, strategy = create_components(config, custom_settings)

                # Initialize session state
                st.session_state.simulator = simulator
                st.session_state.strategy = strategy
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
tab1, tab2, tab3, tab4 = st.tabs(["Live Data", "LLM Responses", "Charts", "History"])

with tab1:
    st.subheader("Current Data Point")
    
    if st.session_state.current_data:
        data = st.session_state.current_data
        
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
    st.subheader("Price Charts")
    
    if len(st.session_state.data_history) > 1:
        # Create DataFrame from history
        df = pd.DataFrame(st.session_state.data_history)
        
        # Group by symbol
        symbols = df['symbol'].unique()
        
        # Create subplots
        fig = make_subplots(
            rows=len(symbols), cols=1,
            subplot_titles=[f"{symbol} Price Movement" for symbol in symbols],
            vertical_spacing=0.1
        )
        
        for idx, symbol in enumerate(symbols, 1):
            symbol_data = df[df['symbol'] == symbol]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(symbol_data))),
                    y=symbol_data['price'],
                    mode='lines+markers',
                    name=symbol,
                    line=dict(width=2),
                ),
                row=idx, col=1
            )
        
        fig.update_layout(height=300*len(symbols), showlegend=True)
        fig.update_xaxes(title_text="Data Point")
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

