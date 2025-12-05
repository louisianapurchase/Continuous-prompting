"""
Flask Web Application with Real-Time Updates via Server-Sent Events (SSE).

This is the main web interface for the Continuous Prompting Framework.
Run with: python frontend/app.py
"""

import json
import time
import logging
import sys
import os
import yaml
from datetime import datetime
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SampleDataSource, CSVDataSource
from src.data import ensure_data_available, LiveDataUpdater, is_market_hours
from src.llm import OllamaClient, PromptManager
from src.strategies.reactive_strategy import ReactiveStrategy
from src.memory import SlidingWindowMemoryManager, ChromaMemoryManager
from src.portfolio import PortfolioManager

logger = logging.getLogger(__name__)

# Set template folder explicitly to handle different import methods
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = 'continuous-prompting-secret-key'

# Global state
stream_thread = None
event_queue = Queue()
is_running = False
portfolio_manager = None
strategy = None
data_source = None
data_history = []
response_history = []
executor = ThreadPoolExecutor(max_workers=2)  # For async LLM processing
live_updater = None  # Live data updater for market hours
update_interval = 0.5  # Seconds between data points


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()


def create_components(settings):
    """Create all necessary components based on settings."""
    global portfolio_manager, strategy, live_updater, data_source, update_interval

    # Store update interval
    update_interval = settings['update_interval']

    # Create data source
    if settings['data_source'] == 'sample':
        data_source = SampleDataSource(
            symbols=settings['symbols'],
            price_volatility=settings['volatility'],
        )
    else:
        # Ensure CSV data is available (download if needed)
        logger.info("Checking for CSV data...")
        ensure_data_available(
            symbols=settings['symbols'],
            csv_path=settings['csv_path']
        )

        # Start live updater if during market hours
        if is_market_hours():
            logger.info("Market is open! Starting live data updater...")
            live_updater = LiveDataUpdater(
                symbols=settings['symbols'],
                csv_path=settings['csv_path'],
                update_interval=60  # Update every 60 seconds
            )
            live_updater.start()
        else:
            logger.info("Market is closed. Using existing data.")

        data_source = CSVDataSource(
            csv_path=settings['csv_path'],
            symbols=settings['symbols']
        )

    # Create LLM client
    llm_client = OllamaClient(
        model=settings['model'],
        temperature=settings['temperature'],
        max_tokens=settings['max_tokens'],
        num_gpu=config['llm'].get('num_gpu', 0),
    )

    # Create prompt manager
    prompt_manager = PromptManager(
        system_prompt=config['prompts']['system_prompt'],
    )

    # Create memory manager
    memory_type = settings.get('memory_type', 'sliding_window')
    if memory_type == 'chromadb':
        memory_config = config.get('memory', {}).get('chromadb', {})
        memory_manager = ChromaMemoryManager(memory_config)
    elif memory_type == 'sliding_window':
        memory_config = config.get('memory', {}).get('sliding_window', {})
        memory_manager = SlidingWindowMemoryManager(memory_config)
    else:
        memory_manager = None

    # Create portfolio manager (use global)
    if settings.get('enable_trading', True):
        portfolio_manager = PortfolioManager(
            symbols=settings['symbols'],
            initial_cash_per_symbol=settings.get('initial_cash_per_symbol', 1000.0)
        )
        logger.info(f"Portfolio initialized with ${portfolio_manager.cash:.2f} cash")

    # Create reactive strategy (only strategy supported)
    strategy_config = config.get('strategy', {}).get('reactive', {})

    # Add enable_trading to strategy config
    if settings.get('enable_trading', True):
        strategy_config['enable_trading'] = True

    strategy = ReactiveStrategy(
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        config=strategy_config,
        memory_manager=memory_manager,
        portfolio_manager=portfolio_manager
    )

    return strategy


def process_stream(strategy):
    """Process data stream directly from data source and send events to clients."""
    global is_running, data_history, response_history, data_source, update_interval

    try:
        last_check_date = None

        while is_running:
            # Check if we've entered a new trading day
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo('America/New_York')
            current_date = datetime.now(et_tz).date()

            if last_check_date is not None and current_date > last_check_date:
                logger.info(f"New trading day detected: {current_date}. Reloading data source...")
                # Force reload of data source to pick up new day's data
                data_source.reload_if_modified()
                last_check_date = current_date
            elif last_check_date is None:
                last_check_date = current_date

            # Get next data point from source
            data_point = data_source.get_next()

            if data_point is None:
                # Check if market is open - if so, wait for new data
                if is_market_hours():
                    logger.info("Caught up with live data. Waiting for new data...")
                    time.sleep(10)  # Wait 10 seconds before checking again
                    continue
                else:
                    logger.info("Data source exhausted and market is closed. Waiting for next trading day...")
                    # Wait 5 minutes before checking again (in case market opens)
                    time.sleep(300)
                    continue

            try:
                # Handle batch data
                if data_point.get('type') == 'batch':
                    stocks = data_point.get('stocks', [])

                    # Update portfolio prices
                    if portfolio_manager:
                        for stock_data in stocks:
                            portfolio_manager.update_price(
                                stock_data.get('symbol'),
                                stock_data.get('price')
                            )

                    # Add to history
                    for stock_data in stocks:
                        data_history.append(stock_data)
                        if len(data_history) > 400:
                            data_history.pop(0)

                    # Send data update to clients (use CSV timestamp, not current time!)
                    event_queue.put({
                        'type': 'data',
                        'data': data_point
                    })

                    # Process with strategy (async in background - don't block stream!)
                    def process_strategy_async(data_pt):
                        """Process strategy in background thread."""
                        try:
                            response = strategy.process_data_point(data_pt)
                            if response:
                                response_entry = {
                                    'timestamp': data_pt.get('timestamp'),  # Use data timestamp, not current time
                                    'data': data_pt,
                                    'response': response,
                                }
                                response_history.append(response_entry)

                                # Send response to clients
                                event_queue.put({
                                    'type': 'response',
                                    'data': response_entry
                                })
                        except Exception as e:
                            logger.error(f"Error processing strategy: {e}", exc_info=True)

                    # Submit to thread pool - don't wait for result
                    executor.submit(process_strategy_async, data_point)

                    # Send portfolio update
                    if portfolio_manager:
                        try:
                            summary = portfolio_manager.get_summary()

                            # Debug logging
                            if summary['total_trades'] > 0:
                                logger.info(f"Portfolio: Cash=${summary['cash']:.2f}, Value=${summary['total_value']:.2f}, Positions={len(summary['positions'])}")

                            event_queue.put({
                                'type': 'portfolio',
                                'data': summary
                            })
                        except Exception as e:
                            logger.error(f"Error getting portfolio summary: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Error processing data point: {e}", exc_info=True)
                continue

            # Wait before next data point
            time.sleep(update_interval)

    except Exception as e:
        logger.error(f"Fatal error in stream processing: {e}", exc_info=True)
        event_queue.put({
            'type': 'error',
            'data': str(e)
        })
    finally:
        logger.info("Stream processing ended")
        is_running = False


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start the simulation."""
    global stream_thread, is_running, data_history, response_history

    if is_running:
        return jsonify({'error': 'Simulation already running'}), 400

    # Get settings from request
    settings = request.json or {}

    # Apply defaults from config
    settings.setdefault('data_source', config['data']['source'])
    settings.setdefault('csv_path', config['data'].get('csv_path', 'data/raw/real_trading_data_1m_1d.csv'))
    settings.setdefault('symbols', config['data']['sample']['symbols'])
    settings.setdefault('volatility', config['data']['sample']['price_volatility'])
    settings.setdefault('update_interval', config['data']['update_interval'])
    settings.setdefault('model', config['llm']['model'])
    settings.setdefault('temperature', config['llm']['temperature'])
    settings.setdefault('max_tokens', config['llm']['max_tokens'])
    settings.setdefault('strategy', config['strategy']['type'])
    settings.setdefault('enable_trading', config['trading']['enabled'])
    settings.setdefault('initial_cash_per_symbol', config['trading']['initial_cash_per_symbol'])

    # Clear history
    data_history.clear()
    response_history.clear()

    # Create components
    strat = create_components(settings)

    # Start stream thread
    is_running = True
    stream_thread = Thread(target=process_stream, args=(strat,))
    stream_thread.daemon = True
    stream_thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation."""
    global is_running, live_updater
    is_running = False

    # Stop live updater if running
    if live_updater:
        live_updater.stop()

    return jsonify({'status': 'stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset the simulation."""
    global is_running, data_history, response_history, portfolio_manager, live_updater

    is_running = False
    data_history.clear()
    response_history.clear()
    portfolio_manager = None

    # Stop live updater if running
    if live_updater:
        live_updater.stop()
        live_updater = None

    return jsonify({'status': 'reset'})


@app.route('/api/status')
def get_status():
    """Get current status."""
    return jsonify({
        'running': is_running,
        'data_points': len(data_history),
        'responses': len(response_history),
    })


@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio summary."""
    if portfolio_manager:
        return jsonify(portfolio_manager.get_summary())
    return jsonify({'error': 'Portfolio not initialized'}), 404


@app.route('/stream')
def stream():
    """Server-Sent Events stream for real-time updates."""
    def event_stream():
        try:
            while True:
                try:
                    if not event_queue.empty():
                        event = event_queue.get(timeout=0.1)
                        yield f"data: {json.dumps(event)}\n\n"
                    else:
                        # Send heartbeat to keep connection alive
                        time.sleep(0.5)
                        yield f": heartbeat\n\n"
                except Exception as e:
                    logger.error(f"Error in event stream: {e}")
                    break
        except GeneratorExit:
            logger.info("Client disconnected from stream")

    return Response(event_stream(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)


