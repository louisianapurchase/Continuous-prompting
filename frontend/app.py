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

from src.data import TradingDataSimulator, SampleDataSource, CSVDataSource
from src.data.news_generator import NewsGenerator
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
simulator_thread = None
event_queue = Queue()
is_running = False
portfolio_manager = None
strategy = None
data_history = []
response_history = []
executor = ThreadPoolExecutor(max_workers=2)  # For async LLM processing


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()


def create_components(settings):
    """Create all necessary components based on settings."""
    global portfolio_manager, strategy

    # Create data source
    if settings['data_source'] == 'sample':
        data_source = SampleDataSource(
            symbols=settings['symbols'],
            price_volatility=settings['volatility'],
        )
    else:
        data_source = CSVDataSource(
            csv_path=settings['csv_path'],
            symbols=settings['symbols']
        )

    # Create news generator
    news_generator = None
    if settings.get('enable_news', True):
        news_generator = NewsGenerator(
            symbols=settings['symbols'],
            events_per_day=settings.get('news_events_per_day', 1.5)
        )

    # Create simulator
    simulator = TradingDataSimulator(
        data_source=data_source,
        update_interval=settings['update_interval'],
        news_generator=news_generator,
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
    
    return simulator, strategy


def process_stream(simulator, strategy):
    """Process data stream and send events to clients."""
    global is_running, data_history, response_history

    try:
        for data_point in simulator.stream():
            if not is_running:
                logger.info("Stream stopped by user")
                break

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

                    # Send data update to clients
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
                                    'timestamp': datetime.now().isoformat(),
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
    global simulator_thread, is_running, data_history, response_history

    if is_running:
        return jsonify({'error': 'Simulation already running'}), 400

    # Get settings from request
    settings = request.json or {}

    # Apply defaults from config
    settings.setdefault('data_source', 'sample')
    settings.setdefault('symbols', config['data']['sample']['symbols'])
    settings.setdefault('volatility', config['data']['sample']['price_volatility'])
    settings.setdefault('update_interval', config['data']['update_interval'])
    settings.setdefault('model', config['llm']['model'])
    settings.setdefault('temperature', config['llm']['temperature'])
    settings.setdefault('max_tokens', config['llm']['max_tokens'])
    settings.setdefault('strategy', config['strategy']['type'])
    settings.setdefault('enable_trading', config['trading']['enabled'])
    settings.setdefault('initial_cash_per_symbol', config['trading']['initial_cash_per_symbol'])
    settings.setdefault('enable_news', config['data']['news']['enabled'])
    settings.setdefault('news_events_per_day', config['data']['news']['events_per_day'])

    # Clear history
    data_history.clear()
    response_history.clear()

    # Create components
    simulator, strat = create_components(settings)

    # Start simulation thread
    is_running = True
    simulator_thread = Thread(target=process_stream, args=(simulator, strat))
    simulator_thread.daemon = True
    simulator_thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation."""
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset the simulation."""
    global is_running, data_history, response_history, portfolio_manager

    is_running = False
    data_history.clear()
    response_history.clear()
    portfolio_manager = None

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


