"""
Main entry point for the Continuous Prompting Framework.

This script orchestrates the data streaming and LLM interaction based on
the configured strategy.
"""

import argparse
import yaml
import logging
from pathlib import Path
import signal
import sys

from src.data import TradingDataSimulator, SampleDataSource, CSVDataSource
from src.data.news_generator import NewsGenerator
from src.llm import OllamaClient, PromptManager
from src.strategies import ContinuousStrategy, EventDrivenStrategy
from src.strategies.reactive_strategy import ReactiveStrategy
from src.memory import SlidingWindowMemoryManager, ChromaMemoryManager
from src.portfolio import PortfolioManager
from src.utils import setup_logger, MetricsTracker, TerminalDisplay, CompactDisplay


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    print("\n\nShutdown requested. Cleaning up...")
    shutdown_requested = True


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_data_source(config: dict):
    """
    Create data source based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataSource instance
    """
    source_type = config['data']['source']
    
    if source_type == 'sample':
        sample_config = config['data'].get('sample', {})
        return SampleDataSource(
            symbols=sample_config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA']),
            price_volatility=sample_config.get('price_volatility', 0.02),
            volume_range=tuple(sample_config.get('volume_range', [1000, 10000])),
        )
    elif source_type == 'csv':
        csv_path = config['data'].get('csv_path', 'data/raw/trading_data.csv')
        return CSVDataSource(csv_path)
    else:
        raise ValueError(f"Unknown data source type: {source_type}")


def create_memory_manager(config: dict):
    """
    Create memory manager based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Memory manager instance or None
    """
    memory_config = config.get('memory', {})
    memory_type = memory_config.get('type', 'sliding_window')

    if memory_type == 'none':
        return None
    elif memory_type == 'sliding_window':
        return SlidingWindowMemoryManager(memory_config.get('sliding_window', {}))
    elif memory_type == 'chromadb':
        return ChromaMemoryManager(memory_config.get('chromadb', {}))
    else:
        logging.warning(f"Unknown memory type: {memory_type}, using sliding window")
        return SlidingWindowMemoryManager({})


def create_strategy(config: dict, llm_client, prompt_manager, portfolio_manager=None):
    """
    Create prompting strategy based on configuration.

    Args:
        config: Configuration dictionary
        llm_client: LLM client instance
        prompt_manager: Prompt manager instance
        portfolio_manager: Portfolio manager instance (optional)

    Returns:
        Strategy instance
    """
    strategy_type = config['strategy']['type']
    memory_manager = create_memory_manager(config)

    if strategy_type == 'continuous':
        strategy_config = config['strategy'].get('continuous', {})
        strategy_config['memory'] = config.get('memory', {})
        return ContinuousStrategy(llm_client, prompt_manager, strategy_config, memory_manager)
    elif strategy_type == 'event_driven':
        strategy_config = config['strategy'].get('event_driven', {})
        strategy_config['memory'] = config.get('memory', {})
        return EventDrivenStrategy(llm_client, prompt_manager, strategy_config, memory_manager)
    elif strategy_type == 'reactive':
        strategy_config = config['strategy'].get('reactive', {})
        strategy_config['memory'] = config.get('memory', {})
        return ReactiveStrategy(llm_client, prompt_manager, strategy_config, memory_manager, portfolio_manager)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def main():
    """Main application entry point."""
    global shutdown_requested
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Continuous Prompting Framework for LLM Experimentation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum number of data points to process (default: unlimited)'
    )
    parser.add_argument(
        '--display',
        type=str,
        choices=['full', 'compact', 'minimal'],
        default='full',
        help='Display mode: full (show data stream), compact (responses only), minimal (quiet)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = 'WARNING' if args.display == 'minimal' else config['logging']['level']
    logger = setup_logger(
        level=log_level,
        log_to_file=True,
    )

    # Setup display
    if args.display == 'compact':
        display = CompactDisplay()
    elif args.display == 'minimal':
        display = None
    else:  # full
        display = TerminalDisplay(show_data_stream=True)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Create data source
        data_source = create_data_source(config)

        # Create news generator if enabled
        news_config = config['data'].get('news', {})
        news_generator = None
        if news_config.get('enabled', False):
            sample_config = config['data'].get('sample', {})
            symbols = sample_config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
            news_generator = NewsGenerator(
                symbols=symbols,
                events_per_day=news_config.get('events_per_day', 1.5),
                enabled=True
            )
            logger.info("News generation enabled")

        # Create data simulator
        simulator = TradingDataSimulator(
            data_source=data_source,
            update_interval=config['data']['update_interval'],
            news_generator=news_generator,
        )
        
        # Create LLM client
        llm_config = config['llm']
        llm_client = OllamaClient(
            model=llm_config['model'],
            base_url=llm_config.get('base_url', 'http://localhost:11434'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 2048),
            num_gpu=llm_config.get('num_gpu', 0),
        )
        
        # Create prompt manager
        prompts = config.get('prompts', {})
        prompt_manager = PromptManager(
            system_prompt=prompts.get('system_prompt'),
            continuous_prompt_template=prompts.get('continuous_prompt'),
            event_prompt_template=prompts.get('event_prompt'),
        )

        # Create portfolio manager if trading is enabled
        portfolio_manager = None
        trading_config = config.get('trading', {})
        if trading_config.get('enabled', True):
            sample_config = config['data'].get('sample', {})
            symbols = sample_config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
            initial_cash_per_symbol = trading_config.get('initial_cash_per_symbol', 1000.0)
            portfolio_manager = PortfolioManager(
                symbols=symbols,
                initial_cash_per_symbol=initial_cash_per_symbol
            )
            total_cash = initial_cash_per_symbol * len(symbols)
            logger.info(f"Portfolio trading enabled: ${initial_cash_per_symbol:.2f} per stock (${total_cash:.2f} total)")

        # Create strategy
        strategy = create_strategy(config, llm_client, prompt_manager, portfolio_manager)
        
        # Create metrics tracker
        experiment_config = config.get('experiment', {})
        metrics = MetricsTracker(
            experiment_name=experiment_config.get('name', 'default_experiment'),
            save_dir=config['logging'].get('metrics_dir', 'logs/metrics'),
        )
        
        logger.info("All components initialized successfully!")

        # Print header with display
        if display:
            display.print_header(
                strategy_name=strategy.__class__.__name__,
                model_name=llm_config['model'],
                update_interval=config['data']['update_interval'],
            )
        else:
            logger.info(f"Strategy: {strategy.__class__.__name__}")
            logger.info(f"LLM Model: {llm_config['model']}")
            logger.info(f"Data update interval: {config['data']['update_interval']}s")
            logger.info("Starting data stream...")
        
        # Main processing loop
        iteration = 0
        for data_point in simulator.stream():
            if shutdown_requested:
                break

            # Check max iterations
            if args.max_iterations and iteration >= args.max_iterations:
                if display:
                    print()  # New line after data stream
                logger.info(f"Reached max iterations: {args.max_iterations}")
                break

            # Display data point
            if display:
                display.print_data_point(data_point, iteration)

            # Log data point (debug level)
            logger.debug(f"Data point {iteration}: {data_point}")

            # Process with strategy
            response = strategy.process_data_point(data_point)

            # Record metrics
            metrics.record_data_point()

            # Handle response
            if response:
                if display:
                    display.print_llm_response(
                        response=response,
                        data=data_point,
                        iteration=iteration,
                        response_number=len(strategy.response_history),
                    )
                else:
                    # Fallback to simple print
                    print(f"\n[Response {len(strategy.response_history)}] {response}\n")

                metrics.record_llm_call(
                    prompt="",  # Strategy handles prompt internally
                    response=response,
                    data=data_point,
                )

            # Periodic status update (every 50 iterations in full mode)
            if display and isinstance(display, TerminalDisplay) and iteration > 0 and iteration % 50 == 0:
                display.print_status_update(iteration, len(strategy.response_history))

            iteration += 1
        
        # Cleanup
        logger.info("\nStopping data stream...")
        simulator.stop()

        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("Experiment Statistics")
        logger.info("="*60)
        stats = strategy.get_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        # Print portfolio summary if trading was enabled
        if portfolio_manager:
            logger.info("\n" + "="*60)
            logger.info("Portfolio Performance")
            logger.info("="*60)
            summary = portfolio_manager.get_summary()
            logger.info(f"Initial Value: ${portfolio_manager.total_initial_cash:.2f}")
            logger.info(f"Final Value: ${summary['total_value']:.2f}")
            logger.info(f"Cash Remaining: ${summary['cash']:.2f}")
            logger.info(f"Portfolio Return: {summary['total_return_pct']:.2f}%")
            logger.info(f"Buy & Hold Return: {summary['buy_hold_return_pct']:.2f}%")
            logger.info(f"Outperformance: {summary['outperformance']:.2f}%")
            logger.info(f"Total Trades: {summary['total_trades']}")
            logger.info(f"Winning Trades: {summary['winning_trades']}")
            logger.info(f"Losing Trades: {summary['losing_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']:.1f}%")

            if summary['positions']:
                logger.info("\nOpen Positions:")
                for pos in summary['positions']:
                    logger.info(f"  {pos['symbol']}: {pos['shares']:.4f} shares @ ${pos['avg_cost']:.2f} "
                              f"(Current: ${pos['current_price']:.2f}, P/L: ${pos['profit_loss']:.2f})")

        # Save metrics
        if config['logging'].get('save_metrics', True):
            metrics.save()
            metrics.print_summary()

        logger.info("="*60)
        logger.info("Experiment completed successfully!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

