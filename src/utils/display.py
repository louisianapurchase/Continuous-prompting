"""Display utilities for human-friendly terminal output."""

import sys
from typing import Dict, Any, Optional
from datetime import datetime


class TerminalDisplay:
    """
    Manages clean, human-friendly terminal output for continuous prompting.
    
    Provides a live dashboard that shows:
    - Current data stream status
    - Latest data point
    - LLM responses (non-spammy)
    - Statistics
    """
    
    def __init__(self, show_data_stream: bool = True):
        """
        Initialize terminal display.
        
        Args:
            show_data_stream: Whether to show every data point or just LLM responses
        """
        self.show_data_stream = show_data_stream
        self.data_count = 0
        self.llm_response_count = 0
        self.start_time = datetime.now()
        self.last_data = None
        
    def clear_line(self):
        """Clear the current line."""
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
    
    def print_header(self, strategy_name: str, model_name: str, update_interval: float):
        """Print the application header."""
        print("\n" + "="*80)
        print("CONTINUOUS PROMPTING FRAMEWORK".center(80))
        print("="*80)
        print(f"Strategy: {strategy_name}".center(80))
        print(f"Model: {model_name} | Update Interval: {update_interval}s".center(80))
        print("="*80)
        print("\nPress Ctrl+C to stop gracefully\n")
    
    def print_data_point(self, data: Dict[str, Any], iteration: int):
        """
        Print a data point in a compact, non-spammy way.

        Args:
            data: Data point dictionary
            iteration: Current iteration number
        """
        self.data_count += 1
        self.last_data = data

        if self.show_data_stream:
            # Check if this is a news event
            if data.get('type') == 'news':
                # News events get special formatting
                self._print_news_event(data, iteration)
            else:
                # Regular trading data
                # Compact one-line format
                symbol = data.get('symbol', 'N/A')
                price = data.get('price', 0)
                change = data.get('change', 0)
                volume = data.get('volume', 0)
                timestamp = data.get('timestamp', '')[:19]  # Trim to readable format

                # Color code the change
                change_color = '\033[92m' if change >= 0 else '\033[91m'  # Green/Red
                reset_color = '\033[0m'

                status = (
                    f"[{iteration:4d}] {timestamp} | "
                    f"{symbol:6s} ${price:7.2f} "
                    f"{change_color}{change:+6.2f}%{reset_color} | "
                    f"Vol: {volume:,}"
                )

                # Overwrite the same line for continuous updates
                self.clear_line()
                sys.stdout.write(status)
                sys.stdout.flush()

    def _print_news_event(self, data: Dict[str, Any], iteration: int):
        """Print a news event with special formatting."""
        self.clear_line()
        print()  # New line

        # Color based on sentiment
        sentiment = data.get('sentiment', 'neutral')
        if sentiment == 'positive':
            color = '\033[92m'  # Green
        elif sentiment == 'negative':
            color = '\033[91m'  # Red
        else:
            color = '\033[93m'  # Yellow
        reset = '\033[0m'

        impact = data.get('impact', 'unknown').upper()
        symbol = data.get('symbol', 'N/A')
        headline = data.get('headline', 'No headline')
        timestamp = data.get('timestamp', '')[:19]

        print(f"\n{'='*80}")
        print(f"{color}NEWS EVENT [{impact} IMPACT]{reset}".center(90))
        print(f"{'='*80}")
        print(f"[{iteration:4d}] {timestamp}")
        print(f"Symbol: {symbol}")
        print(f"Sentiment: {color}{sentiment.upper()}{reset}")
        print(f"Headline: {headline}")
        print(f"{'='*80}\n")
    
    def print_llm_response(
        self,
        response: str,
        data: Dict[str, Any],
        iteration: int,
        response_number: int,
    ):
        """
        Print an LLM response in a clean, formatted way.
        
        Args:
            response: LLM response text
            data: Associated data point
            iteration: Current iteration number
            response_number: Number of this response
        """
        self.llm_response_count += 1
        
        # Clear the data stream line if it exists
        if self.show_data_stream:
            print()  # New line after data stream
        
        # Print response header
        print("\n" + "┌" + "─"*78 + "┐")
        print(f"│ LLM RESPONSE #{response_number:<3} (at iteration {iteration})".ljust(79) + "│")
        print("├" + "─"*78 + "┤")

        # Print associated data
        symbol = data.get('symbol', 'N/A')
        price = data.get('price', 0)
        change = data.get('change', 0)
        volume = data.get('volume', 0)

        change_symbol = "UP" if change >= 0 else "DN"
        print(f"│ {change_symbol} {symbol}: ${price:.2f} ({change:+.2f}%) | Volume: {volume:,}".ljust(79) + "│")
        print("├" + "─"*78 + "┤")
        
        # Print response (word-wrapped)
        response_lines = self._wrap_text(response, 76)
        for line in response_lines:
            print(f"│ {line.ljust(77)}│")
        
        print("└" + "─"*78 + "┘\n")
    
    def print_status_update(self, iteration: int, responses: int):
        """
        Print a periodic status update.
        
        Args:
            iteration: Current iteration
            responses: Number of LLM responses so far
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = iteration / elapsed if elapsed > 0 else 0

        print(f"\nStatus: {iteration} data points | {responses} LLM responses | "
              f"{rate:.1f} points/sec")
    
    def print_summary(self, stats: Dict[str, Any]):
        """
        Print final summary statistics.
        
        Args:
            stats: Statistics dictionary
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()

        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY".center(80))
        print("="*80)
        print(f"  Strategy: {stats.get('strategy', 'N/A')}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Data Points Processed: {stats.get('data_points_processed', 0)}")
        print(f"  LLM Responses Generated: {stats.get('responses_generated', 0)}")
        
        if stats.get('data_points_processed', 0) > 0:
            response_rate = (stats.get('responses_generated', 0) / 
                           stats.get('data_points_processed', 1) * 100)
            print(f"  Response Rate: {response_rate:.1f}%")
        
        print("="*80 + "\n")
    
    def print_error(self, error: str):
        """
        Print an error message.

        Args:
            error: Error message
        """
        print(f"\nERROR: {error}\n")

    def print_warning(self, warning: str):
        """
        Print a warning message.

        Args:
            warning: Warning message
        """
        print(f"\nWARNING: {warning}\n")

    def print_info(self, info: str):
        """
        Print an info message.

        Args:
            info: Info message
        """
        print(f"INFO: {info}")
    
    @staticmethod
    def _wrap_text(text: str, width: int) -> list:
        """
        Wrap text to specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            if current_length + word_length + len(current_line) <= width:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else ['']


class CompactDisplay:
    """
    Ultra-compact display that only shows LLM responses.
    
    Perfect for when you only care about the LLM output, not the data stream.
    """
    
    def __init__(self):
        """Initialize compact display."""
        self.response_count = 0
        self.data_count = 0
    
    def print_header(self, strategy_name: str, model_name: str, update_interval: float):
        """Print compact header."""
        print(f"\n{model_name} | {strategy_name} | {update_interval}s intervals")
        print("─"*60)
    
    def print_data_point(self, data: Dict[str, Any], iteration: int):
        """Silently count data points."""
        self.data_count += 1
    
    def print_llm_response(
        self,
        response: str,
        data: Dict[str, Any],
        iteration: int,
        response_number: int,
    ):
        """Print response in compact format."""
        self.response_count += 1
        
        symbol = data.get('symbol', 'N/A')
        price = data.get('price', 0)
        change = data.get('change', 0)

        print(f"\n[{response_number}] {symbol} ${price:.2f} ({change:+.2f}%)")
        print(f"Response: {response}")
        print("─"*60)
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print compact summary."""
        print(f"\nDone: {stats.get('data_points_processed', 0)} points, "
              f"{stats.get('responses_generated', 0)} responses")

