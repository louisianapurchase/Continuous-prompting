"""Sliding window memory manager with automatic summarization."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque

from .base_memory import BaseMemoryManager

logger = logging.getLogger(__name__)


class SlidingWindowMemoryManager(BaseMemoryManager):
    """
    Sliding window memory manager with summarization.
    
    Keeps recent data in full detail, summarizes older data to save tokens.
    No external dependencies required.
    
    Strategy:
    - Keep last N data points in full detail (window)
    - Summarize older data into compact statistics
    - Periodically create summaries of batches
    
    Advantages:
    - No external dependencies
    - Predictable token usage
    - Good balance of recent detail + historical context
    
    Best for: Quick experiments, development, no database setup
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sliding window memory manager.
        
        Args:
            config: Configuration with options:
                - window_size: Number of recent items to keep in full (default: 20)
                - summary_batch_size: Items per summary batch (default: 10)
                - max_summaries: Maximum number of summaries to keep (default: 10)
                - enable_llm_summary: Use LLM for summaries (default: False)
        """
        super().__init__(config)
        
        # Configuration
        self.window_size = self.config.get('window_size', 20)
        self.summary_batch_size = self.config.get('summary_batch_size', 10)
        self.max_summaries = self.config.get('max_summaries', 10)
        self.enable_llm_summary = self.config.get('enable_llm_summary', False)
        
        # Storage
        self.recent_window = deque(maxlen=self.window_size)
        self.summaries = deque(maxlen=self.max_summaries)
        self.pending_for_summary = []
        
        # Statistics
        self.total_items_processed = 0
        
        logger.info(
            f"Initialized SlidingWindowMemoryManager: "
            f"window={self.window_size}, batch={self.summary_batch_size}"
        )
    
    def add_data_point(self, data: Dict[str, Any], response: Optional[str] = None) -> None:
        """
        Add data point to sliding window.
        
        Args:
            data: Data point to store
            response: LLM response to store
        """
        item = {
            'data': data,
            'response': response,
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        }
        
        # If window is full, move oldest item to pending summary
        if len(self.recent_window) >= self.window_size:
            oldest = self.recent_window[0]
            self.pending_for_summary.append(oldest)
        
        # Add to recent window
        self.recent_window.append(item)
        self.conversation_history.append(item)
        self.total_items_processed += 1
        
        # Check if we should create a summary
        if len(self.pending_for_summary) >= self.summary_batch_size:
            self._create_summary()
        
        logger.debug(f"Added item to sliding window (total: {self.total_items_processed})")
    
    def _create_summary(self) -> None:
        """Create a summary of pending items."""
        if not self.pending_for_summary:
            return
        
        if self.enable_llm_summary:
            # TODO: Implement LLM-based summarization
            summary = self._create_statistical_summary()
        else:
            summary = self._create_statistical_summary()
        
        self.summaries.append(summary)
        self.pending_for_summary = []
        
        logger.info(f"Created summary batch (total summaries: {len(self.summaries)})")
    
    def _create_statistical_summary(self) -> Dict[str, Any]:
        """
        Create a statistical summary of pending items.
        
        Returns:
            Summary dictionary with statistics
        """
        if not self.pending_for_summary:
            return {}
        
        # Extract data
        symbols = {}
        prices = []
        changes = []
        volumes = []
        response_count = 0
        
        for item in self.pending_for_summary:
            data = item['data']
            symbol = data.get('symbol', 'N/A')
            
            # Track per-symbol stats
            if symbol not in symbols:
                symbols[symbol] = {
                    'count': 0,
                    'prices': [],
                    'changes': [],
                }
            
            symbols[symbol]['count'] += 1
            symbols[symbol]['prices'].append(data.get('price', 0))
            symbols[symbol]['changes'].append(data.get('change', 0))
            
            prices.append(data.get('price', 0))
            changes.append(data.get('change', 0))
            volumes.append(data.get('volume', 0))
            
            if item['response']:
                response_count += 1
        
        # Calculate statistics
        summary = {
            'period_start': self.pending_for_summary[0]['timestamp'],
            'period_end': self.pending_for_summary[-1]['timestamp'],
            'total_points': len(self.pending_for_summary),
            'response_count': response_count,
            'symbols': {},
        }
        
        # Per-symbol summaries
        for symbol, stats in symbols.items():
            if stats['prices']:
                summary['symbols'][symbol] = {
                    'count': stats['count'],
                    'avg_price': sum(stats['prices']) / len(stats['prices']),
                    'min_price': min(stats['prices']),
                    'max_price': max(stats['prices']),
                    'avg_change': sum(stats['changes']) / len(stats['changes']),
                    'max_change': max(stats['changes']),
                    'min_change': min(stats['changes']),
                }
        
        return summary
    
    def get_context(self, current_data: Dict[str, Any], max_tokens: int = 2000) -> str:
        """
        Get context from recent window and summaries.
        
        Args:
            current_data: Current data point
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add summaries first (most compact)
        if self.summaries:
            context_parts.append("Historical Summary:")
            for i, summary in enumerate(self.summaries):
                summary_text = self._format_summary(summary)
                context_parts.append(f"Period {i+1}: {summary_text}")
        
        # Add recent window (full detail)
        if self.recent_window:
            context_parts.append("\nRecent Data (detailed):")
            for i, item in enumerate(list(self.recent_window)[-10:]):  # Last 10 for brevity
                data_text = self.format_data_point(item['data'])
                context_parts.append(f"{i+1}. {data_text}")
                if item['response']:
                    context_parts.append(f"   Analysis: {item['response'][:100]}...")
        
        # Rough token limit check (4 chars ≈ 1 token)
        full_context = '\n'.join(context_parts)
        estimated_tokens = len(full_context) / 4
        
        if estimated_tokens > max_tokens:
            # Trim recent window if needed
            context_parts = context_parts[:len(context_parts)//2]
            full_context = '\n'.join(context_parts)
        
        return full_context if context_parts else "No historical data available."
    
    def _format_summary(self, summary: Dict[str, Any]) -> str:
        """Format a summary for display."""
        if not summary:
            return "Empty summary"
        
        parts = [f"{summary.get('total_points', 0)} data points"]
        
        symbols = summary.get('symbols', {})
        for symbol, stats in symbols.items():
            parts.append(
                f"{symbol}: ${stats['avg_price']:.2f} avg "
                f"(${stats['min_price']:.2f}-${stats['max_price']:.2f}), "
                f"{stats['avg_change']:+.2f}% avg change"
            )
        
        return ' | '.join(parts)
    
    def clear(self) -> None:
        """Clear all stored memory."""
        self.recent_window.clear()
        self.summaries.clear()
        self.pending_for_summary = []
        self.conversation_history = []
        self.total_items_processed = 0
        logger.info("Cleared sliding window memory")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_items': self.total_items_processed,
            'memory_type': 'Sliding Window + Summarization',
            'window_size': self.window_size,
            'recent_items': len(self.recent_window),
            'summaries': len(self.summaries),
            'pending_summary': len(self.pending_for_summary),
        }
    
    def force_summarize(self) -> None:
        """Force creation of a summary from pending items."""
        if self.pending_for_summary:
            self._create_summary()

