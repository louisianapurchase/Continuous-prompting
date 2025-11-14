"""Base class for memory management strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseMemoryManager(ABC):
    """
    Abstract base class for memory management in continuous prompting.
    
    Memory managers handle how historical data and LLM responses are stored,
    retrieved, and managed to stay within token limits while maintaining
    relevant context.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Configuration dictionary for the memory manager
        """
        self.config = config or {}
        self.conversation_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def add_data_point(self, data: Dict[str, Any], response: Optional[str] = None) -> None:
        """
        Add a new data point and optional LLM response to memory.
        
        Args:
            data: The data point to store
            response: Optional LLM response to the data point
        """
        pass
    
    @abstractmethod
    def get_context(self, current_data: Dict[str, Any], max_tokens: int = 2000) -> str:
        """
        Retrieve relevant context for the current data point.
        
        Args:
            current_data: The current data point being processed
            max_tokens: Maximum tokens to use for context
            
        Returns:
            Formatted context string to include in the prompt
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memory."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            'total_items': len(self.conversation_history),
            'memory_type': self.__class__.__name__,
        }
    
    def format_data_point(self, data: Dict[str, Any]) -> str:
        """
        Format a data point for display in context.
        
        Args:
            data: Data point to format
            
        Returns:
            Formatted string representation
        """
        symbol = data.get('symbol', 'N/A')
        price = data.get('price', 0)
        change = data.get('change', 0)
        volume = data.get('volume', 0)
        timestamp = data.get('timestamp', 'N/A')
        
        return f"{symbol}: ${price:.2f} ({change:+.2f}%) | Vol: {volume:,} | {timestamp}"

