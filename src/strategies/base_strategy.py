"""Base class for prompting strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from ..llm.ollama_client import OllamaClient
from ..llm.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for prompting strategies.
    
    Defines the interface that all prompting strategies must implement.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_manager: PromptManager,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize base strategy.
        
        Args:
            llm_client: LLM client for generating responses
            prompt_manager: Prompt manager for formatting prompts
            config: Strategy-specific configuration
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.config = config or {}
        
        # Data history
        self.data_history: List[Dict[str, Any]] = []
        
        # Response history
        self.response_history: List[Dict[str, Any]] = []
        
        logger.info(f"{self.__class__.__name__} initialized.")
    
    @abstractmethod
    def process_data_point(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Process a new data point.
        
        Args:
            data: New data point
            
        Returns:
            LLM response if generated, None otherwise
        """
        pass
    
    def add_to_history(self, data: Dict[str, Any]) -> None:
        """
        Add data point to history.
        
        Args:
            data: Data point to add
        """
        self.data_history.append(data)
        
        # Optionally limit history size
        max_history = self.config.get('max_history_length', 100)
        if len(self.data_history) > max_history:
            self.data_history = self.data_history[-max_history:]
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent data history.
        
        Args:
            n: Number of recent items to retrieve
            
        Returns:
            List of recent data points
        """
        return self.data_history[-n:] if self.data_history else []
    
    def save_response(
        self,
        data: Dict[str, Any],
        prompt: str,
        response: str,
    ) -> None:
        """
        Save a response to history.
        
        Args:
            data: Data point that triggered the response
            prompt: Prompt sent to LLM
            response: LLM response
        """
        self.response_history.append({
            'data': data,
            'prompt': prompt,
            'response': response,
            'timestamp': data.get('timestamp', 'N/A'),
        })
        
        logger.debug(f"Response saved. Total responses: {len(self.response_history)}")
    
    def get_response_history(self) -> List[Dict[str, Any]]:
        """Get all response history."""
        return self.response_history.copy()
    
    def clear_history(self) -> None:
        """Clear all history."""
        self.data_history = []
        self.response_history = []
        self.llm_client.clear_history()
        logger.info("All history cleared.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'strategy': self.__class__.__name__,
            'data_points_processed': len(self.data_history),
            'responses_generated': len(self.response_history),
            'config': self.config,
        }

