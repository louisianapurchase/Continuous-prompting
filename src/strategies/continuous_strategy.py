"""Continuous prompting strategy."""

from typing import Dict, Any, Optional
import logging

from .base_strategy import BaseStrategy
from src.memory import BaseMemoryManager, SlidingWindowMemoryManager

logger = logging.getLogger(__name__)


class ContinuousStrategy(BaseStrategy):
    """
    Continuous prompting strategy.
    
    Sends every data point (or batches of data points) to the LLM for analysis.
    This is the most straightforward approach but may be resource-intensive.
    """
    
    def __init__(self, llm_client, prompt_manager, config=None, memory_manager: Optional[BaseMemoryManager] = None):
        """
        Initialize continuous strategy.

        Config options:
            - batch_size: Number of data points before prompting (default: 1)
            - include_history: Whether to include historical data (default: True)
            - max_history_length: Maximum history items to include (default: 10)
            - memory_manager: Optional memory manager instance
        """
        super().__init__(llm_client, prompt_manager, config)

        self.batch_size = self.config.get('batch_size', 1)
        self.include_history = self.config.get('include_history', True)
        self.max_history_display = self.config.get('max_history_length', 10)

        # Memory manager for context management
        self.memory_manager = memory_manager
        if self.memory_manager is None:
            # Default to sliding window if none provided
            self.memory_manager = SlidingWindowMemoryManager(config.get('memory', {}))

        self.current_batch = []

        logger.info(
            f"Continuous strategy configured: "
            f"batch_size={self.batch_size}, "
            f"include_history={self.include_history}, "
            f"memory={self.memory_manager.__class__.__name__}"
        )
    
    def process_data_point(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Process a new data point with continuous prompting.
        
        Args:
            data: New data point
            
        Returns:
            LLM response if batch is complete, None otherwise
        """
        # Add to history
        self.add_to_history(data)
        
        # Add to current batch
        self.current_batch.append(data)
        
        # Check if batch is complete
        if len(self.current_batch) >= self.batch_size:
            response = self._process_batch()
            self.current_batch = []
            return response
        
        return None
    
    def _process_batch(self) -> str:
        """
        Process the current batch of data points.

        Returns:
            LLM response
        """
        # Use the most recent data point as primary
        current_data = self.current_batch[-1]

        # Get context from memory manager
        context = ""
        if self.include_history and self.memory_manager:
            context = self.memory_manager.get_context(
                current_data=current_data,
                max_tokens=2000
            )

        # Build prompt with memory context
        if context:
            prompt = f"{context}\n\nCurrent data:\n"
            prompt += self.prompt_manager.format_data_point(current_data)
            prompt += "\n\nWhat do you observe? Any significant patterns or recommendations?"
        else:
            # Fallback to old method if no memory manager
            history = []
            if self.include_history:
                history = self.get_recent_history(self.max_history_display)
                history = history[:-len(self.current_batch)]

            prompt = self.prompt_manager.build_continuous_prompt(
                current_data=current_data,
                history=history,
            )

        # Get system prompt
        system_prompt = self.prompt_manager.get_system_prompt()

        # Generate response
        logger.info(f"Sending batch of {len(self.current_batch)} data points to LLM...")

        response = self.llm_client.chat(
            message=prompt,
            system_prompt=system_prompt,
            maintain_history=True,  # Enable conversation memory!
        )

        # Add to memory manager
        if self.memory_manager:
            self.memory_manager.add_data_point(current_data, response)

        # Save response
        self.save_response(
            data=current_data,
            prompt=prompt,
            response=response,
        )

        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = super().get_stats()
        stats.update({
            'batch_size': self.batch_size,
            'current_batch_size': len(self.current_batch),
            'batches_processed': len(self.response_history),
        })

        # Add memory stats
        if self.memory_manager:
            stats['memory'] = self.memory_manager.get_stats()

        return stats

