"""Prompt management for continuous prompting strategies."""

import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt construction and formatting for different prompting strategies.
    
    Handles data formatting, history management, and template rendering.
    """
    
    def __init__(
        self,
        system_prompt: str = None,
        continuous_prompt_template: str = None,
        event_prompt_template: str = None,
    ):
        """
        Initialize prompt manager.
        
        Args:
            system_prompt: System prompt for the LLM
            continuous_prompt_template: Template for continuous prompting
            event_prompt_template: Template for event-driven prompting
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.continuous_prompt_template = (
            continuous_prompt_template or self._default_continuous_template()
        )
        self.event_prompt_template = (
            event_prompt_template or self._default_event_template()
        )
        
        logger.info("Prompt manager initialized.")
    
    @staticmethod
    def _default_system_prompt() -> str:
        """Default system prompt."""
        return (
            "You are a trading data analyst. You receive streaming market data in real-time. "
            "Your role is to observe patterns, identify significant events, and provide insights "
            "when appropriate. Be concise and focus on actionable observations."
        )
    
    @staticmethod
    def _default_continuous_template() -> str:
        """Default continuous prompting template."""
        return (
            "New market data:\n{data}\n\n"
            "Recent history:\n{history}\n\n"
            "What do you observe? Any significant patterns or recommendations?"
        )
    
    @staticmethod
    def _default_event_template() -> str:
        """Default event-driven prompting template."""
        return (
            "ALERT: Significant event detected!\n"
            "Event Type: {event_type}\n"
            "Current Data: {data}\n"
            "Context: {context}\n\n"
            "Please analyze this situation and provide your assessment."
        )
    
    def format_data_point(self, data: Dict[str, Any]) -> str:
        """
        Format a single data point for display.
        
        Args:
            data: Data point dictionary
            
        Returns:
            Formatted string representation
        """
        # Remove metadata fields for cleaner display
        display_data = {
            k: v for k, v in data.items()
            if k not in ['stream_timestamp', 'index']
        }
        
        # Format as readable text
        lines = []
        for key, value in display_data.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def format_history(
        self,
        history: List[Dict[str, Any]],
        max_items: int = 5,
    ) -> str:
        """
        Format historical data points.
        
        Args:
            history: List of historical data points
            max_items: Maximum number of items to include
            
        Returns:
            Formatted history string
        """
        if not history:
            return "No previous data."
        
        # Take most recent items
        recent_history = history[-max_items:]
        
        lines = []
        for i, data in enumerate(recent_history, 1):
            symbol = data.get('symbol', 'N/A')
            price = data.get('price', 'N/A')
            change = data.get('change', 'N/A')
            timestamp = data.get('timestamp', 'N/A')
            
            lines.append(
                f"  [{i}] {symbol}: ${price} ({change:+.2f}%) at {timestamp}"
            )
        
        return "\n".join(lines)
    
    def build_continuous_prompt(
        self,
        current_data: Dict[str, Any],
        history: List[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a prompt for continuous prompting strategy.
        
        Args:
            current_data: Current data point
            history: Historical data points
            
        Returns:
            Formatted prompt
        """
        data_str = self.format_data_point(current_data)
        history_str = self.format_history(history or [])
        
        prompt = self.continuous_prompt_template.format(
            data=data_str,
            history=history_str,
        )
        
        return prompt
    
    def build_event_prompt(
        self,
        event_type: str,
        current_data: Dict[str, Any],
        context: str = "",
    ) -> str:
        """
        Build a prompt for event-driven prompting strategy.
        
        Args:
            event_type: Type of event detected
            current_data: Current data point
            context: Additional context
            
        Returns:
            Formatted prompt
        """
        data_str = self.format_data_point(current_data)
        
        prompt = self.event_prompt_template.format(
            event_type=event_type,
            data=data_str,
            context=context or "No additional context.",
        )
        
        return prompt
    
    def build_adaptive_prompt(
        self,
        current_data: Dict[str, Any],
        question: str,
    ) -> str:
        """
        Build a prompt for adaptive prompting strategy.
        
        Args:
            current_data: Current data point
            question: Question to ask the LLM
            
        Returns:
            Formatted prompt
        """
        data_str = self.format_data_point(current_data)
        
        prompt = f"Current market data:\n{data_str}\n\n{question}"
        
        return prompt
    
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set a new system prompt."""
        self.system_prompt = prompt
        logger.info("System prompt updated.")
    
    def set_continuous_template(self, template: str) -> None:
        """Set a new continuous prompt template."""
        self.continuous_prompt_template = template
        logger.info("Continuous prompt template updated.")
    
    def set_event_template(self, template: str) -> None:
        """Set a new event prompt template."""
        self.event_prompt_template = template
        logger.info("Event prompt template updated.")

