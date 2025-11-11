"""Event-driven prompting strategy."""

from typing import Dict, Any, Optional, List
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class EventDrivenStrategy(BaseStrategy):
    """
    Event-driven prompting strategy.
    
    Only prompts the LLM when specific events or conditions are detected
    in the data stream. This is more efficient and focuses on significant moments.
    """
    
    def __init__(self, llm_client, prompt_manager, config=None):
        """
        Initialize event-driven strategy.
        
        Config options:
            - triggers: List of trigger configurations
              Each trigger can have:
                - type: 'price_change', 'volume_spike', 'time_interval'
                - threshold: Numeric threshold for the trigger
                - interval: For time-based triggers
        """
        super().__init__(llm_client, prompt_manager, config)
        
        self.triggers = self.config.get('triggers', [])
        
        # State tracking
        self.last_trigger_time = None
        self.last_prices = {}
        self.volume_history = []
        
        logger.info(
            f"Event-driven strategy configured with {len(self.triggers)} triggers."
        )
    
    def process_data_point(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Process a new data point and check for trigger events.
        
        Args:
            data: New data point
            
        Returns:
            LLM response if event triggered, None otherwise
        """
        # Add to history
        self.add_to_history(data)
        
        # Check all triggers
        for trigger in self.triggers:
            event_detected, event_info = self._check_trigger(trigger, data)
            
            if event_detected:
                logger.info(f"Event detected: {event_info['type']}")
                return self._handle_event(event_info, data)
        
        # Update state for future trigger checks
        self._update_state(data)
        
        return None
    
    def _check_trigger(
        self,
        trigger: Dict[str, Any],
        data: Dict[str, Any],
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a trigger condition is met.
        
        Args:
            trigger: Trigger configuration
            data: Current data point
            
        Returns:
            Tuple of (triggered, event_info)
        """
        trigger_type = trigger.get('type')
        
        if trigger_type == 'price_change':
            return self._check_price_change(trigger, data)
        elif trigger_type == 'volume_spike':
            return self._check_volume_spike(trigger, data)
        elif trigger_type == 'time_interval':
            return self._check_time_interval(trigger, data)
        else:
            logger.warning(f"Unknown trigger type: {trigger_type}")
            return False, None
    
    def _check_price_change(
        self,
        trigger: Dict[str, Any],
        data: Dict[str, Any],
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Check for significant price change."""
        symbol = data.get('symbol')
        current_price = data.get('price')
        threshold = trigger.get('threshold', 0.05)  # 5% default
        
        if symbol not in self.last_prices or current_price is None:
            return False, None
        
        last_price = self.last_prices[symbol]
        price_change = abs(current_price - last_price) / last_price
        
        if price_change >= threshold:
            return True, {
                'type': 'price_change',
                'symbol': symbol,
                'change_percent': price_change * 100,
                'threshold': threshold * 100,
                'old_price': last_price,
                'new_price': current_price,
            }
        
        return False, None
    
    def _check_volume_spike(
        self,
        trigger: Dict[str, Any],
        data: Dict[str, Any],
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Check for volume spike."""
        volume = data.get('volume')
        threshold = trigger.get('threshold', 2.0)  # 2x average default
        
        if volume is None or len(self.volume_history) < 5:
            return False, None
        
        avg_volume = sum(self.volume_history[-10:]) / len(self.volume_history[-10:])
        
        if volume >= avg_volume * threshold:
            return True, {
                'type': 'volume_spike',
                'symbol': data.get('symbol'),
                'volume': volume,
                'average_volume': avg_volume,
                'spike_ratio': volume / avg_volume,
            }
        
        return False, None
    
    def _check_time_interval(
        self,
        trigger: Dict[str, Any],
        data: Dict[str, Any],
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Check for time-based trigger."""
        interval = trigger.get('interval', 10)  # 10 data points default
        
        # Check based on data point count
        if len(self.data_history) % interval == 0:
            return True, {
                'type': 'time_interval',
                'interval': interval,
                'data_points_processed': len(self.data_history),
            }
        
        return False, None
    
    def _handle_event(
        self,
        event_info: Dict[str, Any],
        data: Dict[str, Any],
    ) -> str:
        """
        Handle a detected event by prompting the LLM.
        
        Args:
            event_info: Information about the event
            data: Current data point
            
        Returns:
            LLM response
        """
        # Build context from event info
        context_parts = []
        for key, value in event_info.items():
            if key != 'type':
                context_parts.append(f"{key}: {value}")
        context = "\n".join(context_parts)
        
        # Build prompt
        prompt = self.prompt_manager.build_event_prompt(
            event_type=event_info['type'],
            current_data=data,
            context=context,
        )
        
        # Get system prompt
        system_prompt = self.prompt_manager.get_system_prompt()
        
        # Generate response
        logger.info(f"Prompting LLM for event: {event_info['type']}")
        
        response = self.llm_client.chat(
            message=prompt,
            system_prompt=system_prompt,
            maintain_history=False,
        )
        
        # Save response
        self.save_response(
            data=data,
            prompt=prompt,
            response=response,
        )
        
        return response
    
    def _update_state(self, data: Dict[str, Any]) -> None:
        """Update internal state for trigger checking."""
        # Update last prices
        symbol = data.get('symbol')
        price = data.get('price')
        if symbol and price:
            self.last_prices[symbol] = price
        
        # Update volume history
        volume = data.get('volume')
        if volume:
            self.volume_history.append(volume)
            # Keep only recent history
            if len(self.volume_history) > 50:
                self.volume_history = self.volume_history[-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = super().get_stats()
        stats.update({
            'triggers_configured': len(self.triggers),
            'events_detected': len(self.response_history),
            'symbols_tracked': len(self.last_prices),
        })
        return stats

