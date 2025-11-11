"""Metrics tracking for experiments."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and save metrics from continuous prompting experiments.
    """
    
    def __init__(self, experiment_name: str = "default", save_dir: str = "logs/metrics"):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save metrics
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'data_points_processed': 0,
            'llm_calls': 0,
            'responses': [],
            'errors': [],
        }
        
        logger.info(f"Metrics tracker initialized for experiment: {experiment_name}")
    
    def record_data_point(self) -> None:
        """Record that a data point was processed."""
        self.metrics['data_points_processed'] += 1
    
    def record_llm_call(
        self,
        prompt: str,
        response: str,
        data: Dict[str, Any] = None,
    ) -> None:
        """
        Record an LLM call.
        
        Args:
            prompt: Prompt sent to LLM
            response: LLM response
            data: Associated data point
        """
        self.metrics['llm_calls'] += 1
        self.metrics['responses'].append({
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(prompt),
            'response_length': len(response),
            'response': response,
            'data': data,
        })
    
    def record_error(self, error: str, context: str = "") -> None:
        """
        Record an error.
        
        Args:
            error: Error message
            context: Additional context
        """
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context,
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def save(self) -> None:
        """Save metrics to file."""
        self.metrics['end_time'] = datetime.now().isoformat()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = self.save_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {filepath}")
    
    def print_summary(self) -> None:
        """Print a summary of metrics."""
        print("\n" + "="*60)
        print(f"Experiment: {self.experiment_name}")
        print("="*60)
        print(f"Data points processed: {self.metrics['data_points_processed']}")
        print(f"LLM calls made: {self.metrics['llm_calls']}")
        print(f"Errors encountered: {len(self.metrics['errors'])}")
        
        if self.metrics['llm_calls'] > 0:
            avg_response_length = sum(
                r['response_length'] for r in self.metrics['responses']
            ) / len(self.metrics['responses'])
            print(f"Average response length: {avg_response_length:.0f} characters")
        
        print("="*60 + "\n")

