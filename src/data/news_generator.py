"""News event generator for trading simulation."""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class NewsEvent:
    """Represents a news event that could affect stock prices."""
    
    def __init__(
        self,
        symbol: str,
        headline: str,
        sentiment: str,  # 'positive', 'negative', 'neutral'
        impact: str,     # 'low', 'medium', 'high'
        timestamp: datetime,
        category: str,   # 'earnings', 'product', 'regulatory', 'market', 'executive'
    ):
        self.symbol = symbol
        self.headline = headline
        self.sentiment = sentiment
        self.impact = impact
        self.timestamp = timestamp
        self.category = category
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'type': 'news',
            'symbol': self.symbol,
            'headline': self.headline,
            'sentiment': self.sentiment,
            'impact': self.impact,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"[{self.impact.upper()} {self.sentiment.upper()}] {self.symbol}: {self.headline}"


class NewsGenerator:
    """
    Generates realistic news events for trading simulation.
    
    News events are injected at configurable intervals (e.g., 1-2 times per day)
    and can affect stock prices and trading decisions.
    """
    
    # News templates by category and sentiment
    NEWS_TEMPLATES = {
        'earnings': {
            'positive': [
                "{symbol} beats earnings expectations by {percent}%",
                "{symbol} reports record quarterly revenue of ${amount}B",
                "{symbol} announces {percent}% increase in profit margins",
                "{symbol} exceeds analyst estimates with strong Q{quarter} results",
            ],
            'negative': [
                "{symbol} misses earnings forecast by {percent}%",
                "{symbol} reports disappointing quarterly results",
                "{symbol} warns of lower-than-expected revenue",
                "{symbol} announces {percent}% decline in quarterly profits",
            ],
            'neutral': [
                "{symbol} releases Q{quarter} earnings report",
                "{symbol} announces quarterly financial results",
            ],
        },
        'product': {
            'positive': [
                "{symbol} launches innovative new product line",
                "{symbol} announces breakthrough technology advancement",
                "{symbol} unveils highly anticipated product at major event",
                "{symbol} receives strong pre-orders for new product",
            ],
            'negative': [
                "{symbol} delays major product launch",
                "{symbol} recalls product due to quality issues",
                "{symbol} faces criticism over new product features",
                "{symbol} discontinues underperforming product line",
            ],
            'neutral': [
                "{symbol} announces product update schedule",
                "{symbol} reveals product roadmap for next year",
            ],
        },
        'regulatory': {
            'positive': [
                "{symbol} wins major regulatory approval",
                "{symbol} settles lawsuit favorably",
                "{symbol} receives government contract worth ${amount}M",
            ],
            'negative': [
                "{symbol} faces regulatory investigation",
                "{symbol} hit with ${amount}M fine",
                "{symbol} loses major legal battle",
                "{symbol} under scrutiny for compliance issues",
            ],
            'neutral': [
                "{symbol} files regulatory documents",
                "{symbol} responds to regulatory inquiry",
            ],
        },
        'market': {
            'positive': [
                "Analysts upgrade {symbol} to 'Buy' rating",
                "{symbol} added to major market index",
                "Institutional investors increase {symbol} holdings by {percent}%",
                "{symbol} stock buyback program announced",
            ],
            'negative': [
                "Analysts downgrade {symbol} to 'Sell' rating",
                "Major investor reduces {symbol} stake by {percent}%",
                "{symbol} faces increased short interest",
                "Market concerns grow over {symbol} valuation",
            ],
            'neutral': [
                "Analysts maintain neutral stance on {symbol}",
                "{symbol} trading volume increases",
            ],
        },
        'executive': {
            'positive': [
                "{symbol} appoints experienced CEO from Fortune 500",
                "{symbol} announces strategic leadership expansion",
                "{symbol} executive team strengthened with key hire",
            ],
            'negative': [
                "{symbol} CEO unexpectedly resigns",
                "{symbol} CFO steps down amid controversy",
                "{symbol} faces executive turnover concerns",
            ],
            'neutral': [
                "{symbol} announces management changes",
                "{symbol} executive gives keynote speech",
            ],
        },
    }
    
    def __init__(
        self,
        symbols: List[str],
        events_per_day: float = 1.5,  # Average events per day
        enabled: bool = True,
    ):
        """
        Initialize news generator.
        
        Args:
            symbols: List of stock symbols to generate news for
            events_per_day: Average number of news events per day
            enabled: Whether news generation is enabled
        """
        self.symbols = symbols
        self.events_per_day = events_per_day
        self.enabled = enabled
        
        # Calculate seconds between events on average
        self.avg_seconds_between_events = (24 * 60 * 60) / events_per_day if events_per_day > 0 else float('inf')
        
        self.last_event_time = datetime.now()
        self.generated_events = []
        
        logger.info(
            f"News generator initialized: {events_per_day} events/day, "
            f"avg interval: {self.avg_seconds_between_events/3600:.1f} hours"
        )
    
    def should_generate_event(self, current_time: datetime) -> bool:
        """
        Determine if a news event should be generated now.
        
        Uses probabilistic approach to achieve target events_per_day.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if event should be generated
        """
        if not self.enabled:
            return False
        
        # Time since last event
        time_since_last = (current_time - self.last_event_time).total_seconds()
        
        # Probability increases with time since last event
        # Use exponential distribution
        probability = 1 - (2.71828 ** (-time_since_last / self.avg_seconds_between_events))
        
        return random.random() < probability
    
    def generate_event(self, current_time: datetime) -> Optional[NewsEvent]:
        """
        Generate a random news event.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            NewsEvent or None
        """
        if not self.should_generate_event(current_time):
            return None
        
        # Select random symbol
        symbol = random.choice(self.symbols)
        
        # Select random category
        category = random.choice(list(self.NEWS_TEMPLATES.keys()))
        
        # Select sentiment with weighted probabilities
        # 40% positive, 40% negative, 20% neutral
        sentiment = random.choices(
            ['positive', 'negative', 'neutral'],
            weights=[0.4, 0.4, 0.2],
            k=1
        )[0]
        
        # Select impact level
        # 60% low, 30% medium, 10% high
        impact = random.choices(
            ['low', 'medium', 'high'],
            weights=[0.6, 0.3, 0.1],
            k=1
        )[0]
        
        # Get template and fill in variables
        templates = self.NEWS_TEMPLATES[category][sentiment]
        template = random.choice(templates)
        
        # Fill in template variables
        headline = template.format(
            symbol=symbol,
            percent=random.randint(5, 25),
            amount=random.randint(1, 50),
            quarter=random.randint(1, 4),
        )
        
        event = NewsEvent(
            symbol=symbol,
            headline=headline,
            sentiment=sentiment,
            impact=impact,
            timestamp=current_time,
            category=category,
        )
        
        self.last_event_time = current_time
        self.generated_events.append(event)
        
        logger.info(f"Generated news event: {event}")
        
        return event
    
    def get_recent_events(self, hours: int = 24) -> List[NewsEvent]:
        """
        Get news events from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent NewsEvent objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.generated_events
            if event.timestamp >= cutoff_time
        ]
    
    def reset(self) -> None:
        """Reset the news generator."""
        self.last_event_time = datetime.now()
        self.generated_events = []
        logger.info("News generator reset")

