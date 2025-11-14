"""
Portfolio Manager for tracking trading performance.

Manages a virtual portfolio with $1000 per stock, executes buy/sell decisions,
and tracks performance vs market (buy-and-hold strategy).
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single trade."""
    
    def __init__(self, symbol: str, action: str, shares: float, price: float, timestamp: str):
        self.symbol = symbol
        self.action = action  # 'buy' or 'sell'
        self.shares = shares
        self.price = price
        self.timestamp = timestamp
        self.total = shares * price
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'timestamp': self.timestamp,
            'total': self.total
        }


class Position:
    """Represents a position in a single stock."""
    
    def __init__(self, symbol: str, shares: float = 0, avg_cost: float = 0):
        self.symbol = symbol
        self.shares = shares
        self.avg_cost = avg_cost
    
    def add_shares(self, shares: float, price: float):
        """Add shares to position (buy)."""
        if self.shares == 0:
            self.avg_cost = price
        else:
            total_cost = (self.shares * self.avg_cost) + (shares * price)
            self.shares += shares
            self.avg_cost = total_cost / self.shares
    
    def remove_shares(self, shares: float) -> bool:
        """Remove shares from position (sell). Returns True if successful."""
        if shares > self.shares:
            return False
        self.shares -= shares
        if self.shares == 0:
            self.avg_cost = 0
        return True
    
    def get_value(self, current_price: float) -> float:
        """Get current value of position."""
        return self.shares * current_price
    
    def get_profit_loss(self, current_price: float) -> float:
        """Get profit/loss for this position."""
        if self.shares == 0:
            return 0
        return (current_price - self.avg_cost) * self.shares
    
    def get_profit_loss_pct(self, current_price: float) -> float:
        """Get profit/loss percentage."""
        if self.avg_cost == 0:
            return 0
        return ((current_price - self.avg_cost) / self.avg_cost) * 100
    
    def to_dict(self, current_price: float) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'avg_cost': self.avg_cost,
            'current_price': current_price,
            'value': self.get_value(current_price),
            'profit_loss': self.get_profit_loss(current_price),
            'profit_loss_pct': self.get_profit_loss_pct(current_price)
        }


class PortfolioManager:
    """
    Manages a virtual trading portfolio.
    
    Each stock starts with $1000 allocated. The LLM can make buy/sell decisions.
    Tracks performance vs buy-and-hold strategy.
    """
    
    def __init__(self, symbols: List[str], initial_cash_per_symbol: float = 1000.0):
        """
        Initialize portfolio.
        
        Args:
            symbols: List of stock symbols to track
            initial_cash_per_symbol: Starting cash allocation per symbol (default: $1000)
        """
        self.symbols = symbols
        self.initial_cash_per_symbol = initial_cash_per_symbol
        self.total_initial_cash = initial_cash_per_symbol * len(symbols)
        
        # Current state
        self.cash = self.total_initial_cash
        self.positions: Dict[str, Position] = {symbol: Position(symbol) for symbol in symbols}
        self.trades: List[Trade] = []
        
        # Track initial prices for buy-and-hold comparison
        self.initial_prices: Dict[str, float] = {}
        self.buy_and_hold_shares: Dict[str, float] = {}
        
        # Current prices
        self.current_prices: Dict[str, float] = {}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def initialize_buy_and_hold(self, symbol: str, price: float):
        """
        Initialize buy-and-hold strategy for a symbol.
        Called when we first see a price for a symbol.
        """
        if symbol not in self.initial_prices:
            self.initial_prices[symbol] = price
            # Buy-and-hold: invest $1000 at initial price
            self.buy_and_hold_shares[symbol] = self.initial_cash_per_symbol / price
            logger.info(f"Buy-and-hold initialized for {symbol}: {self.buy_and_hold_shares[symbol]:.4f} shares @ ${price:.2f}")
    
    def update_price(self, symbol: str, price: float):
        """Update current price for a symbol."""
        self.current_prices[symbol] = price
        
        # Initialize buy-and-hold if this is the first price we've seen
        if symbol not in self.initial_prices:
            self.initialize_buy_and_hold(symbol, price)
    
    def execute_trade(self, symbol: str, action: str, amount: float, price: float, timestamp: str) -> Dict[str, Any]:
        """
        Execute a trade.
        
        Args:
            symbol: Stock symbol
            action: 'buy' or 'sell'
            amount: Dollar amount to trade (e.g., 500 = $500 worth)
            price: Current price per share
            timestamp: Trade timestamp
        
        Returns:
            Dict with trade result and status
        """
        action = action.lower()
        
        if action not in ['buy', 'sell']:
            return {'success': False, 'reason': f'Invalid action: {action}'}
        
        if symbol not in self.symbols:
            return {'success': False, 'reason': f'Symbol {symbol} not in portfolio'}
        
        shares = amount / price
        position = self.positions[symbol]
        
        if action == 'buy':
            # Check if we have enough cash
            cost = shares * price
            if cost > self.cash:
                return {'success': False, 'reason': f'Insufficient cash: ${self.cash:.2f} < ${cost:.2f}'}
            
            # Execute buy
            position.add_shares(shares, price)
            self.cash -= cost
            trade = Trade(symbol, 'buy', shares, price, timestamp)
            self.trades.append(trade)
            self.total_trades += 1
            
            logger.info(f"BUY: {shares:.4f} shares of {symbol} @ ${price:.2f} = ${cost:.2f}")
            
            return {
                'success': True,
                'action': 'buy',
                'shares': shares,
                'price': price,
                'cost': cost,
                'remaining_cash': self.cash
            }
        
        else:  # sell
            # Check if we have enough shares
            if shares > position.shares:
                return {'success': False, 'reason': f'Insufficient shares: {position.shares:.4f} < {shares:.4f}'}
            
            # Execute sell
            proceeds = shares * price
            profit_loss = (price - position.avg_cost) * shares
            position.remove_shares(shares)
            self.cash += proceeds
            trade = Trade(symbol, 'sell', shares, price, timestamp)
            self.trades.append(trade)
            self.total_trades += 1
            
            # Track win/loss
            if profit_loss > 0:
                self.winning_trades += 1
            elif profit_loss < 0:
                self.losing_trades += 1
            
            logger.info(f"SELL: {shares:.4f} shares of {symbol} @ ${price:.2f} = ${proceeds:.2f} (P/L: ${profit_loss:.2f})")
            
            return {
                'success': True,
                'action': 'sell',
                'shares': shares,
                'price': price,
                'proceeds': proceeds,
                'profit_loss': profit_loss,
                'remaining_cash': self.cash
            }
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        positions_value = sum(
            pos.get_value(self.current_prices.get(symbol, 0))
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_buy_and_hold_value(self) -> float:
        """Get value of buy-and-hold strategy."""
        return sum(
            self.buy_and_hold_shares.get(symbol, 0) * self.current_prices.get(symbol, 0)
            for symbol in self.symbols
        )
    
    def get_performance_vs_market(self) -> Dict[str, float]:
        """Get performance comparison vs buy-and-hold."""
        portfolio_value = self.get_portfolio_value()
        buy_hold_value = self.get_buy_and_hold_value()
        
        portfolio_return = ((portfolio_value - self.total_initial_cash) / self.total_initial_cash) * 100
        buy_hold_return = ((buy_hold_value - self.total_initial_cash) / self.total_initial_cash) * 100
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_return_pct': portfolio_return,
            'buy_hold_value': buy_hold_value,
            'buy_hold_return_pct': buy_hold_return,
            'outperformance': portfolio_return - buy_hold_return
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete portfolio summary."""
        performance = self.get_performance_vs_market()
        
        positions_summary = [
            pos.to_dict(self.current_prices.get(symbol, 0))
            for symbol, pos in self.positions.items()
            if pos.shares > 0
        ]
        
        return {
            'cash': self.cash,
            'positions': positions_summary,
            'total_value': performance['portfolio_value'],
            'initial_value': self.total_initial_cash,
            'total_return_pct': performance['portfolio_return_pct'],
            'buy_hold_value': performance['buy_hold_value'],
            'buy_hold_return_pct': performance['buy_hold_return_pct'],
            'outperformance': performance['outperformance'],
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        }

