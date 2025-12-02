"""Reactive prompting strategy - LLM only responds when it detects something important."""

from typing import Dict, Any, Optional, List
import logging
import re

from .base_strategy import BaseStrategy
from src.memory import BaseMemoryManager, SlidingWindowMemoryManager
from src.portfolio import PortfolioManager

logger = logging.getLogger(__name__)


class ReactiveStrategy(BaseStrategy):
    """
    Reactive prompting strategy.
    
    The LLM sees all incoming data but only responds when it detects something
    important or noteworthy. This solves the continuous prompting problem by:
    1. Maintaining full context of all data (via memory manager)
    2. Only generating responses when necessary (saves tokens, reduces latency)
    3. Providing more useful, actionable alerts instead of constant commentary
    
    The strategy uses configurable triggers to determine when to prompt the LLM:
    - Significant price changes
    - Volume spikes
    - News events
    - Pattern detection (multiple related events)
    - Time-based (periodic check-ins)
    """
    
    def __init__(
        self,
        llm_client,
        prompt_manager,
        config=None,
        memory_manager: Optional[BaseMemoryManager] = None,
        portfolio_manager: Optional[PortfolioManager] = None
    ):
        """
        Initialize reactive strategy.

        Config options:
            - triggers: List of trigger configurations
            - check_interval: How often to check for patterns (in data points)
            - alert_cooldown: Minimum seconds between alerts for same symbol
            - enable_trading: Whether to enable trading decisions (default: True)
        """
        super().__init__(llm_client, prompt_manager, config)

        # Memory manager for context management
        self.memory_manager = memory_manager
        if self.memory_manager is None:
            self.memory_manager = SlidingWindowMemoryManager(config.get('memory', {}))

        # Portfolio manager for trading
        self.portfolio_manager = portfolio_manager
        self.enable_trading = self.config.get('enable_trading', True)

        # Trigger configuration
        self.triggers = self.config.get('triggers', [])
        self.check_interval = self.config.get('check_interval', 1)
        self.alert_cooldown = self.config.get('alert_cooldown', 60)  # seconds

        # State tracking
        self.data_points_since_check = 0
        self.last_alert_time = {}  # symbol -> timestamp
        self.baseline_metrics = {}  # symbol -> {avg_volume, avg_price, etc}
        self.data_buffer = []  # Buffer for pattern detection

        logger.info(
            f"Reactive strategy configured: "
            f"triggers={len(self.triggers)}, "
            f"check_interval={self.check_interval}, "
            f"memory={self.memory_manager.__class__.__name__}, "
            f"trading={'enabled' if self.enable_trading and self.portfolio_manager else 'disabled'}"
        )
    
    def process_data_point(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Process a new data point reactively.

        The LLM sees all data (stored in memory) but only responds when
        something important is detected.

        Args:
            data: New data point (can be a single stock or a batch of stocks)

        Returns:
            LLM response if triggered, None otherwise
        """
        # Handle batch data (all stocks at once)
        if data.get('type') == 'batch':
            stocks = data.get('stocks', [])

            # Update portfolio prices for all stocks
            if self.portfolio_manager:
                for stock_data in stocks:
                    symbol = stock_data.get('symbol')
                    price = stock_data.get('price')
                    if symbol and price:
                        self.portfolio_manager.update_price(symbol, price)

            # Add batch to history and memory
            self.add_to_history(data)
            self.data_buffer.append(data)

            # Update baseline metrics for each stock
            for stock_data in stocks:
                self._update_baseline_metrics(stock_data)

            # Check if we should evaluate triggers
            self.data_points_since_check += 1

            # Evaluate triggers across all stocks in the batch
            triggered, trigger_info = self._evaluate_triggers_batch(stocks)

            if triggered:
                # Generate LLM response with ALL stocks visible
                response = self._generate_alert_batch(data, trigger_info)

                # Store in memory with response
                self.memory_manager.add_data_point(data, response)

                # Update last alert time
                from datetime import datetime
                for stock_data in stocks:
                    symbol = stock_data.get('symbol', 'UNKNOWN')
                    self.last_alert_time[symbol] = datetime.now()

                # Reset check counter
                self.data_points_since_check = 0

                return response
            else:
                # Store in memory without response (silent observation)
                self.memory_manager.add_data_point(data, None)
                return None

        else:
            # Single stock data point (legacy)
            # Always add to history and memory
            self.add_to_history(data)
            self.data_buffer.append(data)

            # Update portfolio prices on EVERY data point (not just alerts)
            if self.portfolio_manager and data.get('type') != 'news':
                symbol = data.get('symbol')
                price = data.get('price')
                if symbol and price:
                    self.portfolio_manager.update_price(symbol, price)

            # Update baseline metrics
            self._update_baseline_metrics(data)

            # Check if we should evaluate triggers
            self.data_points_since_check += 1

            # Evaluate triggers
            triggered, trigger_info = self._evaluate_triggers(data)

            if triggered:
                # Generate LLM response
                response = self._generate_alert(data, trigger_info)

                # Store in memory with response
                self.memory_manager.add_data_point(data, response)

                # Update last alert time
                symbol = data.get('symbol', 'UNKNOWN')
                from datetime import datetime
                self.last_alert_time[symbol] = datetime.now()

                # Reset check counter
                self.data_points_since_check = 0

                return response
            else:
                # Store in memory without response (silent observation)
                self.memory_manager.add_data_point(data, None)
                return None
    
    def _evaluate_triggers(self, data: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """
        Evaluate all triggers to determine if LLM should respond.
        
        Args:
            data: Current data point
            
        Returns:
            Tuple of (should_trigger, trigger_info)
        """
        symbol = data.get('symbol', 'UNKNOWN')
        
        # Check cooldown
        if not self._check_cooldown(symbol):
            return False, {}
        
        # Check each trigger type
        for trigger in self.triggers:
            trigger_type = trigger.get('type')
            
            if trigger_type == 'price_change':
                triggered, info = self._check_price_change(data, trigger)
                if triggered:
                    return True, {'type': 'price_change', **info}
            
            elif trigger_type == 'volume_spike':
                triggered, info = self._check_volume_spike(data, trigger)
                if triggered:
                    return True, {'type': 'volume_spike', **info}
            
            elif trigger_type == 'news_event':
                triggered, info = self._check_news_event(data, trigger)
                if triggered:
                    return True, {'type': 'news_event', **info}
            
            elif trigger_type == 'pattern':
                triggered, info = self._check_pattern(data, trigger)
                if triggered:
                    return True, {'type': 'pattern', **info}
            
            elif trigger_type == 'time_interval':
                triggered, info = self._check_time_interval(data, trigger)
                if triggered:
                    return True, {'type': 'time_interval', **info}
        
        return False, {}

    def _evaluate_triggers_batch(self, stocks: List[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """
        Evaluate triggers across a batch of stocks.

        Args:
            stocks: List of stock data points

        Returns:
            Tuple of (should_trigger, trigger_info)
        """
        # Check if any stock triggers
        for stock_data in stocks:
            triggered, info = self._evaluate_triggers(stock_data)
            if triggered:
                # Return info about which stock triggered
                info['triggered_symbol'] = stock_data.get('symbol')
                return True, info

        return False, {}

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last alert for this symbol."""
        if symbol not in self.last_alert_time:
            return True
        
        from datetime import datetime
        time_since_last = (datetime.now() - self.last_alert_time[symbol]).total_seconds()
        return time_since_last >= self.alert_cooldown
    
    def _check_price_change(self, data: Dict[str, Any], trigger: Dict) -> tuple[bool, Dict]:
        """Check for significant price change."""
        threshold = trigger.get('threshold', 0.02)  # 2% default
        change = abs(data.get('change', 0)) / 100  # Convert percentage to decimal
        
        if change >= threshold:
            return True, {
                'change': data.get('change', 0),
                'threshold': threshold * 100,
                'price': data.get('price', 0),
            }
        return False, {}
    
    def _check_volume_spike(self, data: Dict[str, Any], trigger: Dict) -> tuple[bool, Dict]:
        """Check for volume spike compared to baseline."""
        threshold = trigger.get('threshold', 2.0)  # 2x average
        symbol = data.get('symbol')
        current_volume = data.get('volume', 0)
        
        if symbol in self.baseline_metrics:
            avg_volume = self.baseline_metrics[symbol].get('avg_volume', current_volume)
            if avg_volume > 0 and current_volume >= avg_volume * threshold:
                return True, {
                    'current_volume': current_volume,
                    'avg_volume': avg_volume,
                    'multiplier': current_volume / avg_volume,
                }
        
        return False, {}
    
    def _check_news_event(self, data: Dict[str, Any], trigger: Dict) -> tuple[bool, Dict]:
        """Check if this is a news event."""
        if data.get('type') == 'news':
            # News events always trigger
            return True, {
                'headline': data.get('headline', ''),
                'sentiment': data.get('sentiment', 'neutral'),
                'impact': data.get('impact', 'unknown'),
            }
        return False, {}
    
    def _check_pattern(self, data: Dict[str, Any], trigger: Dict) -> tuple[bool, Dict]:
        """Check for patterns in recent data."""
        pattern_type = trigger.get('pattern_type', 'consecutive_moves')
        
        if pattern_type == 'consecutive_moves':
            # Check for N consecutive moves in same direction
            required_count = trigger.get('count', 3)
            symbol = data.get('symbol')
            
            # Get recent data for this symbol
            recent = [d for d in self.data_buffer[-10:] if d.get('symbol') == symbol]
            
            if len(recent) >= required_count:
                # Check if all moves are in same direction
                changes = [d.get('change', 0) for d in recent[-required_count:]]
                all_positive = all(c > 0 for c in changes)
                all_negative = all(c < 0 for c in changes)
                
                if all_positive or all_negative:
                    direction = 'upward' if all_positive else 'downward'
                    total_change = sum(changes)
                    return True, {
                        'pattern': 'consecutive_moves',
                        'direction': direction,
                        'count': required_count,
                        'total_change': total_change,
                    }
        
        return False, {}
    
    def _check_time_interval(self, data: Dict[str, Any], trigger: Dict) -> tuple[bool, Dict]:
        """Check if enough data points have passed for periodic check-in."""
        interval = trigger.get('interval', 60)  # data points
        
        if self.data_points_since_check >= interval:
            return True, {
                'data_points': self.data_points_since_check,
                'interval': interval,
            }
        
        return False, {}
    
    def _update_baseline_metrics(self, data: Dict[str, Any]) -> None:
        """Update baseline metrics for anomaly detection."""
        symbol = data.get('symbol')
        if not symbol:
            return
        
        if symbol not in self.baseline_metrics:
            self.baseline_metrics[symbol] = {
                'avg_volume': data.get('volume', 0),
                'avg_price': data.get('price', 0),
                'count': 1,
            }
        else:
            metrics = self.baseline_metrics[symbol]
            count = metrics['count']
            
            # Update running averages
            metrics['avg_volume'] = (
                (metrics['avg_volume'] * count + data.get('volume', 0)) / (count + 1)
            )
            metrics['avg_price'] = (
                (metrics['avg_price'] * count + data.get('price', 0)) / (count + 1)
            )
            metrics['count'] = count + 1
    
    def _generate_alert(self, data: Dict[str, Any], trigger_info: Dict[str, Any]) -> str:
        """
        Generate LLM alert for triggered event.

        Args:
            data: Current data point
            trigger_info: Information about what triggered the alert

        Returns:
            LLM response
        """
        # Get relevant context from memory
        context = self.memory_manager.get_context(
            current_data=data,
            max_tokens=2000
        )

        # Build alert prompt
        trigger_type = trigger_info.get('type', 'unknown')

        # Add portfolio context if trading is enabled
        portfolio_context = ""
        if self.enable_trading and self.portfolio_manager:
            summary = self.portfolio_manager.get_summary()
            symbol = data.get('symbol')
            position = None
            if symbol:
                for pos in summary['positions']:
                    if pos['symbol'] == symbol:
                        position = pos
                        break

            portfolio_context = f"""
Portfolio Status:
- Cash Available: ${summary['cash']:.2f}
- Total Portfolio Value: ${summary['total_value']:.2f}
- Portfolio Return: {summary['total_return_pct']:.2f}%
- Buy & Hold Return: {summary['buy_hold_return_pct']:.2f}%
- Outperformance: {summary['outperformance']:.2f}%
"""
            if position:
                portfolio_context += f"""
Current Position in {symbol}:
- Shares Owned: {position['shares']:.4f}
- Average Cost: ${position['avg_cost']:.2f}
- Current Price: ${position['current_price']:.2f}
- Position Value: ${position['value']:.2f}
- Profit/Loss: ${position['profit_loss']:.2f} ({position['profit_loss_pct']:.2f}%)
"""
            else:
                portfolio_context += f"\nNo current position in {symbol}\n"

        trading_instruction = ""
        if self.enable_trading and self.portfolio_manager:
            trading_instruction = """
TRADING DECISION REQUIRED:
Based on your analysis, make a trading decision. Respond with ONE of the following:
- BUY $X (where X is dollar amount, e.g., "BUY $500")
- SELL $X (where X is dollar amount, e.g., "SELL $300")
- HOLD (no action)

Include your trading decision at the END of your response on a new line starting with "DECISION:"
Example: "DECISION: BUY $500" or "DECISION: HOLD"
"""

        prompt = f"""ALERT: {trigger_type.upper().replace('_', ' ')} DETECTED

Current Data:
{self.prompt_manager.format_data_point(data)}

Trigger Details:
{self._format_trigger_info(trigger_info)}

Historical Context:
{context}
{portfolio_context}

Please analyze this situation and provide:
1. What is happening and why it's significant
2. Potential implications for the stock
3. Recommended action or monitoring focus
{trading_instruction}
"""

        # Get LLM response
        response = self.llm_client.chat(
            prompt,
            system_prompt=self.prompt_manager.system_prompt,
            maintain_history=True
        )

        # Execute trading decision if enabled
        if self.enable_trading and self.portfolio_manager and data.get('type') != 'news':
            self._execute_trading_decision(response, data)

        return response

    def _generate_alert_batch(self, batch_data: Dict[str, Any], trigger_info: Dict[str, Any]) -> str:
        """
        Generate LLM alert for triggered event with ALL stocks visible.

        Args:
            batch_data: Batch data containing all stocks
            trigger_info: Information about what triggered the alert

        Returns:
            LLM response
        """
        stocks = batch_data.get('stocks', [])

        # Get relevant context from memory
        context = self.memory_manager.get_context(
            current_data=batch_data,
            max_tokens=2000
        )

        # Build alert prompt
        trigger_type = trigger_info.get('type', 'unknown')
        triggered_symbol = trigger_info.get('triggered_symbol', 'UNKNOWN')

        # Format ALL stocks data
        stocks_data_str = ""
        for stock in stocks:
            stocks_data_str += f"\n{stock['symbol']}:\n"
            stocks_data_str += f"  Price: ${stock['price']:.2f}\n"
            stocks_data_str += f"  Change: {stock['change']:+.2f}%\n"
            stocks_data_str += f"  Volume: {stock['volume']:,}\n"

        # Add portfolio context if trading is enabled
        portfolio_context = ""
        if self.enable_trading and self.portfolio_manager:
            summary = self.portfolio_manager.get_summary()

            portfolio_context = f"""
Portfolio Status:
- Cash Available: ${summary['cash']:.2f}
- Total Portfolio Value: ${summary['total_value']:.2f}
- Portfolio Return: {summary['total_return_pct']:.2f}%
- Buy & Hold Return: {summary['buy_hold_return_pct']:.2f}%
- Outperformance: {summary['outperformance']:.2f}%

Current Positions:
"""
            for pos in summary['positions']:
                if pos['shares'] > 0:
                    portfolio_context += f"""
{pos['symbol']}:
  - Shares: {pos['shares']:.4f}
  - Avg Cost: ${pos['avg_cost']:.2f}
  - Current Price: ${pos['current_price']:.2f}
  - Value: ${pos['value']:.2f}
  - P/L: ${pos['profit_loss']:.2f} ({pos['profit_loss_pct']:.2f}%)
"""

        trading_instruction = ""
        if self.enable_trading and self.portfolio_manager:
            summary = self.portfolio_manager.get_summary()

            # List stocks we currently own
            owned_stocks = [pos['symbol'] for pos in summary['positions'] if pos['shares'] > 0]
            owned_str = ", ".join(owned_stocks) if owned_stocks else "None"

            # Build detailed position info
            position_details = ""
            if owned_stocks:
                position_details = "\n\nCURRENT POSITIONS (STOCKS YOU OWN):"
                for pos in summary['positions']:
                    if pos['shares'] > 0:
                        position_details += f"\n  • {pos['symbol']}: {pos['shares']:.4f} shares @ ${pos['avg_cost']:.2f} avg cost (Current: ${pos['current_price']:.2f}, P/L: ${pos['profit_loss']:.2f})"
            else:
                position_details = "\n\nCURRENT POSITIONS: None (you don't own any stocks yet)"

            trading_instruction = f"""
═══════════════════════════════════════════════════════════════
CRITICAL: READ THIS FIRST - PORTFOLIO STATUS
═══════════════════════════════════════════════════════════════

CASH AVAILABLE: ${summary['cash']:.2f}
TOTAL PORTFOLIO VALUE: ${summary['total_value']:.2f}
{position_details}

═══════════════════════════════════════════════════════════════
TRADING RULES (MUST FOLLOW):
═══════════════════════════════════════════════════════════════

YOU CAN BUY: Any stock, but ONLY if you have cash available
   - Available cash: ${summary['cash']:.2f}
   - If cash is $0.00, you CANNOT buy anything!

YOU CAN SELL: ONLY stocks you currently own
   - Stocks you own: {owned_str}
   - If you don't own a stock, you CANNOT sell it!

YOU CAN HOLD: ONLY stocks you currently own
   - Don't say "HOLD AAPL" if you don't own AAPL!

DECISION FORMAT:
DECISION: BUY AAPL $500      (Buy $500 worth of AAPL - only if cash >= $500)
DECISION: SELL MSFT $200     (Sell $200 worth of MSFT - only if you own MSFT)
DECISION: HOLD GOOGL         (Keep holding GOOGL - only if you own GOOGL)

COMMON MISTAKES TO AVOID:
- DON'T try to buy if cash is $0.00
- DON'T say HOLD for stocks you don't own
- DON'T try to sell stocks you don't own
- DON'T exceed available cash with BUY orders

═══════════════════════════════════════════════════════════════
"""

        prompt = f"""{trading_instruction}

ALERT: {trigger_type.upper().replace('_', ' ')} DETECTED IN {triggered_symbol}

Current Market Data (ALL STOCKS):
{stocks_data_str}

Trigger Details:
{self._format_trigger_info(trigger_info)}

{portfolio_context}

Historical Context:
{context}

Please analyze this situation across ALL stocks and provide:
1. What is happening and why it's significant
2. How this affects the overall market/portfolio
3. Recommended actions for each stock
"""

        # Get LLM response
        response = self.llm_client.chat(
            prompt,
            system_prompt=self.prompt_manager.system_prompt,
            maintain_history=True
        )

        # Execute trading decisions if enabled
        if self.enable_trading and self.portfolio_manager:
            # Parse ALL trading decisions from response (not per stock)
            self._execute_all_trading_decisions(response, stocks)

        return response

    def _format_trigger_info(self, trigger_info: Dict[str, Any]) -> str:
        """Format trigger information for display."""
        lines = []
        for key, value in trigger_info.items():
            if key != 'type':
                lines.append(f"  {key}: {value}")
        return '\n'.join(lines) if lines else "  No additional details"

    def _execute_all_trading_decisions(self, llm_response: str, stocks: List[Dict[str, Any]]) -> None:
        """
        Parse ALL trading decisions from LLM response and execute them.

        Args:
            llm_response: LLM's response text
            stocks: List of current stock data
        """
        # Create a lookup for current prices
        stock_prices = {stock['symbol']: stock for stock in stocks}

        # Find all DECISION: lines in the response
        # Matches: "DECISION: BUY AAPL $500" or "DECISION: SELL GOOGL $300" or "DECISION: HOLD MSFT"
        decision_pattern = r'DECISION:\s*(BUY|SELL|HOLD)\s+([A-Z]+)\s*\$?(\d+(?:\.\d+)?)?'
        decisions = re.findall(decision_pattern, llm_response, re.IGNORECASE)

        if not decisions:
            logger.debug("No trading decisions found in LLM response")
            return

        logger.info(f"Found {len(decisions)} trading decision(s) in LLM response")

        for action, symbol, amount_str in decisions:
            action = action.upper()
            symbol = symbol.upper()

            # Check if symbol is valid
            if symbol not in stock_prices:
                logger.warning(f"Invalid symbol in decision: {symbol}")
                continue

            if action == 'HOLD':
                logger.info(f"Trading decision: HOLD {symbol}")
                continue

            # Parse amount
            if not amount_str:
                logger.warning(f"Trading decision {action} {symbol} has no amount specified, skipping")
                continue

            amount = float(amount_str)
            stock_data = stock_prices[symbol]
            price = stock_data['price']
            timestamp = stock_data.get('timestamp', '')

            # Execute the trade
            result = self.portfolio_manager.execute_trade(
                symbol=symbol,
                action=action.lower(),
                amount=amount,
                price=price,
                timestamp=timestamp
            )

            if result['success']:
                logger.info(f"✓ Trade executed: {action} ${amount} of {symbol} @ ${price:.2f}")
                logger.info(f"  Remaining cash: ${result.get('remaining_cash', 0):.2f}")

                # Log portfolio summary after trade
                summary = self.portfolio_manager.get_summary()
                logger.info(f"  Portfolio value: ${summary['total_value']:.2f} (Cash: ${summary['cash']:.2f} + Positions: ${summary['total_value'] - summary['cash']:.2f})")
            else:
                logger.warning(f"✗ Trade failed: {action} ${amount} of {symbol} - {result.get('reason', 'Unknown error')}")

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.data_points_since_check = 0
        self.last_alert_time = {}
        self.baseline_metrics = {}
        self.data_buffer = []
        if self.memory_manager:
            self.memory_manager.clear()
        logger.info("Reactive strategy reset")

