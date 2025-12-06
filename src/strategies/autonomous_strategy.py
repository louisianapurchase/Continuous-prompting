"""
Autonomous LLM Strategy - Self-Activating with Modern Techniques.

This strategy uses cutting-edge LLM techniques to make the model truly autonomous:
1. Chain-of-Thought (CoT) reasoning for better decisions
2. ReAct pattern (Reasoning + Acting) for dynamic activation
3. Self-reflection and confidence scoring
4. Tool use - LLM decides when to activate itself
5. Multi-step reasoning with verification
6. Adaptive context based on market conditions
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.memory import BaseMemoryManager, SlidingWindowMemoryManager
from src.portfolio import PortfolioManager

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AutonomousStrategy(BaseStrategy):
    """
    Autonomous LLM trading strategy with self-activation.
    
    Instead of hardcoded triggers, the LLM itself decides:
    - When to activate and analyze the market
    - What patterns are significant
    - When to trade and when to wait
    - How confident it is in its decisions
    
    Uses modern techniques:
    - Chain-of-Thought prompting for reasoning
    - ReAct pattern for decision making
    - Self-reflection for quality control
    - Confidence scoring for risk management
    - Adaptive memory based on market volatility
    """
    
    def __init__(
        self,
        llm_client: Any,
        prompt_manager: Any,
        config: Optional[Dict[str, Any]] = None,
        memory_manager: Optional[BaseMemoryManager] = None,
        portfolio_manager: Optional[PortfolioManager] = None
    ) -> None:
        """
        Initialize autonomous strategy.

        Args:
            llm_client: LLM client for making inference calls
            prompt_manager: Manager for prompt templates
            config: Configuration dictionary with strategy settings
            memory_manager: Optional memory manager for context storage
            portfolio_manager: Optional portfolio manager for trading
        """
        super().__init__(llm_client, prompt_manager, config)
        
        # Memory manager
        self.memory_manager = memory_manager
        if self.memory_manager is None:
            self.memory_manager = SlidingWindowMemoryManager(config.get('memory', {}))
        
        # Portfolio manager
        self.portfolio_manager = portfolio_manager
        self.enable_trading = self.config.get('enable_trading', True)
        
        # Configuration
        self.activation_check_interval = self.config.get('activation_check_interval', 5)  # Check every N data points
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)  # 70% confidence minimum
        self.enable_self_reflection = self.config.get('enable_self_reflection', True)
        self.enable_cot = self.config.get('enable_chain_of_thought', True)
        
        # State tracking
        self.data_points_since_check = 0
        self.data_buffer = []
        self.recent_decisions = []  # Track recent decisions for learning
        self.market_volatility = 0.0  # Track market volatility
        
        logger.info(
            f"Autonomous strategy initialized: "
            f"activation_interval={self.activation_check_interval}, "
            f"min_confidence={self.min_confidence_threshold}, "
            f"cot={'enabled' if self.enable_cot else 'disabled'}, "
            f"reflection={'enabled' if self.enable_self_reflection else 'disabled'}"
        )
    
    def process_data_point(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Process data point with autonomous activation.

        The LLM decides when to activate based on what it sees.

        Args:
            data: Data point containing market information

        Returns:
            LLM response string if activated, None otherwise
        """
        # Handle batch data
        if data.get('type') == 'batch':
            stocks = data.get('stocks', [])
            
            # Update portfolio prices
            if self.portfolio_manager:
                for stock_data in stocks:
                    symbol = stock_data.get('symbol')
                    price = stock_data.get('price')
                    if symbol and price:
                        self.portfolio_manager.update_price(symbol, price)
            
            # Add to history and buffer
            self.add_to_history(data)
            self.data_buffer.append(data)
            
            # Update market volatility
            self._update_market_volatility(stocks)
            
            # Store in memory (silent observation)
            self.memory_manager.add_data_point(data, None)
            
            # Increment counter
            self.data_points_since_check += 1
            
            # Check if we should ask LLM if it wants to activate
            if self.data_points_since_check >= self.activation_check_interval:
                self.data_points_since_check = 0
                
                # Ask LLM: "Should I activate and analyze this?"
                should_activate, activation_reason = self._check_activation(data, stocks)
                
                if should_activate:
                    logger.info(f"LLM activated itself: {activation_reason}")
                    
                    # LLM decided to activate - perform full analysis
                    response = self._perform_autonomous_analysis(data, stocks, activation_reason)
                    
                    # Store in memory with response
                    self.memory_manager.add_data_point(data, response)
                    
                    return response
            
            return None

        return None

    def _check_activation(self, batch_data: Dict[str, Any], stocks: List[Dict[str, Any]]) -> tuple[bool, str]:
        """
        Ask LLM if it should activate and analyze the current market state.

        This is a lightweight check - LLM decides if something is worth deeper analysis.

        Args:
            batch_data: Batch data point containing timestamp and stocks
            stocks: List of stock data dictionaries

        Returns:
            Tuple of (should_activate: bool, reason: str)
        """
        # Format current market snapshot
        market_snapshot = self._format_market_snapshot(stocks)

        # Get recent context (last few data points)
        recent_context = self._get_recent_context(lookback=10)

        # Build activation check prompt
        prompt = f"""You are monitoring live market data. Your job is to decide if the current market state requires your attention and analysis.

CURRENT MARKET STATE:
{market_snapshot}

RECENT MARKET ACTIVITY (last 10 data points):
{recent_context}

MARKET VOLATILITY: {self.market_volatility:.4f}

QUESTION: Should you activate and perform a detailed analysis right now?

Think step-by-step:
1. Is there any unusual price movement?
2. Are there any emerging patterns?
3. Is this just normal market noise?
4. Would analyzing this lead to actionable insights?

Respond in JSON format:
{{
    "should_activate": true/false,
    "reason": "brief explanation",
    "confidence": 0.0-1.0
}}

Be selective - only activate when you see something genuinely interesting or actionable."""

        try:
            response = self.llm_client.chat(
                prompt,
                system_prompt="You are an expert market analyst. Be selective and only activate when necessary.",
                maintain_history=False
            )

            # Parse JSON response
            result = self._parse_json_response(response)

            if result and result.get('should_activate', False):
                confidence = result.get('confidence', 0.0)
                reason = result.get('reason', 'No reason provided')

                # Only activate if confidence is high enough
                if confidence >= self.min_confidence_threshold:
                    return True, reason
                else:
                    logger.debug(f"LLM wants to activate but confidence too low: {confidence:.2f}")
                    return False, ""

            return False, ""

        except Exception as e:
            logger.error(f"Error in activation check: {e}")
            return False, ""

    def _perform_autonomous_analysis(
        self,
        batch_data: Dict[str, Any],
        stocks: List[Dict[str, Any]],
        activation_reason: str
    ) -> str:
        """
        Perform full autonomous analysis with Chain-of-Thought reasoning.

        Uses ReAct pattern: Reasoning → Acting → Observing

        Args:
            batch_data: Batch data point containing timestamp and stocks
            stocks: List of stock data dictionaries
            activation_reason: Reason why LLM activated itself

        Returns:
            LLM analysis response string
        """
        # Get comprehensive context
        context = self.memory_manager.get_context(
            current_data=batch_data,
            max_tokens=3000
        )

        # Format market data
        market_snapshot = self._format_market_snapshot(stocks)

        # Get portfolio status
        portfolio_status = self._get_portfolio_status()

        # Build Chain-of-Thought analysis prompt
        if self.enable_cot:
            analysis_prompt = self._build_cot_analysis_prompt(
                market_snapshot,
                context,
                portfolio_status,
                activation_reason
            )
        else:
            analysis_prompt = self._build_standard_analysis_prompt(
                market_snapshot,
                context,
                portfolio_status,
                activation_reason
            )

        # Get LLM analysis
        analysis_response = self.llm_client.chat(
            analysis_prompt,
            system_prompt=self.prompt_manager.system_prompt,
            maintain_history=True
        )

        # Self-reflection: Ask LLM to verify its own decision
        if self.enable_self_reflection and self.enable_trading:
            analysis_response = self._perform_self_reflection(
                analysis_response,
                market_snapshot,
                portfolio_status
            )

        # Execute trading decisions
        if self.enable_trading and self.portfolio_manager:
            self._execute_autonomous_trades(analysis_response, stocks)

        return analysis_response

    def _build_cot_analysis_prompt(
        self,
        market_snapshot: str,
        context: str,
        portfolio_status: str,
        activation_reason: str
    ) -> str:
        """
        Build Chain-of-Thought analysis prompt.

        Args:
            market_snapshot: Current market state formatted string
            context: Historical context from memory
            portfolio_status: Current portfolio status formatted string
            activation_reason: Reason why LLM activated

        Returns:
            Formatted prompt string for Chain-of-Thought analysis
        """
        return f"""You activated yourself because: {activation_reason}

Now perform a detailed analysis using step-by-step reasoning.

═══════════════════════════════════════════════════════════════
CURRENT MARKET STATE:
═══════════════════════════════════════════════════════════════
{market_snapshot}

═══════════════════════════════════════════════════════════════
YOUR PORTFOLIO:
═══════════════════════════════════════════════════════════════
{portfolio_status}

═══════════════════════════════════════════════════════════════
HISTORICAL CONTEXT:
═══════════════════════════════════════════════════════════════
{context}

═══════════════════════════════════════════════════════════════
ANALYSIS FRAMEWORK (Think step-by-step):
═══════════════════════════════════════════════════════════════

STEP 1 - OBSERVATION:
What do you observe in the current market data? List specific facts.

STEP 2 - PATTERN RECOGNITION:
Are there any patterns, trends, or anomalies? Compare to historical context.

STEP 3 - OPPORTUNITY IDENTIFICATION:
Based on your observations, are there any trading opportunities?
- Which stocks look attractive to BUY?
- Which positions should you SELL?
- Which positions should you HOLD?

STEP 4 - RISK ASSESSMENT:
What are the risks of each potential action?
- What could go wrong?
- How confident are you? (0-100%)

STEP 5 - DECISION:
Make your final trading decision(s).

FORMAT YOUR RESPONSE AS:

OBSERVATION:
[Your observations here]

PATTERNS:
[Patterns you identified]

OPPORTUNITIES:
[Trading opportunities]

RISKS:
[Risk assessment]

CONFIDENCE: [0-100%]

DECISION: [BUY/SELL/HOLD] [SYMBOL] $[AMOUNT]
(You can make multiple decisions, one per line)

Example:
DECISION: BUY AAPL $500
DECISION: SELL MSFT $200
DECISION: HOLD GOOGL
"""

    def _build_standard_analysis_prompt(
        self,
        market_snapshot: str,
        context: str,
        portfolio_status: str,
        activation_reason: str
    ) -> str:
        """
        Build standard analysis prompt without Chain-of-Thought.

        Args:
            market_snapshot: Current market state formatted string
            context: Historical context from memory
            portfolio_status: Current portfolio status formatted string
            activation_reason: Reason why LLM activated

        Returns:
            Formatted prompt string for standard analysis
        """
        return f"""You activated yourself because: {activation_reason}

CURRENT MARKET:
{market_snapshot}

YOUR PORTFOLIO:
{portfolio_status}

HISTORICAL CONTEXT:
{context}

Analyze the situation and make trading decisions.

DECISION: [BUY/SELL/HOLD] [SYMBOL] $[AMOUNT]
"""

    def _perform_self_reflection(
        self,
        initial_analysis: str,
        market_snapshot: str,
        portfolio_status: str
    ) -> str:
        """
        Self-reflection: LLM critiques its own decision.

        This helps catch mistakes and improve decision quality.

        Args:
            initial_analysis: Initial LLM analysis and decision
            market_snapshot: Current market state formatted string
            portfolio_status: Current portfolio status formatted string

        Returns:
            Combined initial analysis and reflection response
        """
        reflection_prompt = f"""You just made the following analysis and trading decision:

═══════════════════════════════════════════════════════════════
YOUR INITIAL ANALYSIS:
═══════════════════════════════════════════════════════════════
{initial_analysis}

═══════════════════════════════════════════════════════════════
SELF-REFLECTION TASK:
═══════════════════════════════════════════════════════════════

Now step back and critique your own decision:

1. SANITY CHECK:
   - Did you check if you have enough cash before deciding to BUY?
   - Did you verify you own the stock before deciding to SELL?
   - Are your decisions based on solid reasoning or emotion?

2. RISK ASSESSMENT:
   - What's the worst-case scenario if this trade goes wrong?
   - Is the potential reward worth the risk?
   - Are you being too aggressive or too conservative?

3. ALTERNATIVE PERSPECTIVES:
   - What would a contrarian investor say about this decision?
   - Are you missing any important information?
   - Is there a better alternative action?

4. FINAL VERDICT:
   - Do you stand by your original decision? (YES/NO)
   - If NO, what should you do instead?
   - Confidence level: (0-100%)

CURRENT MARKET (for reference):
{market_snapshot}

YOUR PORTFOLIO (for reference):
{portfolio_status}

Respond with:
REFLECTION: [Your self-critique]
FINAL_DECISION: [CONFIRMED/REVISED]
REVISED_DECISION: [Only if you changed your mind]
CONFIDENCE: [0-100%]
"""

        try:
            reflection_response = self.llm_client.chat(
                reflection_prompt,
                system_prompt="You are reviewing your own trading decision. Be honest and critical.",
                maintain_history=True
            )

            # Check if LLM revised its decision
            if "REVISED_DECISION:" in reflection_response:
                logger.info("LLM revised its decision after self-reflection")
                # Return the reflection which includes the revised decision
                return f"{initial_analysis}\n\n--- SELF-REFLECTION ---\n{reflection_response}"
            else:
                # LLM confirmed its decision
                return f"{initial_analysis}\n\n--- SELF-REFLECTION ---\n{reflection_response}"

        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            return initial_analysis

    def _format_market_snapshot(self, stocks: List[Dict[str, Any]]) -> str:
        """
        Format current market state for LLM.

        Args:
            stocks: List of stock data dictionaries

        Returns:
            Formatted market snapshot string
        """
        lines = []
        for stock in stocks:
            lines.append(
                f"{stock['symbol']}: ${stock['price']:.2f} "
                f"({stock['change']:+.2f}%) "
                f"Vol: {stock['volume']:,}"
            )
        return "\n".join(lines)

    def _get_recent_context(self, lookback: int = 10) -> str:
        """
        Get recent market activity summary.

        Args:
            lookback: Number of recent data points to include

        Returns:
            Formatted string of recent market activity
        """
        if len(self.data_buffer) < 2:
            return "Not enough data yet"

        recent = self.data_buffer[-lookback:]
        lines = []

        for data_point in recent:
            if data_point.get('type') == 'batch':
                timestamp = data_point.get('timestamp', 'N/A')
                stocks = data_point.get('stocks', [])
                stock_summary = ", ".join([
                    f"{s['symbol']}: ${s['price']:.2f} ({s['change']:+.2f}%)"
                    for s in stocks
                ])
                lines.append(f"[{timestamp}] {stock_summary}")

        return "\n".join(lines) if lines else "No recent data"

    def _get_portfolio_status(self) -> str:
        """
        Get formatted portfolio status.

        Returns:
            Formatted string of current portfolio status
        """
        if not self.portfolio_manager:
            return "Trading disabled"

        summary = self.portfolio_manager.get_summary()

        lines = [
            f"Cash Available: ${summary['cash']:.2f}",
            f"Total Portfolio Value: ${summary['total_value']:.2f}",
            f"Portfolio Return: {summary['total_return_pct']:.2f}%",
            "",
            "Positions:"
        ]

        if summary['positions']:
            for pos in summary['positions']:
                if pos['shares'] > 0:
                    lines.append(
                        f"  {pos['symbol']}: {pos['shares']:.4f} shares @ ${pos['avg_cost']:.2f} "
                        f"(Current: ${pos['current_price']:.2f}, P/L: ${pos['profit_loss']:.2f})"
                    )
        else:
            lines.append("  None")

        return "\n".join(lines)

    def _update_market_volatility(self, stocks: List[Dict[str, Any]]) -> None:
        """
        Update market volatility metric.

        Args:
            stocks: List of stock data dictionaries
        """
        # Calculate average absolute price change
        changes = [abs(stock.get('change', 0)) for stock in stocks]
        if changes:
            current_volatility = sum(changes) / len(changes)

            # Exponential moving average
            alpha = 0.1
            self.market_volatility = alpha * current_volatility + (1 - alpha) * self.market_volatility

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response.

        Args:
            response: LLM response string potentially containing JSON

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None

    def _execute_autonomous_trades(self, llm_response: str, stocks: List[Dict[str, Any]]) -> None:
        """
        Execute trades from LLM response.

        Handles both initial decisions and revised decisions from self-reflection.

        Args:
            llm_response: LLM response containing trading decisions
            stocks: List of stock data dictionaries for price lookup
        """
        # Create stock price lookup
        stock_prices = {stock['symbol']: stock for stock in stocks}

        # Find all DECISION lines (including REVISED_DECISION)
        decision_pattern = r'(?:REVISED_)?DECISION:\s*(BUY|SELL|HOLD)\s+([A-Z]+)\s*\$?(\d+(?:\.\d+)?)?'
        decisions = re.findall(decision_pattern, llm_response, re.IGNORECASE)

        if not decisions:
            logger.debug("No trading decisions found in LLM response")
            return

        logger.info(f"Found {len(decisions)} trading decision(s)")

        for action, symbol, amount_str in decisions:
            action = action.upper()
            symbol = symbol.upper()

            # Validate symbol
            if symbol not in stock_prices:
                logger.warning(f"Invalid symbol: {symbol}")
                continue

            if action == 'HOLD':
                logger.info(f"Decision: HOLD {symbol}")
                continue

            # Parse amount
            if not amount_str:
                logger.warning(f"No amount specified for {action} {symbol}")
                continue

            amount = float(amount_str)
            stock_data = stock_prices[symbol]
            price = stock_data['price']
            timestamp = stock_data.get('timestamp', '')

            # Execute trade
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

                # Track decision for learning
                self.recent_decisions.append({
                    'timestamp': timestamp,
                    'action': action,
                    'symbol': symbol,
                    'amount': amount,
                    'price': price,
                    'result': result
                })
            else:
                logger.warning(f"✗ Trade failed: {action} ${amount} of {symbol} - {result.get('reason', 'Unknown')}")

    def reset(self) -> None:
        """
        Reset strategy state.

        Clears all internal state including data buffer, decisions, and memory.
        """
        super().reset()
        self.data_points_since_check = 0
        self.data_buffer = []
        self.recent_decisions = []
        self.market_volatility = 0.0
        if self.memory_manager:
            self.memory_manager.clear()
        logger.info("Autonomous strategy reset")

