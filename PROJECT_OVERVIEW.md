# Continuous Prompting Framework - Project Overview

## Purpose

This project addresses a fundamental challenge in LLM applications: **how to effectively prompt Large Language Models with continuously streaming data**. Traditional LLM prompting is one-shot—you send a prompt, receive a response, and the interaction ends. However, many real-world applications involve continuous data streams (market data, sensor readings, log files, social media feeds) where the LLM must maintain context across thousands of data points while operating within strict token limits.

We built this framework as an experimental testbed using financial market data as the streaming source, creating a realistic environment where an LLM must observe, remember, reason, and act on continuous information flow.

## Why This Approach Works

The framework succeeds by combining several complementary techniques that address different aspects of the continuous prompting challenge:

**Autonomous Activation with Chain-of-Thought Reasoning**: Rather than using hardcoded triggers (e.g., "activate when price changes >1.5%"), we let the LLM decide when to activate itself. The system periodically asks the LLM: "Should you analyze this data?" The LLM uses Chain-of-Thought (CoT) prompting to reason through what it observes, explaining its thinking step-by-step before making decisions. This produces more reliable decisions because the LLM must articulate its reasoning, catching logical errors through self-explanation.

**Sliding Window Memory with Statistical Summarization**: To manage token limits, we maintain recent data in full detail while summarizing older data into compact statistics. This balances the need for recent granularity with historical context. The sliding window approach is deterministic and predictable—you always know exactly how many tokens you're using—making it ideal for production environments where token costs matter.

**ReAct Pattern for Decision Making**: We implement the ReAct (Reasoning + Acting) pattern where the LLM alternates between reasoning about the market state and taking actions (buy/sell/hold). This structured approach prevents impulsive decisions and ensures the LLM considers multiple factors before acting.

**Self-Reflection and Confidence Scoring**: After making a decision, the LLM critiques its own reasoning and assigns a confidence score. Trades only execute when confidence exceeds a threshold (default 70%). This meta-cognitive layer catches mistakes and prevents low-quality decisions from affecting the portfolio.

**News Event Injection**: We inject simulated news events (1-2 per day) that the LLM must interpret and incorporate into its decision-making. This tests the LLM's ability to integrate qualitative information with quantitative market data, mimicking real-world scenarios where multiple information sources must be synthesized.

## Innovation and Contribution

The framework demonstrates that LLMs can operate autonomously in streaming data environments without hardcoded rules. By combining modern prompting techniques (CoT, ReAct, self-reflection) with practical memory management, we show that LLMs can maintain coherent long-term behavior across thousands of data points. The autonomous activation mechanism is particularly novel—the LLM learns what patterns are significant rather than relying on predetermined thresholds.

The portfolio tracking system provides concrete feedback on decision quality. Unlike abstract benchmarks, we can measure whether the LLM's trading decisions actually work, creating a tight feedback loop between reasoning quality and measurable outcomes.

This work contributes a reusable architecture for continuous prompting that extends beyond trading to any domain with streaming data: system monitoring, real-time analytics, IoT sensor processing, or live content moderation. The techniques are domain-agnostic and the codebase is designed for experimentation and extension.

