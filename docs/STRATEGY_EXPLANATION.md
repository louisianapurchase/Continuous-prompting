# Continuous Prompting Strategy - Detailed Explanation

## Current Implementation

### How It Works

The current "continuous prompting" strategy (`src/strategies/continuous_strategy.py`) works like this:

1. **Data arrives** (every 1 second by default)
2. **Add to batch** - Accumulates data points until `batch_size` is reached
3. **Build prompt** - Creates a NEW prompt containing:
   - Current data point
   - Limited history (last 10 points by default)
4. **Send to LLM** - Makes a fresh API call to Ollama
5. **Get response** - LLM responds based on this single prompt
6. **Repeat** - Process continues for next batch

### Code Flow

```python
# Simplified version
def process_data_point(data):
    # Add to batch
    current_batch.append(data)
    
    # When batch is full
    if len(current_batch) >= batch_size:
        # Build prompt with current data + history
        prompt = f"""
        Current data: {current_batch[-1]}
        Recent history: {last_10_points}
        
        What do you observe?
        """
        
        # Send to LLM (NEW conversation each time!)
        response = llm.chat(prompt, maintain_history=False)
        
        # Clear batch and repeat
        current_batch = []
        return response
```

## The Problem

### This is NOT True Continuous Prompting

**What it actually is:** Rapid one-shot prompting

**Why it's problematic:**

1. **No Persistent Memory**
   - Each prompt is independent
   - LLM doesn't "remember" previous responses
   - We manually inject history into each prompt

2. **Token Limit Constraints**
   - Can only include ~10 historical points
   - Older data is lost forever
   - Context window fills up quickly

3. **Inefficient**
   - Repeating context in every prompt
   - Wasting tokens on redundant information
   - High API costs for long streams

4. **No Learning**
   - LLM doesn't adapt to patterns
   - Can't build understanding over time
   - Each response is isolated

### Example of the Issue

```
Data Point 1: AAPL $150 (+2%)
Prompt: "AAPL is at $150, up 2%. What do you think?"
Response: "Positive momentum"

Data Point 2: AAPL $152 (+1.3%)
Prompt: "AAPL is at $152, up 1.3%. History: [Point 1]. What do you think?"
Response: "Continuing upward trend"

Data Point 100: AAPL $180 (+0.5%)
Prompt: "AAPL is at $180, up 0.5%. History: [Points 90-99]. What do you think?"
Response: "Slight gain"  # Lost all context from points 1-89!
```

## Better Approaches to Explore

### 1. Conversation-Based Approach (Easiest)

**Idea:** Use Ollama's conversation history feature

```python
# Instead of maintain_history=False
response = llm.chat(
    message=f"New data: {data}",
    maintain_history=True  # Let Ollama manage conversation
)
```

**Pros:**
- Simple to implement
- LLM maintains context
- Natural conversation flow

**Cons:**
- Still hits token limits eventually
- Need to manage conversation truncation
- No long-term memory

**Implementation:**
```python
class ConversationStrategy(BaseStrategy):
    def process_data_point(self, data):
        # Just send new data, LLM remembers previous
        prompt = f"New data point: {data['symbol']} ${data['price']}"
        response = self.llm_client.chat(
            message=prompt,
            maintain_history=True  # Key difference!
        )
        
        # Periodically summarize and reset
        if len(self.llm_client.conversation_history) > 50:
            summary = self.llm_client.chat("Summarize key patterns so far")
            self.llm_client.clear_history()
            self.llm_client.chat(f"Context: {summary}")
        
        return response
```

### 2. Vector Database + RAG (Recommended)

**Idea:** Store all data in vector DB, retrieve relevant context

```python
# Pseudocode
class VectorRAGStrategy(BaseStrategy):
    def __init__(self):
        self.vector_db = ChromaDB()  # or Pinecone, Weaviate
    
    def process_data_point(self, data):
        # Store data point
        self.vector_db.add(
            text=f"{data['symbol']} ${data['price']} {data['change']}%",
            metadata=data
        )
        
        # Retrieve relevant context
        query = f"Recent patterns for {data['symbol']}"
        relevant_history = self.vector_db.query(query, top_k=5)
        
        # Build prompt with relevant context only
        prompt = f"""
        Current: {data}
        Relevant history: {relevant_history}
        Analysis?
        """
        
        response = self.llm_client.chat(prompt)
        return response
```

**Pros:**
- Unlimited history storage
- Retrieves only relevant context
- Efficient token usage
- Scales to long streams

**Cons:**
- Requires vector DB setup
- More complex architecture
- Embedding costs

**Tools:**
- ChromaDB (local, free)
- Pinecone (cloud, paid)
- Weaviate (self-hosted)
- FAISS (local, free)

### 3. Sliding Window + Summarization

**Idea:** Maintain a sliding window, periodically summarize

```python
class SlidingWindowStrategy(BaseStrategy):
    def __init__(self):
        self.window_size = 50
        self.summary = ""
    
    def process_data_point(self, data):
        self.data_history.append(data)
        
        # When window is full
        if len(self.data_history) >= self.window_size:
            # Summarize old data
            old_data = self.data_history[:25]
            summary_prompt = f"Summarize key patterns: {old_data}"
            new_summary = self.llm_client.chat(summary_prompt)
            
            # Update summary
            self.summary = f"{self.summary}\n{new_summary}"
            
            # Keep only recent data
            self.data_history = self.data_history[25:]
        
        # Prompt with summary + recent data
        prompt = f"""
        Summary of earlier data: {self.summary}
        Recent data: {self.data_history[-10:]}
        Current: {data}
        Analysis?
        """
        
        return self.llm_client.chat(prompt)
```

**Pros:**
- No external dependencies
- Compresses history
- Maintains long-term context

**Cons:**
- Summarization costs tokens
- May lose important details
- Summary quality varies

### 4. Fine-Tuning Approach (Advanced)

**Idea:** Continuously fine-tune model on incoming data

```python
class FineTuningStrategy(BaseStrategy):
    def __init__(self):
        self.training_buffer = []
        self.epoch_size = 100
        self.model_version = 0
    
    def process_data_point(self, data):
        # Add to training buffer
        self.training_buffer.append({
            'input': f"Analyze: {data}",
            'output': "Expected analysis pattern"
        })
        
        # When epoch is complete
        if len(self.training_buffer) >= self.epoch_size:
            # Fine-tune model
            self.fine_tune_model(self.training_buffer)
            self.model_version += 1
            self.training_buffer = []
        
        # Use current model version
        response = self.llm_client.chat(
            f"Analyze: {data}",
            model=f"custom-model-v{self.model_version}"
        )
        
        return response
```

**Pros:**
- Model actually learns
- Adapts to data patterns
- No context in prompts needed

**Cons:**
- Requires training infrastructure
- Slow (training takes time)
- Complex to implement
- May need labeled data

## Recommended Next Steps

### For Quick Improvement:
1. **Switch to conversation mode** (`maintain_history=True`)
2. **Add periodic summarization** when history gets long
3. **Implement conversation reset** with context preservation

### For Production:
1. **Set up ChromaDB** for vector storage
2. **Implement RAG pattern** for context retrieval
3. **Add embedding generation** for data points
4. **Build relevance-based retrieval**

### For Research:
1. **Experiment with fine-tuning** on small batches
2. **Compare token usage** across approaches
3. **Measure response quality** over time
4. **Test different window sizes**

## Code Examples

### Quick Fix: Enable Conversation History

```python
# In src/strategies/continuous_strategy.py, line 96-100
# Change from:
response = self.llm_client.chat(
    message=prompt,
    system_prompt=system_prompt,
    maintain_history=False,  # Currently False!
)

# To:
response = self.llm_client.chat(
    message=prompt,
    system_prompt=system_prompt,
    maintain_history=True,  # Enable conversation memory
)

# Add periodic reset:
if len(self.llm_client.conversation_history) > 50:
    summary = self.llm_client.chat("Summarize key insights so far")
    self.llm_client.clear_history()
    self.llm_client.chat(f"Previous context: {summary}")
```

### Add Vector DB (ChromaDB)

```bash
# Install
pip install chromadb

# In your strategy:
import chromadb

class VectorStrategy(BaseStrategy):
    def __init__(self, llm_client, prompt_manager, config=None):
        super().__init__(llm_client, prompt_manager, config)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("trading_data")
    
    def process_data_point(self, data):
        # Store in vector DB
        self.collection.add(
            documents=[f"{data['symbol']} ${data['price']} {data['change']}%"],
            metadatas=[data],
            ids=[str(len(self.data_history))]
        )
        
        # Query for relevant context
        results = self.collection.query(
            query_texts=[f"patterns for {data['symbol']}"],
            n_results=5
        )
        
        # Build prompt with relevant context
        prompt = f"""
        Current: {data}
        Relevant history: {results['documents']}
        What patterns do you see?
        """
        
        return self.llm_client.chat(prompt)
```

## Summary

**Current State:** Basic batching with limited history injection  
**Problem:** Not true continuous prompting, token limits, no memory  
**Quick Fix:** Enable conversation history + summarization  
**Best Solution:** Vector DB + RAG for unlimited, relevant context  
**Advanced:** Fine-tuning for true learning

The framework gives you the infrastructure - now experiment with these approaches!

