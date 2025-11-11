"""
Simple example demonstrating the core concepts of the framework.

This is a minimal example showing how to:
1. Create a data source
2. Set up an LLM client
3. Stream data and get LLM responses

Run this to understand the basics before diving into the full framework.
"""

from src.data import TradingDataSimulator, SampleDataSource
from src.llm import OllamaClient, PromptManager

def main():
    print("="*60)
    print("Simple Continuous Prompting Example")
    print("="*60)
    print()
    
    # 1. Create a data source
    print("Creating sample data source...")
    data_source = SampleDataSource(
        symbols=["AAPL", "GOOGL"],
        price_volatility=0.03,  # 3% price swings
    )
    
    # 2. Create a data simulator
    print("Creating data simulator...")
    simulator = TradingDataSimulator(
        data_source=data_source,
        update_interval=2.0,  # 2 seconds between updates
    )
    
    # 3. Create LLM client
    print("Connecting to Ollama...")
    try:
        llm = OllamaClient(
            model="mistral",  # Change to your preferred model
            temperature=0.7,
        )
        print(f"✓ Connected to Ollama with model: mistral")
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed and running")
        print("2. You have pulled a model: ollama pull mistral")
        return
    
    # 4. Create prompt manager
    prompt_manager = PromptManager()
    
    print()
    print("="*60)
    print("Starting data stream (will process 5 data points)...")
    print("="*60)
    print()
    
    # 5. Stream data and get LLM responses
    history = []
    count = 0
    max_iterations = 5
    
    try:
        for data_point in simulator.stream():
            count += 1
            
            # Display the data
            print(f"\n--- Data Point {count} ---")
            print(f"Symbol: {data_point['symbol']}")
            print(f"Price: ${data_point['price']}")
            print(f"Change: {data_point['change']:+.2f}%")
            print(f"Volume: {data_point['volume']:,}")
            
            # Build a simple prompt
            prompt = f"""
New market data just arrived:
- Symbol: {data_point['symbol']}
- Price: ${data_point['price']}
- Change: {data_point['change']:+.2f}%
- Volume: {data_point['volume']:,}

Based on this data point, what do you observe? Keep your response to 1-2 sentences.
"""
            
            # Get LLM response
            print("\n🤖 LLM Response:")
            response = llm.generate(
                prompt=prompt,
                system_prompt="You are a concise trading analyst. Provide brief observations.",
            )
            print(response)
            
            # Add to history
            history.append(data_point)
            
            # Stop after max iterations
            if count >= max_iterations:
                break
        
        simulator.stop()
        
        print("\n" + "="*60)
        print("Example completed!")
        print("="*60)
        print(f"\nProcessed {count} data points")
        print("\nNext steps:")
        print("1. Run the full framework: python main.py --max-iterations 10")
        print("2. Customize config.yaml to experiment with different strategies")
        print("3. Read QUICKSTART.md for more examples")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        simulator.stop()
    except Exception as e:
        print(f"\n\nError: {e}")
        simulator.stop()


if __name__ == "__main__":
    main()

