"""ChromaDB-based vector memory manager with RAG retrieval."""

import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from .base_memory import BaseMemoryManager

logger = logging.getLogger(__name__)


class ChromaMemoryManager(BaseMemoryManager):
    """
    Vector database memory manager using ChromaDB.
    
    Stores all data points and responses as embeddings, retrieves
    only the most relevant context based on semantic similarity.
    
    Advantages:
    - Unlimited history storage
    - Efficient token usage (only relevant context)
    - Semantic search for similar patterns
    - Persistent storage across sessions
    
    Best for: Production systems, long-running experiments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaDB memory manager.
        
        Args:
            config: Configuration with options:
                - collection_name: Name for the ChromaDB collection
                - persist_directory: Directory to persist the database
                - top_k: Number of similar items to retrieve (default: 5)
                - embedding_model: Model for embeddings (default: all-MiniLM-L6-v2)
        """
        super().__init__(config)
        
        if chromadb is None:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb sentence-transformers"
            )
        
        # Configuration
        self.collection_name = self.config.get('collection_name', 'trading_data')
        self.persist_directory = self.config.get('persist_directory', './data/chroma')
        self.top_k = self.config.get('top_k', 5)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False,
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Trading data and LLM responses"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        self.item_count = 0
    
    def add_data_point(self, data: Dict[str, Any], response: Optional[str] = None) -> None:
        """
        Add data point and response to vector database.
        
        Args:
            data: Data point to store
            response: LLM response to store
        """
        # Create document text for embedding
        doc_text = self.format_data_point(data)
        if response:
            doc_text += f"\nLLM Analysis: {response}"
        
        # Create metadata
        metadata = {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'symbol': data.get('symbol', 'N/A'),
            'price': float(data.get('price', 0)),
            'change': float(data.get('change', 0)),
            'has_response': response is not None,
        }
        
        # Add to collection
        self.collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[f"item_{self.item_count}"]
        )
        
        self.item_count += 1
        
        # Also keep in conversation history for stats
        self.conversation_history.append({
            'data': data,
            'response': response,
            'timestamp': metadata['timestamp']
        })
        
        logger.debug(f"Added item {self.item_count} to ChromaDB")
    
    def get_context(self, current_data: Dict[str, Any], max_tokens: int = 2000) -> str:
        """
        Retrieve relevant context using semantic search.
        
        Args:
            current_data: Current data point
            max_tokens: Maximum tokens for context (approximate)
            
        Returns:
            Formatted context with most relevant historical data
        """
        if self.item_count == 0:
            return "No historical data available."
        
        # Create query from current data
        query_text = self.format_data_point(current_data)
        
        # Retrieve similar items
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(self.top_k, self.item_count)
        )
        
        # Format context
        context_parts = ["Recent relevant data:"]
        
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                # Rough token estimation (4 chars ≈ 1 token)
                if len('\n'.join(context_parts)) * 0.25 > max_tokens:
                    break
                context_parts.append(f"{i+1}. {doc}")
        
        return '\n'.join(context_parts)
    
    def clear(self) -> None:
        """Clear all stored memory."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Trading data and LLM responses"}
            )
            self.conversation_history = []
            self.item_count = 0
            logger.info("Cleared ChromaDB memory")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_items': self.item_count,
            'memory_type': 'ChromaDB Vector Storage',
            'collection_name': self.collection_name,
            'top_k_retrieval': self.top_k,
            'persist_directory': self.persist_directory,
        }
    
    def search_by_symbol(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for data points by symbol.
        
        Args:
            symbol: Stock symbol to search for
            limit: Maximum number of results
            
        Returns:
            List of matching data points
        """
        results = self.collection.get(
            where={"symbol": symbol},
            limit=limit
        )
        
        return results
    
    def search_by_price_range(self, min_price: float, max_price: float, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for data points within a price range.
        
        Args:
            min_price: Minimum price
            max_price: Maximum price
            limit: Maximum number of results
            
        Returns:
            List of matching data points
        """
        results = self.collection.get(
            where={
                "$and": [
                    {"price": {"$gte": min_price}},
                    {"price": {"$lte": max_price}}
                ]
            },
            limit=limit
        )
        
        return results

