"""Ollama LLM client for continuous prompting."""

import logging
from typing import Dict, Any, List, Optional
import json

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama LLM.
    
    Provides methods for sending prompts and managing conversations
    in a continuous prompting context.
    """
    
    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        num_gpu: int = 0,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            num_gpu: Number of GPU layers to use (0 for CPU only)
        """
        if ollama is None:
            raise ImportError(
                "ollama-python package not installed. "
                "Install with: pip install ollama-python"
            )

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_gpu = num_gpu
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Verify model is available
        self._verify_model()
        
        logger.info(f"Ollama client initialized with model: {model}")
    
    def _verify_model(self) -> None:
        """Verify that the specified model is available."""
        try:
            # Try to list models to verify connection
            models = ollama.list()
            if isinstance(models, dict) and 'models' in models:
                model_list = models['models']
                model_names = []
                for m in model_list:
                    if isinstance(m, dict):
                        model_names.append(m.get('name', m.get('model', str(m))))
                    else:
                        model_names.append(str(m))

                if not any(self.model in name for name in model_names):
                    logger.warning(
                        f"Model '{self.model}' not found in available models: {model_names}. "
                        f"You may need to run: ollama pull {self.model}"
                    )
            else:
                logger.info(f"Using model: {self.model}")
        except Exception as e:
            logger.warning(f"Could not verify model availability: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # Add user prompt
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            # Generate response
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'num_gpu': self.num_gpu,
                    'num_thread': 4,  # Limit CPU threads to avoid memory issues
                    'num_ctx': 2048,  # Context window size - smaller = less VRAM usage
                },
                stream=stream,
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    content = chunk.get('message', {}).get('content', '')
                    full_response += content
                    print(content, end='', flush=True)
                print()  # New line after streaming
                return full_response
            else:
                # Handle non-streaming response
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        maintain_history: bool = True,
    ) -> str:
        """
        Send a message in a conversational context.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            maintain_history: Whether to maintain conversation history
            
        Returns:
            LLM response
        """
        # Build messages with history
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add conversation history
        if maintain_history:
            messages.extend(self.conversation_history)
        
        # Add current message
        messages.append({
            'role': 'user',
            'content': message
        })
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'num_gpu': self.num_gpu,
                    'num_thread': 4,  # Limit CPU threads to avoid memory issues
                    'num_ctx': 2048,  # Context window size - smaller = less VRAM usage
                }
            )
            
            response_text = response['message']['content']
            
            # Update conversation history
            if maintain_history:
                self.conversation_history.append({
                    'role': 'user',
                    'content': message
                })
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response_text
                })
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {str(e)}"
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def truncate_history(self, max_messages: int = 10) -> None:
        """
        Truncate conversation history to most recent messages.
        
        Args:
            max_messages: Maximum number of message pairs to keep
        """
        if len(self.conversation_history) > max_messages * 2:
            self.conversation_history = self.conversation_history[-(max_messages * 2):]
            logger.info(f"Conversation history truncated to {max_messages} message pairs.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            return {
                'model': self.model,
                'base_url': self.base_url,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

