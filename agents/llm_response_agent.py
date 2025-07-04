import os
from typing import Dict, Any, List, Optional, Union, Callable, Literal
import json
import requests
from enum import Enum

from ..core.mcp import Message, MessageType
from .base_agent import BaseAgent

class LLMProvider(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"

class LLMClient:
    """Wrapper class for different LLM providers"""
    
    def __init__(
        self, 
        provider: str = "openai", 
        model_name: str = None, 
        **kwargs
    ):
        """
        Initialize the LLM client
        
        Args:
            provider: LLM provider (openai, huggingface, ollama, local)
            model_name: Name of the model to use
            **kwargs: Additional provider-specific arguments
        """
        self.provider = LLMProvider(provider.lower())
        self.model_name = model_name or self._get_default_model()
        self.client = self._initialize_client(**kwargs)
    
    def _get_default_model(self) -> str:
        """Get default model name based on provider"""
        defaults = {
            LLMProvider.OPENAI: "gpt-3.5-turbo",
            LLMProvider.HUGGINGFACE: "meta-llama/Llama-2-7b-chat-hf",
            LLMProvider.OLLAMA: "llama2",
            LLMProvider.LOCAL: "TheBloke/Llama-2-7B-Chat-GGUF"
        }
        return defaults.get(self.provider, "gpt-3.5-turbo")
    
    def _initialize_client(self, **kwargs):
        """Initialize the appropriate LLM client"""
        try:
            if self.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required")
                return OpenAI(api_key=api_key)
            
            elif self.provider == LLMProvider.HUGGINGFACE:
                api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not api_key:
                    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is required")
                return {"api_key": api_key}
            
            elif self.provider == LLMProvider.OLLAMA:
                # Test if Ollama is running
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    response.raise_for_status()
                except Exception as e:
                    raise ConnectionError("Ollama server is not running. Please start it with 'ollama serve'") from e
                return None  # We'll use direct HTTP requests for Ollama
            
            elif self.provider == LLMProvider.LOCAL:
                # For local models, you might use llama-cpp-python or similar
                try:
                    from llama_cpp import Llama
                    return Llama(
                        model_path=os.path.join("models", f"{self.model_name}.bin"),
                        n_ctx=2048,
                        n_threads=4
                    )
                except ImportError:
                    raise ImportError("llama-cpp-python is required for local models. Install with: pip install llama-cpp-python")
            
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
        except ImportError as e:
            raise ImportError(f"Failed to initialize {self.provider} client. Make sure you have installed the required dependencies.") from e
    
    async def generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate a response using the configured LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.provider == LLMProvider.OPENAI:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        
        elif self.provider == LLMProvider.HUGGINGFACE:
            headers = {
                "Authorization": f"Bearer {self.client['api_key']}",
                "Content-Type": "application/json"
            }
            
            # Format messages for the Hugging Face API
            prompt = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant: "
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False,
                    **kwargs
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model_name}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        
        elif self.provider == LLMProvider.OLLAMA:
            # Format messages for Ollama API
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            data = {
                "model": self.model_name,
                "messages": formatted_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=data
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        
        elif self.provider == LLMProvider.LOCAL:
            # Format messages for local model
            prompt = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant: "
            
            response = self.client(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response["choices"][0]["text"]
        
        else:
            raise ValueError(f"Generation not implemented for provider: {self.provider}")

class LLMResponseAgent(BaseAgent):
    """Agent responsible for generating responses using LLM"""
    
    def __init__(
        self, 
        mcp_server, 
        provider: str = "openai",
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the LLM Response Agent
        
        Args:
            mcp_server: Reference to the MCP server
            provider: LLM provider (openai, huggingface, ollama, local)
            model_name: Name of the LLM model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__("llm_response_agent", mcp_server)
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = float(os.getenv("LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", max_tokens))
        
        # Initialize the LLM client
        self.llm_client = LLMClient(
            provider=self.provider,
            model_name=self.model_name
        )
        
        # Store conversation history
        self.conversation_history = {}
    
    def setup_handlers(self):
        """Set up message handlers for this agent"""
        self.register_handler(MessageType.LLM_REQUEST, self.handle_llm_request)
    
    async def handle_llm_request(self, message: Message):
        """
        Handle LLM generation requests
        
        Args:
            message: MCP message containing LLM request
        """
        try:
            payload = message.payload
            query = payload.get("query")
            context = payload.get("context", [])
            conversation_id = payload.get("conversation_id", "default")
            
            if not query:
                raise ValueError("No query provided in LLM request")
            
            # Get or initialize conversation history
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            # Format the context into a single string
            context_str = "\n\n".join([
                f"[Source: {chunk.get('metadata', {}).get('source', 'unknown')}]\n{chunk['text']}"
                for chunk in context
            ])
            
            # Prepare the messages for the LLM
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            If the context doesn't contain the answer, say "I don't have enough information to answer that question."
            Be concise and accurate in your responses."""
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history
            for msg in self.conversation_history[conversation_id][-4:]:  # Keep last 4 exchanges
                messages.append({"role": "user", "content": msg["query"]})
                if "response" in msg:
                    messages.append({"role": "assistant", "content": msg["response"]})
            
            # Add current query and context
            if context_str:
                user_message = f"""Context:
                {context_str}
                
                Question: {query}
                
                Answer based on the context above. If the context doesn't contain the answer, say "I don't have enough information to answer that question."
                """
            else:
                user_message = query
            
            messages.append({"role": "user", "content": user_message})
            
            # Call the LLM
            response_text = await asyncio.to_thread(
                self.llm_client.generate,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Update conversation history
            self.conversation_history[conversation_id].append({
                "query": query,
                "response": response_text,
                "context": context
            })
            
            # Send the response back
            await self.send_message(
                receiver_id=message.sender,
                message_type=MessageType.LLM_RESPONSE,
                payload={
                    "status": "success",
                    "query": query,
                    "response": response_text,
                    "sources": [chunk.get("metadata", {}).get("source") for chunk in context],
                    "trace_id": message.trace_id
                }
            )
            
        except Exception as e:
            print(f"Error in LLM response agent: {e}")
            await self.handle_error(e, message.trace_id)
    
    def clear_conversation(self, conversation_id: str = "default"):
        """Clear the conversation history for a given conversation ID"""
        if conversation_id in self.conversation_history:
            self.conversation_history[conversation_id] = []
