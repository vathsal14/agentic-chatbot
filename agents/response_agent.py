from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from agents.base_agent import BaseAgent, MCPMessage  
import logging
class ResponseAgent(BaseAgent):
    """Generates responses using an LLM"""
    def __init__(self, mcp_server=None, model_name: str = 'distilgpt2'):
        """
        Initialize the Response Agent
        Args:
            mcp_server: Reference to the MCP server for message routing
            model_name: Name of the Hugging Face model to use
        """
        super().__init__("response_agent", mcp_server)
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.setup_handlers()
    def setup_handlers(self):
        """Set up message handlers for this agent"""
        self.register_handler("GENERATE_RESPONSE", self.handle_generate_response)
    def _load_model(self):
        """Lazily load the model when needed"""
        if self.model is None or self.tokenizer is None:
            try:
                self.logger.info(f"Loading tokenizer for {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.logger.info(f"Loading model {self.model_name}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info(f"Model loaded on {self.device}")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                raise
    def _format_context(self, context: Any) -> str:
        """Format the context for better model understanding"""
        if not context:
            return "No specific context is available for this query."
        if isinstance(context, str):
            return context[:2000] + ('' if len(context) <= 2000 else '...')
        if isinstance(context, list):
            formatted = []
            for i, item in enumerate(context[:3], 1):  
                if isinstance(item, dict):
                    text = item.get('text', str(item))
                    source = item.get('source', 'unknown source')
                    formatted.append(f"Context {i} (from {source}): {text[:500]}" + ('' if len(text) <= 500 else '...'))
                else:
                    formatted.append(f"Context {i}: {str(item)[:500]}" + ('' if len(str(item)) <= 500 else '...'))
            return '\n\n'.join(formatted)
        return str(context)[:2000] + ('' if len(str(context)) <= 2000 else '...')
    def _create_prompt(self, query: str, context: Any) -> str:
        """Create a well-structured prompt for the model"""
        formatted_context = self._format_context(context)
        return (
            "You are a helpful AI assistant. Use the following context to answer the user's question. "
            "If you don't know the answer, say you don't know.\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )
    async def handle_generate_response(self, message: MCPMessage) -> None:
        """Generate a response to the user's query with improved context handling"""
        try:
            self.logger.info(f"Processing query: {message.payload.get('query')}")
            query = message.payload.get('query', '').strip()
            if not query:
                raise ValueError("No query provided in the message payload")
            context = message.payload.get('context', '')
            max_length = min(int(message.payload.get('max_length', 500)), 1000)  
            prompt = self._create_prompt(query, context)
            self.logger.debug(f"Prompt: {prompt}")
            self._load_model()
            self.logger.info("Generating response...")
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.split('ANSWER:')[-1].strip()
            response = response.split('\n')[0].strip()  
            if not response or response.isspace():
                response = "I'm not sure how to respond to that. Could you provide more details?"
            self.logger.info(f"Response generated: {response}")
            response_message = MCPMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="RESPONSE_GENERATED",
                trace_id=message.trace_id,
                payload={
                    'query': query,
                    'response': response,
                    'context_used': context if isinstance(context, str) else str(context)[:500] + ('...' if len(str(context)) > 500 else ''),
                    'status': 'success'
                }
            )
            await self.send_message(response_message)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            await self.handle_error(e, message.trace_id)