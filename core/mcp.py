from typing import Dict, Any, Optional, List, TypeVar, Generic, Type
from pydantic import BaseModel, Field
from enum import Enum
from uuid import uuid4

class MessageType(str, Enum):
    INGESTION_REQUEST = "INGESTION_REQUEST"
    INGESTION_RESPONSE = "INGESTION_RESPONSE"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESPONSE = "RETRIEVAL_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    ERROR = "ERROR"

class Message(BaseModel):
    """Base message class for MCP communication"""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    message_type: MessageType
    sender: str
    receiver: str
    timestamp: float = Field(default_factory=lambda: datetime.datetime.now().timestamp())
    payload: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class MCPClient:
    """Base MCP client for sending and receiving messages"""
    def __init__(self):
        self.message_handlers = {}
    
    def register_handler(self, message_type: MessageType, handler):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def send_message(self, message: Message) -> None:
        """Send a message to the appropriate handler"""
        if message.message_type in self.message_handlers:
            await self.message_handlers[message.message_type](message)
        else:
            print(f"No handler registered for message type: {message.message_type}")
    
    async def receive_message(self, message: Message) -> None:
        """Receive and process a message"""
        await self.send_message(message)

class MCPServer:
    """MCP Server for message routing between agents"""
    def __init__(self):
        self.clients = {}
    
    def register_client(self, client_id: str, client: MCPClient):
        """Register a client with the server"""
        self.clients[client_id] = client
    
    async def route_message(self, message: Message):
        """Route a message to the appropriate client"""
        if message.receiver in self.clients:
            await self.clients[message.receiver].receive_message(message)
        else:
            print(f"No client found with ID: {message.receiver}")
