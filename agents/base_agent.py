from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class MCPMessage:
    """Model Context Protocol message format."""
    sender: str
    receiver: str
    message_type: str
    trace_id: str
    payload: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'message_type': self.message_type,
            'trace_id': self.trace_id,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Create message from JSON string."""
        return cls.from_dict(json.loads(json_str))

class MCPClient:
    """Base MCP client for sending and receiving messages"""
    def __init__(self):
        self.message_handlers = {}
    
    def register_handler(self, message_type: str, handler):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def send_message(self, message: MCPMessage) -> None:
        """Send a message to the appropriate handler"""
        if message.message_type in self.message_handlers:
            return await self.message_handlers[message.message_type](message)
        else:
            print(f"No handler registered for message type: {message.message_type}")
            return None

class BaseAgent(ABC, MCPClient):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, mcp_server):
        """
        Initialize the base agent
        
        Args:
            agent_id: Unique identifier for the agent
            mcp_server: Reference to the MCP server for message routing
        """
        super().__init__()
        self.agent_id = agent_id
        self.mcp_server = mcp_server
        self.setup_handlers()
        
        # Register with MCP server
        self.mcp_server.register_client(agent_id, self)
    
    @abstractmethod
    def setup_handlers(self):
        """Set up message handlers for this agent"""
        pass
    
    async def send_message(self, receiver_id: str, message_type: MessageType, payload: Dict[str, Any]):
        """
        Send a message to another agent
        
        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of the message
            payload: Message payload
        """
        message = Message(
            sender=self.agent_id,
            receiver=receiver_id,
            message_type=message_type,
            payload=payload
        )
        await self.mcp_server.route_message(message)
    
    async def handle_error(self, error: Exception, trace_id: Optional[str] = None):
        """Handle errors by sending an error message"""
        error_payload = {
            "error": str(error),
            "trace_id": trace_id or ""
        }
        await self.send_message(
            receiver_id="coordinator",  # Assuming we have a coordinator
            message_type=MessageType.ERROR,
            payload=error_payload
        )
    
    async def start(self):
        """Start the agent (can be overridden by subclasses)"""
        pass
    
    async def stop(self):
        """Stop the agent (can be overridden by subclasses)"""
        pass
