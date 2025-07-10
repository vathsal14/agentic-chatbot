from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Type, Union
from dataclasses import dataclass
from datetime import datetime
import json
import uuid
import logging
from pathlib import Path
T = TypeVar('T')
from core.mcp import Message, MessageType
MessageHandler = Callable[[Message], Any]
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
class BaseAgent(ABC):
    """Base class for all agents in the system."""
    def __init__(self, agent_id: str, mcp_server: Any):
        """
        Initialize the base agent.
        Args:
            agent_id: Unique identifier for the agent.
            mcp_server: Reference to the MCP server for message routing.
        """
        self.agent_id = agent_id
        self.mcp_server = mcp_server
        self._message_handlers = {}
        if mcp_server:
            mcp_server.register_client(agent_id, self)
        self.setup_handlers()
    def register_handler(self, message_type: MessageType, handler: callable) -> None:
        """
        Register a message handler for a specific message type.
        Args:
            message_type: The type of message to handle.
            handler: The handler function to call when a message of this type is received.
        """
        self._message_handlers[message_type] = handler
    async def handle_message(self, message: Message) -> Optional[Message]:
        """
        Handle an incoming message by dispatching it to the appropriate handler.
        Args:
            message: The incoming message to handle.
        Returns:
            Optional response message, if any.
        """
        handler = self._message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        return None
    @abstractmethod
    def setup_handlers(self) -> None:
        """
        Set up message handlers for this agent.
        Subclasses should implement this method to register their message handlers
        using self.register_handler().
        """
        pass
    async def send_message(
        self, 
        receiver_id: str, 
        message_type: Union[MessageType, str], 
        payload: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> None:
        """
        Send a message to another agent.
        Args:
            receiver_id: ID of the receiving agent.
            message_type: Type of the message.
            payload: Optional message payload.
            trace_id: Optional trace ID for request tracking.
        """
        if payload is None:
            payload = {}
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        message = Message(
            sender=self.agent_id,
            receiver=receiver_id,
            message_type=message_type.value if isinstance(message_type, MessageType) else message_type,
            payload=payload,
            trace_id=trace_id
        )
        if self.mcp_server:
            await self.mcp_server.route_message(message)
        else:
            self.logger.warning(f"No MCP server configured, message not sent: {message}")
    async def handle_error(
        self, 
        error: Exception, 
        trace_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle errors by logging and optionally sending an error message.
        Args:
            error: The exception that was raised.
            trace_id: Optional trace ID for request tracking.
            context: Additional context about the error.
        """
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.__class__.__name__)
        error_id = str(uuid.uuid4())
        error_message = str(error)
        self.logger.error(
            f"Error {error_id}: {error_message}",
            exc_info=error,
            extra={
                "trace_id": trace_id,
                "error_id": error_id,
                "error_type": error.__class__.__name__,
                "context": context or {}
            }
        )
        if self.mcp_server and hasattr(self, 'agent_id'):
            try:
                await self.send_message(
                    receiver_id="coordinator",
                    message_type=MessageType.ERROR,
                    payload={
                        "error": error_message,
                        "error_type": error.__class__.__name__,
                        "error_id": error_id,
                        "trace_id": trace_id,
                        "agent_id": self.agent_id,
                        "context": context or {}
                    },
                    trace_id=trace_id
                )
            except Exception as send_error:
                self.logger.error(
                    f"Failed to send error message: {send_error}",
                    exc_info=send_error
                )
    async def start(self) -> None:
        """
        Start the agent.
        Subclasses can override this method to perform any necessary setup
        when the agent starts.
        """
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Starting {self.__class__.__name__} (ID: {getattr(self, 'agent_id', 'unknown')})")
    async def stop(self) -> None:
        """
        Stop the agent.
        Subclasses should override this method to perform any necessary cleanup
        when the agent stops.
        """
        if hasattr(self, 'logger'):
            self.logger.info(f"Stopping {self.__class__.__name__} (ID: {getattr(self, 'agent_id', 'unknown')})")