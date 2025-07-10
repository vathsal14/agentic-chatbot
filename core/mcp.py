from __future__ import annotations
from typing import Dict, Any, Optional, List, TypeVar, Generic, Type, Union, Callable, Awaitable, ClassVar
from pydantic import BaseModel, Field, validator
from enum import Enum, auto
from uuid import uuid4, UUID
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
import json
T = TypeVar('T')
MessageHandler = Callable[['Message'], Awaitable[Optional['Message']]]
MessageTypeValue = Union['MessageType', str]
class MessageType(str, Enum):
    """
    Standard message types for the Model-Controller-Protocol (MCP) system.
    These message types define the standard communication patterns between
    different components of the system. Custom message types can be added
    by creating new enum members or by using string literals.
    """
    PING = "PING"
    PONG = "PONG"
    ERROR = "ERROR"
    AGENT_START = "AGENT_START"
    AGENT_STOP = "AGENT_STOP"
    AGENT_STATUS = "AGENT_STATUS"
    INGESTION_REQUEST = "INGESTION_REQUEST"
    INGESTION_RESPONSE = "INGESTION_RESPONSE"
    DOCUMENT_PROCESSED = "DOCUMENT_PROCESSED"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESPONSE = "RETRIEVAL_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    LLM_STREAM_START = "LLM_STREAM_START"
    LLM_STREAM_TOKEN = "LLM_STREAM_TOKEN"
    LLM_STREAM_END = "LLM_STREAM_END"
    MEMORY_GET = "MEMORY_GET"
    MEMORY_SET = "MEMORY_SET"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    MEMORY_DELETE = "MEMORY_DELETE"
    TOOL_EXECUTE = "TOOL_EXECUTE"
    TOOL_RESULT = "TOOL_RESULT"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    SYSTEM_HEALTH_CHECK = "SYSTEM_HEALTH_CHECK"
    @classmethod
    def is_valid(cls, value: Union['MessageType', str]) -> bool:
        """
        Check if a value is a valid MessageType.
        Args:
            value: The value to check.
        Returns:
            bool: True if the value is a valid MessageType, False otherwise.
        """
        if isinstance(value, MessageType):
            return True
        try:
            cls(value.upper())
            return True
        except ValueError:
            return False
    @classmethod
    def normalize(cls, value: Union['MessageType', str]) -> 'MessageType':
        """
        Convert a string to a MessageType if possible.
        Args:
            value: The value to convert.
        Returns:
            MessageType: The normalized MessageType.
        Raises:
            ValueError: If the value cannot be converted to a MessageType.
        """
        if isinstance(value, MessageType):
            return value
        try:
            return cls(value.upper())
        except ValueError as e:
            raise ValueError(f"Invalid MessageType: {value}") from e
class Message(BaseModel):
    """
    Base message class for Model-Controller-Protocol (MCP) communication.
    This class represents a message that can be sent between different components
    of the system, such as agents, services, or external systems.
    """
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this message"
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Correlation ID for tracing requests across services"
    )
    message_type: Union[MessageType, str] = Field(
        ...,
        description="Type of the message, used for routing and handling"
    )
    sender: str = Field(
        ...,
        description="ID of the sender component"
    )
    receiver: str = Field(
        ...,
        description="ID of the intended recipient component"
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        description="When the message was created"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Primary message content"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context or metadata about the message"
    )
    class Config:
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
            UUID: str,
            Path: str,
        }
        arbitrary_types_allowed = True
        use_enum_values = True
    @validator('message_type', pre=True)
    def validate_message_type(cls, v):
        """Ensure message_type is a valid MessageType or string"""
        if isinstance(v, MessageType):
            return v
        try:
            return MessageType(v.upper())
        except (ValueError, AttributeError):
            return v
    @classmethod
    def create(
        cls,
        message_type: Union[MessageType, str],
        sender: str,
        receiver: str,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        **extra
    ) -> 'Message':
        """
        Create a new message with the given parameters.
        Args:
            message_type: Type of the message
            sender: ID of the sender
            receiver: ID of the recipient
            payload: Message payload (dictionary)
            metadata: Additional metadata
            trace_id: Optional trace ID for request correlation
            **extra: Additional fields to include in the message
        Returns:
            A new Message instance
        """
        return cls(
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            payload=payload or {},
            metadata=metadata or {},
            trace_id=trace_id or str(uuid4()),
            **extra
        )
    def reply(
        self,
        message_type: Optional[Union[MessageType, str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        **extra
    ) -> 'Message':
        """
        Create a reply to this message.
        Args:
            message_type: Type of the reply message (defaults to original type with _RESPONSE suffix)
            payload: Reply payload
            **extra: Additional fields for the reply
        Returns:
            A new Message that is a reply to this one
        """
        if message_type is None:
            try:
                message_type = MessageType(f"{self.message_type}_RESPONSE")
            except ValueError:
                message_type = f"{self.message_type}_RESPONSE"
        return self.create(
            message_type=message_type,
            sender=self.receiver,
            receiver=self.sender,
            payload=payload or {},
            trace_id=self.trace_id,
            **extra
        )
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return self.dict()
    def to_json(self) -> str:
        """Serialize message to JSON string"""
        return self.json()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls.parse_obj(data)
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string"""
        return cls.parse_raw(json_str)
class MCPClient:
    """
    Base MCP client for sending and receiving messages.
    This class provides the core functionality for sending and receiving messages
    in the Model-Controller-Protocol (MCP) system. It handles message routing,
    error handling, and provides hooks for custom message processing.
    """
    def __init__(self, client_id: Optional[str] = None):
        """
        Initialize the MCP client.
        Args:
            client_id: Optional unique identifier for this client. If not provided,
                     a UUID will be generated.
        """
        self.client_id = client_id or f"client_{uuid4().hex[:8]}"
        self._message_handlers = {}
        self._default_handler = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}:{self.client_id}")
        self._server = None
    @property
    def server(self) -> Optional['MCPServer']:
        """Get the MCP server this client is connected to, if any."""
        return self._server
    @server.setter
    def server(self, server: Optional['MCPServer']) -> None:
        """Set the MCP server this client is connected to."""
        self._server = server
    def register_handler(
        self, 
        message_type: Union[MessageType, str], 
        handler: Optional[Callable[[Message], Awaitable[Optional[Message]]]] = None
    ) -> None:
        """
        Register a handler for a specific message type.
        Args:
            message_type: The type of message to handle.
            handler: The async function to call when a message of this type is received.
                   If None, the handler will be unregistered.
        """
        if isinstance(message_type, str):
            try:
                message_type = MessageType(message_type.upper())
            except ValueError:
                pass
        if handler is None:
            self._message_handlers.pop(message_type, None)
        else:
            self._message_handlers[message_type] = handler
    def register_default_handler(self, handler: Callable[[Message], Awaitable[Optional[Message]]]) -> None:
        """
        Register a default handler for messages that don't have a specific handler.
        Args:
            handler: The async function to call for unhandled message types.
        """
        self._default_handler = handler
    async def send_message(
        self, 
        message: Union[Message, Dict[str, Any]],
        receiver: Optional[str] = None,
        message_type: Optional[Union[MessageType, str]] = None,
        **kwargs
    ) -> Optional[Message]:
        """
        Send a message to another client via the MCP server.
        Args:
            message: Either a Message object or a dictionary with message data.
            receiver: Optional receiver ID (if not specified in the message).
            message_type: Optional message type (if not specified in the message).
            **kwargs: Additional fields to include in the message.
        Returns:
            The response message, if any.
        Raises:
            ValueError: If the message cannot be sent (e.g., no server connection).
        """
        if isinstance(message, dict):
            if 'sender' not in message:
                message['sender'] = self.client_id
            if receiver and 'receiver' not in message:
                message['receiver'] = receiver
            if message_type and 'message_type' not in message:
                message['message_type'] = message_type
            message = Message(**{**message, **kwargs})
        if not isinstance(message, Message):
            raise ValueError("Message must be a Message instance or a dictionary")
        if not message.sender:
            message.sender = self.client_id
        self._logger.debug(
            f"Sending message: {message.message_type} from {message.sender} to {message.receiver}",
            extra={
                "message_id": message.message_id,
                "trace_id": message.trace_id,
                "message_type": str(message.message_type)
            }
        )
        if self._server:
            return await self._server.route_message(message)
        if message.receiver == self.client_id:
            return await self.receive_message(message)
        raise ValueError("Cannot send message: No server connection and message is not for this client")
    async def receive_message(self, message: Message) -> Optional[Message]:
        """
        Receive and process an incoming message.
        Args:
            message: The incoming message.
        Returns:
            Optional response message.
        """
        self._logger.debug(
            f"Received message: {message.message_type} from {message.sender}",
            extra={
                "message_id": message.message_id,
                "trace_id": message.trace_id,
                "message_type": str(message.message_type)
            }
        )
        handler = self._message_handlers.get(message.message_type, self._default_handler)
        if handler:
            try:
                return await handler(message)
            except Exception as e:
                self._logger.error(
                    f"Error in message handler for {message.message_type}: {e}",
                    exc_info=True,
                    extra={
                        "message_id": message.message_id,
                        "trace_id": message.trace_id
                    }
                )
                error_msg = message.reply(
                    message_type=MessageType.ERROR,
                    payload={
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "original_message_type": str(message.message_type)
                    }
                )
                return await self.send_message(error_msg)
        else:
            self._logger.warning(
                f"No handler registered for message type: {message.message_type}",
                extra={
                    "message_id": message.message_id,
                    "trace_id": message.trace_id
                }
            )
        return None
class MCPServer:
    """
    MCP Server for message routing between clients.
    This class acts as a central message router in the MCP system, directing
    messages from senders to the appropriate recipients based on client IDs.
    """
    def __init__(self):
        """Initialize the MCP server."""
        self._clients = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    @property
    def client_ids(self) -> List[str]:
        """Get a list of all registered client IDs."""
        return list(self._clients.keys())
    def register_client(self, client_id: str, client: MCPClient) -> None:
        """
        Register a client with the server.
        Args:
            client_id: Unique identifier for the client.
            client: The MCPClient instance to register.
        Raises:
            ValueError: If the client ID is already registered.
        """
        if client_id in self._clients:
            raise ValueError(f"Client ID already registered: {client_id}")
        self._clients[client_id] = client
        client.server = self
        self._logger.info(f"Registered client: {client_id}")
    def unregister_client(self, client_id: str) -> None:
        """
        Unregister a client from the server.
        Args:
            client_id: The ID of the client to unregister.
        """
        if client_id in self._clients:
            self._clients[client_id].server = None
            del self._clients[client_id]
            self._logger.info(f"Unregistered client: {client_id}")
    async def route_message(self, message: Message) -> Optional[Message]:
        """
        Route a message to the appropriate client.
        Args:
            message: The message to route.
        Returns:
            The response message from the recipient, if any.
        Raises:
            ValueError: If the recipient is not found.
        """
        if not isinstance(message, Message):
            raise ValueError("Message must be a Message instance")
        self._logger.debug(
            f"Routing message: {message.message_type} from {message.sender} to {message.receiver}",
            extra={
                "message_id": message.message_id,
                "trace_id": message.trace_id,
                "message_type": str(message.message_type)
            }
        )
        if message.receiver not in self._clients:
            error_msg = f"No client registered with ID: {message.receiver}"
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        recipient = self._clients[message.receiver]
        return await recipient.receive_message(message)
    async def broadcast(self, message: Message, exclude_sender: bool = True) -> List[Message]:
        """
        Broadcast a message to all connected clients.
        Args:
            message: The message to broadcast.
            exclude_sender: If True, don't send the message back to the sender.
        Returns:
            List of response messages from recipients.
        """
        responses = []
        for client_id, client in self._clients.items():
            if exclude_sender and client_id == message.sender:
                continue
            try:
                response = await client.receive_message(message)
                if response:
                    responses.append(response)
            except Exception as e:
                self._logger.error(
                    f"Error broadcasting to {client_id}: {e}",
                    exc_info=True
                )
        return responses