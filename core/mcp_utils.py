"""
Utility functions and classes for working with the Model-Controller-Protocol (MCP) system.
This module provides helper functions, decorators, and utilities that make it easier
to work with the MCP message passing system.
"""
from __future__ import annotations
import asyncio
import functools
import inspect
import json
import logging
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union,
    get_type_hints, overload
)
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4
import datetime
from pydantic import BaseModel, Field, validator
from .mcp import Message, MessageType, MCPClient, MCPServer
T = TypeVar('T')
P = TypeVar('P')
R = TypeVar('R')
MessageHandler = Callable[[Message], Awaitable[Optional[Message]]]
MessageHandlerDecorator = Callable[[MessageHandler], MessageHandler]
class MessageHandlerInfo(BaseModel):
    """Information about a registered message handler."""
    message_type: Union[MessageType, str]
    handler: MessageHandler
    is_async: bool
    docstring: Optional[str] = None
    signature: Optional[inspect.Signature] = None
    class Config:
        arbitrary_types_allowed = True
class MessageRouter:
    """
    A utility class for routing messages to registered handlers.
    This class provides a convenient way to register message handlers and
    route incoming messages to the appropriate handler based on message type.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the message router.
        Args:
            logger: Optional logger instance. If not provided, a default logger will be created.
        """
        self._handlers: Dict[Union[MessageType, str], MessageHandlerInfo] = {}
        self._default_handler: Optional[MessageHandler] = None
        self.logger = logger or logging.getLogger(__name__)
    def register(
        self, 
        message_type: Union[MessageType, str],
        **register_kwargs
    ) -> Callable[[MessageHandler], MessageHandler]:
        """
        Decorator to register a message handler for a specific message type.
        Args:
            message_type: The message type to handle.
            **register_kwargs: Additional keyword arguments to store with the handler.
        Returns:
            A decorator function that registers the handler.
        """
        def decorator(handler: MessageHandler) -> MessageHandler:
            is_async = asyncio.iscoroutinefunction(handler)
            docstring = inspect.getdoc(handler)
            try:
                signature = inspect.signature(handler)
            except (ValueError, TypeError):
                signature = None
            self._handlers[message_type] = MessageHandlerInfo(
                message_type=message_type,
                handler=handler,
                is_async=is_async,
                docstring=docstring,
                signature=signature,
                **register_kwargs
            )
            self.logger.debug(
                f"Registered handler for message type: {message_type}",
                extra={"handler": handler.__name__, "is_async": is_async}
            )
            return handler
        return decorator
    def register_default(self, handler: MessageHandler) -> MessageHandler:
        """
        Register a default handler for messages that don't have a specific handler.
        Args:
            handler: The handler function to register.
        Returns:
            The registered handler function.
        """
        self._default_handler = handler
        self.logger.debug(
            f"Registered default message handler: {handler.__name__}",
            extra={"is_async": asyncio.iscoroutinefunction(handler)}
        )
        return handler
    async def handle_message(self, message: Message) -> Optional[Message]:
        """
        Route a message to the appropriate handler.
        Args:
            message: The message to handle.
        Returns:
            The response message, if any.
        """
        if not isinstance(message, Message):
            raise ValueError("Message must be an instance of Message")
        handler_info = self._handlers.get(message.message_type)
        if handler_info:
            try:
                self.logger.debug(
                    f"Dispatching message to handler: {handler_info.handler.__name__}",
                    extra={
                        "message_id": message.message_id,
                        "message_type": str(message.message_type),
                        "trace_id": message.trace_id
                    }
                )
                if handler_info.is_async:
                    return await handler_info.handler(message)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, handler_info.handler, message
                    )
            except Exception as e:
                self.logger.error(
                    f"Error in message handler for {message.message_type}: {e}",
                    exc_info=True,
                    extra={
                        "message_id": message.message_id,
                        "trace_id": message.trace_id
                    }
                )
                raise
        elif self._default_handler is not None:
            self.logger.debug(
                "Using default handler for message",
                extra={
                    "message_id": message.message_id,
                    "message_type": str(message.message_type),
                    "trace_id": message.trace_id
                }
            )
            return await self._default_handler(message)
        else:
            self.logger.warning(
                f"No handler registered for message type: {message.message_type}",
                extra={
                    "message_id": message.message_id,
                    "trace_id": message.trace_id
                }
            )
            return None
    def get_handler_info(
        self, 
        message_type: Union[MessageType, str]
    ) -> Optional[MessageHandlerInfo]:
        """
        Get information about a registered handler.
        Args:
            message_type: The message type to get handler info for.
        Returns:
            Information about the handler, or None if not found.
        """
        return self._handlers.get(message_type)
    def list_handlers(self) -> List[MessageHandlerInfo]:
        """
        Get a list of all registered handlers.
        Returns:
            A list of handler information objects.
        """
        return list(self._handlers.values())
    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._default_handler = None
        self.logger.debug("Cleared all message handlers")