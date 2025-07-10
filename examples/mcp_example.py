"""
Example demonstrating the Model-Controller-Protocol (MCP) system.
This example shows how to set up a simple MCP server with multiple clients
that can communicate with each other using the MCP message passing system.
"""
import asyncio
import logging
import sys
from typing import Dict, Optional, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core import (
    MCPServer,
    MCPClient,
    Message,
    MessageType,
    MessageRouter
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
class ChatClient(MCPClient):
    """A simple chat client that can send and receive messages."""
    def __init__(self, client_id: str, username: str):
        super().__init__(client_id)
        self.username = username
        self.router = MessageRouter(logger=logger.getChild(f"Router-{client_id}"))
        self.setup_handlers()
    def setup_handlers(self):
        """Set up message handlers for this client."""
        @self.router.register(MessageType("CHAT_MESSAGE"))
        async def handle_chat_message(message: Message) -> Optional[Message]:
            if message.sender == self.client_id:
                return None
            print(f"\n[{message.timestamp}] {message.payload.get('sender', 'Unknown')}: "
                  f"{message.payload.get('content', '')}")
            print("> ", end="", flush=True)
            return None
        @self.router.register(MessageType("SYSTEM_MESSAGE"))
        async def handle_system_message(message: Message) -> Optional[Message]:
            print(f"\n[SYSTEM] {message.payload.get('content', '')}")
            print("> ", end="", flush=True)
            return None
    async def receive_message(self, message: Message) -> Optional[Message]:
        """Override to use our message router."""
        return await self.router.handle_message(message)
    async def send_chat_message(self, content: str, recipient: Optional[str] = None) -> None:
        """Send a chat message to a recipient or broadcast to all."""
        message = Message(
            message_type="CHAT_MESSAGE",
            sender=self.client_id,
            receiver=recipient or "broadcast",
            payload={
                "sender": self.username,
                "content": content,
                "timestamp": str(datetime.datetime.now())
            }
        )
        if recipient:
            await self.send_message(message)
            print(f"[You -> {recipient}]: {content}")
        else:
            await self.send_message(message)
            print(f"[You -> Everyone]: {content}")
async def run_chat_client(server: MCPServer, username: str):
    """Run a chat client in the console."""
    client_id = f"user_{username.lower()}"
    client = ChatClient(client_id, username)
    server.register_client(client_id, client)
    await server.broadcast(
        Message(
            message_type="SYSTEM_MESSAGE",
            sender="system",
            receiver="broadcast",
            payload={
                "content": f"{username} has joined the chat."
            }
        )
    )
    print(f"\nWelcome to the chat, {username}! Type '/quit' to exit.")
    print("To send a private message, use: @username your message")
    print("To broadcast to everyone, just type your message\n")
    try:
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() == '/quit':
                break
            if text.startswith('@'):
                parts = text[1:].split(' ', 1)
                if len(parts) == 2:
                    recipient, content = parts
                    await client.send_chat_message(content, recipient=recipient)
                else:
                    print("Invalid private message format. Use: @username message")
            else:
                if text:
                    await client.send_chat_message(text)
    finally:
        server.unregister_client(client_id)
        await server.broadcast(
            Message(
                message_type="SYSTEM_MESSAGE",
                sender="system",
                receiver="broadcast",
                payload={
                    "content": f"{username} has left the chat."
                }
            )
        )
        print("\nGoodbye!")
async def main():
    """Run the MCP chat example."""
    server = MCPServer()
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter your username: ").strip()
        if not username:
            username = f"User_{id(username) % 1000}"
    await run_chat_client(server, username)
if __name__ == "__main__":
    import sys
    import os
    if os.name == 'nt':
        import msvcrt
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)