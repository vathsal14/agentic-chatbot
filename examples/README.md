# MCP Example: Simple Chat Application

This example demonstrates how to use the Model-Controller-Protocol (MCP) system to build a simple chat application with multiple clients that can communicate through a central server.

## Features

- Multiple users can join the chat with unique usernames
- Broadcast messages to all connected users
- Private messaging with @username syntax
- System notifications for user join/leave events
- Clean console-based interface

## How to Run

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies (if any):
   ```bash
   pip install pydantic
   ```

3. Start the chat client by running:
   ```bash
   python mcp_example.py [username]
   ```
   
   If you don't provide a username, you'll be prompted to enter one.

## Usage

- Type a message and press Enter to broadcast to all users
- Start a message with `@username` to send a private message to a specific user
- Type `/quit` to exit the chat

## Example Session

```
$ python mcp_example.py Alice

Welcome to the chat, Alice! Type '/quit' to exit.
To send a private message, use: @username your message
To broadcast to everyone, just type your message

> Hello everyone!
[You -> Everyone]: Hello everyone!
> @Bob Hi Bob, how are you?
[You -> Bob]: Hi Bob, how are you?

[SYSTEM] Bob has joined the chat.

[2023-04-01 14:30:15] Bob: Hi Alice! I'm doing great, thanks for asking!
> /quit
Goodbye!
```

## Implementation Details

This example demonstrates several key concepts of the MCP system:

1. **Message Passing**: Clients communicate by sending and receiving messages through the MCP server.
2. **Message Routing**: The server routes messages to the appropriate recipients based on the message type and receiver ID.
3. **Handlers**: Clients register message handlers to process different types of messages.
4. **Asynchronous I/O**: The system uses Python's asyncio for efficient, non-blocking I/O operations.

## Extending the Example

You can extend this example by:

1. Adding more message types (e.g., file sharing, typing indicators)
2. Implementing user authentication
3. Adding chat rooms or channels
4. Persisting chat history
5. Adding a GUI using a framework like Tkinter or PyQt
