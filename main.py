import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil
import uuid
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from document_processor import DocumentProcessor, TextSplitter
from embedding_service import EmbeddingService
from agents import (
    MCPMessage, 
    IngestionAgent, 
    RetrievalAgent, 
    ResponseAgent, 
    CoordinatorAgent
)

# Initialize components
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()

# Create agents
agents = {
    'ingestion_agent': IngestionAgent(document_processor),
    'retrieval_agent': RetrievalAgent(embedding_service),
    'response_agent': ResponseAgent(),
}

# Initialize coordinator
coordinator = CoordinatorAgent(agents)

# Create necessary directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Agentic RAG Chatbot",
    description="A Retrieval-Augmented Generation chatbot with modular agents",
    version="0.1.0"
)

# Add startup event to initialize agents
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    # Initialize any required resources here
    pass

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Models
class ChatMessage(BaseModel):
    """Model for chat message input"""
    message: str
    conversation_id: Optional[str] = None  # For tracking conversations

class UploadResponse(BaseModel):
    status: str
    message: str
    file_id: Optional[str] = None

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=Dict[str, Any])
async def chat(chat_message: ChatMessage):
    """
    Handle chat messages from the user.
    
    Args:
        chat_message: The incoming chat message with optional conversation_id
        
    Returns:
        JSON response with the assistant's reply
    """
    try:
        logger.info(f"Received chat request: {chat_message.message}")
        
        # Create MCP message with a unique trace_id
        trace_id = str(uuid.uuid4())
        message = MCPMessage(
            sender="web_interface",
            receiver="coordinator_agent",
            message_type="USER_QUERY",
            trace_id=trace_id,
            payload={
                "query": chat_message.message,
                "conversation_id": chat_message.conversation_id
            }
        )
        
        # Send to coordinator
        logger.debug(f"Sending message to coordinator with trace_id: {trace_id}")
        response = await coordinator.send_message(message)
        
        if not response:
            error_msg = "No response received from coordinator"
            logger.error(f"{error_msg} for trace_id: {trace_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
            
        if response.message_type == "ERROR":
            error_msg = response.payload.get("error", "Error processing your request")
            logger.error(f"Error from coordinator: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        if not hasattr(response, 'payload') or 'response' not in response.payload:
            error_msg = f"Invalid response format from coordinator: {response}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response format from the server"
            )
            
        logger.info(f"Successfully generated response for trace_id: {trace_id}")
        return {
            "response": response.payload.get("response", "No response generated"),
            "conversation_id": chat_message.conversation_id or str(uuid.uuid4()),
            "trace_id": trace_id
        }
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions as-is
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create MCP message with a unique trace_id
        message = MCPMessage(
            sender="web_interface",
            receiver="coordinator_agent",
            message_type="UPLOAD_DOCUMENT",
            trace_id=str(uuid.uuid4()),  # Add unique trace_id
            payload={
                "file_path": file_path,
                "metadata": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "source": file.filename
                }
            }
        )
        
        # Send to coordinator
        response = coordinator.send_message(message)
        
        if not response or response.message_type == "ERROR":
            error_msg = response.payload.get("error", "Error processing file") if response else "No response from coordinator"
            raise HTTPException(status_code=400, detail=error_msg)
        
        return {
            "status": "success",
            "message": "File uploaded and processed successfully",
            "file_id": file_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clear")
async def clear_knowledge_base():
    """Clear the knowledge base."""
    try:
        embedding_service.clear()
        return {"status": "success", "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Frontend
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main chat interface."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Health check
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        JSON response with service status
    """
    try:
        # Add any additional health checks here
        return {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "database": "ok",  # Add actual database check
                "models": "ok"     # Add model loading check
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not healthy"
        )

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    (Path(__file__).parent / "templates").mkdir(exist_ok=True)
    
    # Create a simple HTML template if it doesn't exist
    if not (Path(__file__).parent / "templates" / "index.html").exists():
        with open(Path(__file__).parent / "templates" / "index.html", "w", encoding="utf-8") as f:
            f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Agentic RAG Chatbot</title>
                <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
                <style>
                    .chat-container {
                        height: calc(100vh - 200px);
                        overflow-y: auto;
                    }
                    .message {
                        max-width: 80%;
                        margin: 10px;
                        padding: 10px 15px;
                        border-radius: 15px;
                        clear: both;
                    }
                    .user-message {
                        background-color: #3b82f6;
                        color: white;
                        float: right;
                        margin-left: auto;
                        margin-right: 0;
                    }
                    .bot-message {
                        background-color: #e5e7eb;
                        float: left;
                        margin-right: auto;
                        margin-left: 0;
                    }
                </style>
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto px-4 py-8">
                    <h1 class="text-3xl font-bold text-center mb-8">Agentic RAG Chatbot</h1>
                    
                    <!-- Chat container -->
                    <div class="bg-white rounded-lg shadow-lg p-6">
                        <!-- Messages -->
                        <div id="chat-messages" class="chat-container mb-4">
                            <div class="text-center text-gray-500 py-4">
                                Upload documents and ask questions about them!
                            </div>
                        </div>
                        
                        <!-- Input area -->
                        <div class="flex flex-col space-y-4">
                            <!-- File upload -->
                            <div class="flex items-center space-x-4">
                                <input type="file" id="file-upload" class="hidden" accept=".pdf,.docx,.pptx,.txt,.md,.csv">
                                <button id="upload-btn" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded">
                                    Upload Document
                                </button>
                                <span id="file-name" class="text-sm text-gray-500">No file selected</span>
                            </div>
                            
                            <!-- Message input -->
                            <div class="flex space-x-4">
                                <input 
                                    type="text" 
                                    id="user-input" 
                                    placeholder="Type your message..." 
                                    class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    onkeypress="if(event.key === 'Enter') sendMessage()"
                                >
                                <button 
                                    onclick="sendMessage()" 
                                    class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg"
                                >
                                    Send
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                    // DOM elements
                    const chatMessages = document.getElementById('chat-messages');
                    const userInput = document.getElementById('user-input');
                    const fileUpload = document.getElementById('file-upload');
                    const uploadBtn = document.getElementById('upload-btn');
                    const fileName = document.getElementById('file-name');
                    
                    // Event listeners
                    uploadBtn.addEventListener('click', () => fileUpload.click());
                    
                    fileUpload.addEventListener('change', async (e) => {
                        const file = e.target.files[0];
                        if (!file) return;
                        
                        fileName.textContent = file.name;
                        
                        const formData = new FormData();
                        formData.append('file', file);
                        
                        try {
                            const response = await fetch('/api/upload', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const result = await response.json();
                            
                            if (response.ok) {
                                addMessage('system', `Document "${file.name}" uploaded successfully!`);
                            } else {
                                throw new Error(result.detail || 'Failed to upload file');
                            }
                        } catch (error) {
                            addMessage('system', `Error: ${error.message}`, true);
                        }
                        
                        // Reset file input
                        fileUpload.value = '';
                        fileName.textContent = 'No file selected';
                    });
                    
                    // Add a message to the chat
                    function addMessage(sender, text, isError = false) {
                        const messageDiv = document.createElement('div');
                        messageDiv.className = `message ${sender}-message`;
                        messageDiv.style.color = isError ? 'red' : 'inherit';
                        messageDiv.textContent = text;
                        chatMessages.appendChild(messageDiv);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                    
                    // Send a message to the server
                    async function sendMessage() {
                        const message = userInput.value.trim();
                        if (!message) return;
                        
                        // Add user message to chat
                        addMessage('user', message);
                        userInput.value = '';
                        
                        try {
                            const response = await fetch('/api/chat', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    query: message
                                })
                            });
                            
                            const data = await response.json();
                            
                            if (response.ok) {
                                addMessage('bot', data.response);
                            } else {
                                throw new Error(data.detail || 'Failed to get response');
                            }
                        } catch (error) {
                            addMessage('system', `Error: ${error.message}`, true);
                        }
                    }
                    
                    // Clear chat
                    function clearChat() {
                        chatMessages.innerHTML = '';
                    }
                </script>
            </body>
            </html>
            """)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
