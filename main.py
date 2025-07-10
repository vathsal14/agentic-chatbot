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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Agentic RAG Chatbot",
    description="A Retrieval-Augmented Generation chatbot with modular agents",
    version="0.1.0"
)
from core.document_processor import DocumentProcessor, TextSplitter
from core.embedding_service import EmbeddingService
from storage.vector_store import VectorStore
from core.mcp import MCPServer, Message, MessageType
from agents import (
    IngestionAgent, 
    RetrievalAgent, 
    ResponseAgent, 
    CoordinatorAgent
)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory set to: {UPLOAD_DIR}")
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
logger.info(f"Initializing VectorStore with model: {model_name}")
vector_store = VectorStore(model_name=model_name)  
mcp_server = MCPServer()
global mcp_server_global
mcp_server_global = mcp_server
agents = {}
agent_instances = {
    'ingestion_agent': IngestionAgent,
    'retrieval_agent': RetrievalAgent,
    'response_agent': ResponseAgent,
}
coordinator_global = None
@app.on_event("startup")
async def startup_event():
    global coordinator_global
    try:
        for agent_id, agent_class in agent_instances.items():
            try:
                if agent_id in ['ingestion_agent', 'retrieval_agent']:
                    agent = agent_class(mcp_server=mcp_server, vector_store=vector_store)
                else:
                    agent = agent_class(mcp_server=mcp_server)
                if agent_id not in mcp_server._clients:
                    mcp_server.register_client(agent_id, agent)
                    agents[agent_id] = agent
                    logger.info(f"Registered agent: {agent_id}")
                else:
                    logger.info(f"Agent {agent_id} is already registered")
                    agents[agent_id] = mcp_server._clients[agent_id]
            except Exception as e:
                logger.error(f"Error initializing agent {agent_id}: {str(e)}", exc_info=True)
                raise
        coordinator = CoordinatorAgent(mcp_server=mcp_server, agents=agents)
        if 'coordinator' not in mcp_server._clients:
            mcp_server.register_client('coordinator', coordinator)
            logger.info("Registered coordinator agent")
        else:
            logger.info("Coordinator agent is already registered")
            coordinator = mcp_server._clients['coordinator']
        coordinator_global = coordinator
        logger.info("Coordinator initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing coordinator: {str(e)}", exc_info=True)
        raise
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
class ChatMessage(BaseModel):
    """Model for chat message input"""
    message: str
    conversation_id: Optional[str] = None  
class UploadResponse(BaseModel):
    status: str
    message: str
    file_id: Optional[str] = None
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
        trace_id = str(uuid.uuid4())
        logger.debug(f"Sending message to coordinator with trace_id: {trace_id}")
        try:
            response = await coordinator.send_message(
                receiver_id="coordinator_agent",
                message_type=MessageType.LLM_REQUEST,  
                payload={
                    "query": chat_message.message,
                    "conversation_id": chat_message.conversation_id or str(uuid.uuid4()),
                    "trace_id": trace_id
                }
            )
        except Exception as e:
            logger.error(f"Error sending message to coordinator: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process request: {str(e)}"
            )
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
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )
coordinator_global = None
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global coordinator_global
    try:
        logger.info(f"Starting file upload for {file.filename}")
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = ".bin"  
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved to {file_path} (size: {os.path.getsize(file_path)} bytes)")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}"
            )
        trace_id = str(uuid.uuid4())
        logger.info(f"Processing file upload with trace_id: {trace_id}")
        try:
            if not coordinator_global:
                logger.error("Coordinator not available for file upload")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Coordinator service is not available. Please try again later."
                )
            message = Message(
                sender="api",
                receiver="ingestion_agent",
                message_type=MessageType.INGESTION_REQUEST,
                trace_id=trace_id,
                payload={
                    "file_path": file_path,
                    "metadata": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "source": file.filename,
                        "trace_id": trace_id
                    }
                }
            )
            logger.debug(f"Sending message to coordinator: {message}")
            response = await mcp_server_global.route_message(message)
            if not response or response.message_type == MessageType.ERROR:
                error_msg = response.get("payload", {}).get("error", "Error processing file") if response else "No response from coordinator"
                logger.error(f"Error processing file upload {trace_id}: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            logger.info(f"File processed successfully: {file.filename}")
            return {
                "status": "success",
                "message": "File uploaded and processed successfully",
                "file_id": file_id,
                "trace_id": trace_id
            }
        except HTTPException:
            raise  
        except Exception as e:
            logger.error(f"Error during file processing {trace_id}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file: {str(e)}"
            )
    except HTTPException:
        raise  
    except Exception as e:
        logger.error(f"Unexpected error in file upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns:
        JSON response with service status
    """
    trace_id = str(uuid.uuid4())
    health_status = {
        "status": "ok",
        "services": {},
        "version": "1.0.0",
        "trace_id": trace_id,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    try:
        doc_count = vector_store.get_document_count()
        health_status["services"]["vector_store"] = {
            "status": "ok",
            "document_count": doc_count
        }
    except Exception as e:
        health_status["services"]["vector_store"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    try:
        test_embedding = embedding_service.embed_text("health check")
        health_status["services"]["embedding_service"] = {
            "status": "ok",
            "embedding_dimensions": len(test_embedding) if test_embedding else 0
        }
    except Exception as e:
        health_status["services"]["embedding_service"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    health_status["agents"] = {}
    for agent_id in ["ingestion_agent", "retrieval_agent", "response_agent", "coordinator_agent"]:
        try:
            response = await coordinator.send_message(
                receiver_id=agent_id,
                message_type=MessageType.LLM_REQUEST,
                payload={
                    "action": "ping",
                    "trace_id": trace_id
                }
            )
            health_status["agents"][agent_id] = {
                "status": "ok",
                "response_time": response.get("metadata", {}).get("response_time", 0) if response else 0
            }
        except Exception as e:
            health_status["agents"][agent_id] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
    status_code = 200 if health_status["status"] == "ok" else 503
    return JSONResponse(
        content=health_status,
        status_code=status_code
    )
@app.post("/api/clear", response_model=dict)
async def clear_knowledge_base():
    """Clear the knowledge base."""
    trace_id = str(uuid.uuid4())
    logger.info(f"Clearing knowledge base, trace_id: {trace_id}")
    try:
        vector_store.clear()
        try:
            await coordinator.send_message(
                receiver_id="ingestion_agent",
                message_type=MessageType.INGESTION_REQUEST,
                payload={
                    "action": "clear_knowledge_base",
                    "trace_id": trace_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to notify ingestion agent about knowledge base clear: {str(e)}")
        return {
            "status": "success", 
            "message": "Knowledge base cleared successfully",
            "trace_id": trace_id
        }
    except Exception as e:
        error_msg = f"Error clearing knowledge base: {str(e)}"
        logger.error(f"{error_msg}, trace_id: {trace_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=error_msg
        )
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main chat interface."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint
    Returns:
        JSON response with service status
    """
    try:
        return {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "database": "ok",  
                "models": "ok"     
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not healthy"
        )
if __name__ == "__main__":
    (Path(__file__).parent / "templates").mkdir(exist_ok=True)
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
                        background-color: 
                        color: white;
                        float: right;
                        margin-left: auto;
                        margin-right: 0;
                    }
                    .bot-message {
                        background-color: 
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
    uvicorn.run(app, host="0.0.0.0", port=8000)