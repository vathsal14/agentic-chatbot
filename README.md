# Agentic RAG Chatbot with MCP

An advanced Retrieval-Augmented Generation (RAG) chatbot featuring an agent-based architecture using Model Context Protocol (MCP) for seamless communication between components. This implementation uses FastAPI for the backend and includes a modular agent system for processing and responding to user queries.

## 🚀 Features

- **Multi-format Document Support**: Process PDF, DOCX, PPTX, CSV, and TXT files
- **Agent-Based Architecture**: Modular design with specialized agents for ingestion, retrieval, and response generation
- **Model Context Protocol (MCP)**: Standardized communication protocol between agents
- **Vector Search**: Semantic search capabilities using FAISS and sentence-transformers
- **Web Interface**: Interactive UI for document uploads and chat
- **Local-First**: Runs entirely on your machine with local models by default
- **Asynchronous Processing**: Non-blocking operations for better performance

## 🏗️ Architecture

### Core Components

#### Agents
- **`IngestionAgent`**: Processes and chunks uploaded documents, handles text extraction and splitting
- **`RetrievalAgent`**: Performs semantic search using sentence-transformers and FAISS
- **`ResponseAgent`**: Generates contextual responses using language models
- **`CoordinatorAgent`**: Manages workflow and message routing between agents using MCP

#### Storage
- **FAISS Vector Store**: Efficient similarity search with `all-MiniLM-L6-v2` embeddings (384 dimensions)
- **Local File System**: Stores uploaded documents in the `uploads/` directory

#### API Endpoints
- `POST /api/upload`: Upload and process documents
- `POST /api/chat`: Send chat messages and get responses
- `GET /health`: Check service status
- `POST /api/clear_kb`: Clear the knowledge base

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agentic-chatbot.git
   cd agentic-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Unix/MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note**: The first run will download the `all-MiniLM-L6-v2` model (~80MB)

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   The default configuration in `.env` is pre-configured for local development:
   ```env
   # Server
   HOST=0.0.0.0
   PORT=8000
   DEBUG=true
   
   # Models
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   LLM_MODEL=distilgpt2
   
   # Storage
   UPLOAD_FOLDER=uploads
   
   # Optional: Set to "cuda" if you have a CUDA-compatible GPU
   # DEVICE=cuda
   ```

## 🚀 Running the Application

1. Start the FastAPI server with auto-reload for development:
   ```bash
   uvicorn main:app --reload
   ```

2. Access the application:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 🖥️ Usage

### Uploading Documents
1. **Using the Web Interface**:
   - Navigate to http://localhost:8000
   - Click "Upload Document"
   - Select a file (PDF, DOCX, PPTX, CSV, or TXT)
   - Wait for processing to complete

2. **Using the API**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/api/upload' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@your_document.pdf;type=application/pdf'
   ```

### Chat Interface
1. **Web Interface**:
   - Type your question in the input box
   - Press Enter or click "Send"
   - View the response with relevant document sources

2. **API Usage**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/api/chat' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "message": "What is this document about?",
       "conversation_id": "optional_conversation_id"
     }'
   ```

## ⚙️ Configuration

The application can be configured using environment variables in the `.env` file:

```env
# Server Configuration
HOST=0.0.0.0  # Bind address
PORT=8000      # Port to run the server on
DEBUG=true     # Enable debug mode (not recommended for production)

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer model for embeddings
LLM_MODEL=distilgpt2              # Language model for response generation

# Storage Configuration
UPLOAD_FOLDER=uploads  # Directory to store uploaded files

# Hardware Configuration
# DEVICE=cpu  # Set to "cuda" if you have a CUDA-compatible GPU

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
VECTOR_STORE_PATH=vector_store.faiss
```

## 📂 Project Structure

```
agentic-chatbot/
├── main.py               # Main application and web server
├── document_processor.py # Document parsing and text extraction
├── embedding_service.py  # Text embeddings and similarity search
├── agents.py            # Agent system with MCP implementation
├── minimal_requirements.txt  # Python dependencies
├── uploads/             # Directory for uploaded files
└── README.md
```

## Troubleshooting

### Common Issues

1. **Model Loading Issues**:
   - Ensure you have enough disk space for the models
   - Check your internet connection if downloading models for the first time
   - Verify the model names in your `.env` file are correct

2. **Document Processing Failures**:
   - Make sure the uploaded files are not corrupted
   - Check that the file formats are supported
   - Verify file permissions in the uploads directory

3. **Performance Issues**:
   - The application runs locally and may be slow on less powerful machines
   - Consider using a smaller model if performance is a concern

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Hugging Face](https://huggingface.co/) - Transformers and models
- [sentence-transformers](https://www.sbert.net/) - For text embeddings
