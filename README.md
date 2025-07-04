# Agentic RAG Chatbot

A lightweight, local-first Retrieval-Augmented Generation (RAG) chatbot with an agent-based architecture using Model Context Protocol (MCP).

## Features

- **Multi-format Document Support**: Upload and process PDF, DOCX, PPTX, CSV, and plain text files
- **Local-First Architecture**: Runs entirely on your machine with local models
- **Agent-Based Design**: Modular agents for ingestion, retrieval, and response generation
- **Simple Web Interface**: Easy-to-use chat interface for interacting with your documents
- **No External Dependencies**: No API keys or external services required

## Architecture

The system is built using the following components:

1. **Agents**:
   - **Ingestion Agent**: Processes and chunks uploaded documents
   - **Retrieval Agent**: Handles semantic search using sentence-transformers
   - **Response Agent**: Generates responses using local language models
   - **Coordinator Agent**: Manages workflow between agents using MCP

2. **Storage**:
   - **In-Memory Vector Store**: Simple in-memory storage for document chunks and embeddings
   - **File Storage**: Local file system for uploaded documents

3. **API**:
   - FastAPI backend with RESTful endpoints
   - Simple HTML/JavaScript frontend

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Basic command line knowledge

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd agentic-chatbot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r minimal_requirements.txt
   ```

4. Start the server:
   ```bash
   python main.py
   ```

5. Open your browser to http://localhost:8000

## Usage

### Uploading Documents

1. Click the "Upload Document" button in the web interface
2. Select a supported file (PDF, DOCX, PPTX, CSV, TXT)
3. Wait for the upload and processing to complete

### Asking Questions

1. Type your question in the chat input
2. Press Enter or click Send
3. The system will search through your documents and generate a response

## Configuration

Edit the `.env` file to configure the application:

```
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=distilgpt2

# File Storage
UPLOAD_FOLDER=uploads
```

## Project Structure

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
