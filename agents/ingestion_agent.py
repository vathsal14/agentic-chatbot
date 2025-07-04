import os
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

from ..core.mcp import Message, MessageType
from ..core.document_processor import DocumentProcessor
from ..storage.vector_store import VectorStore
from .base_agent import BaseAgent

class IngestionAgent(BaseAgent):
    """Agent responsible for ingesting and processing documents"""
    
    def __init__(self, mcp_server, vector_store: VectorStore, upload_dir: str = "./uploads"):
        """
        Initialize the Ingestion Agent
        
        Args:
            mcp_server: Reference to the MCP server
            vector_store: Vector store instance for storing document embeddings
            upload_dir: Directory to store uploaded files
        """
        super().__init__("ingestion_agent", mcp_server)
        self.vector_store = vector_store
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        self.document_processor = DocumentProcessor()
    
    def setup_handlers(self):
        """Set up message handlers for this agent"""
        self.register_handler(MessageType.INGESTION_REQUEST, self.handle_ingestion_request)
    
    async def handle_ingestion_request(self, message: Message):
        """
        Handle document ingestion requests
        
        Args:
            message: MCP message containing document ingestion request
        """
        try:
            payload = message.payload
            file_paths = payload.get("file_paths", [])
            metadata = payload.get("metadata", {})
            
            if not file_paths:
                raise ValueError("No file paths provided in ingestion request")
            
            # Process each file
            processed_docs = []
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue
                
                # Process the file
                result = self.document_processor.process_file(str(file_path))
                if not result.get("success", False):
                    print(f"Failed to process {file_path}: {result.get('error')}")
                    continue
                
                # Chunk the document
                chunks = self.document_processor.chunk_text(
                    result["content"],
                    chunk_size=1000,
                    overlap=200
                )
                
                # Prepare chunks for storage
                for chunk in chunks:
                    doc_metadata = {
                        "source": str(file_path.name),
                        "file_type": result.get("file_type", "unknown"),
                        **metadata
                    }
                    
                    processed_docs.append({
                        "text": chunk["text"],
                        "metadata": {
                            **doc_metadata,
                            "chunk_start": chunk["start"],
                            "chunk_end": chunk["end"]
                        }
                    })
            
            # Store in vector database
            if processed_docs:
                doc_ids = self.vector_store.add_documents(processed_docs)
                
                # Send success response
                await self.send_message(
                    receiver_id=message.sender,
                    message_type=MessageType.INGESTION_RESPONSE,
                    payload={
                        "status": "success",
                        "document_count": len(processed_docs),
                        "document_ids": doc_ids,
                        "trace_id": message.trace_id
                    }
                )
            else:
                raise ValueError("No valid documents were processed")
                
        except Exception as e:
            print(f"Error in ingestion agent: {e}")
            await self.handle_error(e, message.trace_id)
    
    async def save_uploaded_file(self, file_data: bytes, filename: str) -> str:
        """
        Save an uploaded file to the upload directory
        
        Args:
            file_data: File data as bytes
            filename: Original filename
            
        Returns:
            Path to the saved file
        """
        # Create a safe filename
        safe_filename = "".join(c if c.isalnum() or c in '._- ' else '_' for c in filename)
        file_path = os.path.join(self.upload_dir, safe_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(file_data)
            
        return file_path
