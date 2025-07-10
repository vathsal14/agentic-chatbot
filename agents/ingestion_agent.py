import os
import asyncio
import datetime
import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from core.mcp import Message, MessageType
from core.document_processor import DocumentProcessor
from storage.vector_store import VectorStore
from agents.base_agent import BaseAgent
class IngestionAgent(BaseAgent):
    """Agent responsible for ingesting and processing documents."""
    def __init__(self, mcp_server: Any, vector_store: VectorStore, upload_dir: str = "./uploads") -> None:
        """
        Initialize the Ingestion Agent.
        Args:
            mcp_server: The MCP server instance for message routing.
            vector_store: Vector store instance for document storage and retrieval.
            upload_dir: Directory to store uploaded files.
        """
        super().__init__("ingestion_agent", mcp_server)
        self.vector_store = vector_store
        self.document_processor = DocumentProcessor()
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.setup_handlers()
        self.logger.info(f"Initialized {self.__class__.__name__} with upload directory: {self.upload_dir.absolute()}")
    def setup_handlers(self) -> None:
        """Set up message handlers for different message types."""
        self.register_handler(MessageType.INGESTION_REQUEST, self.handle_ingestion_request)
    async def receive_message(self, message: Message) -> Optional[Message]:
        """
        Receive and process an incoming message.
        Args:
            message: The incoming message to process.
        Returns:
            Optional response message.
        """
        try:
            return await super().handle_message(message)
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_response = message.reply(
                message_type=MessageType.ERROR,
                payload={
                    "error": str(e),
                    "original_message": message.dict()
                }
            )
            return error_response
    async def handle_ingestion_request(self, message: Message) -> Message:
        """
        Handle an ingestion request message.
        Args:
            message: The ingestion request message.
        Returns:
            Response message with the result of the ingestion.
        """
        trace_id = message.payload.get("trace_id", str(uuid.uuid4()))
        self.logger.info(f"Received ingestion request, trace_id: {trace_id}")
        try:
            action = message.payload.get("action")
            if action == "clear_knowledge_base":
                await self.clear_knowledge_base()
                return Message(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type=MessageType.INGESTION_RESPONSE,
                    payload={
                        "status": "success",
                        "message": "Knowledge base cleared successfully",
                        "trace_id": trace_id
                    }
                )
            elif action == "ping":
                return Message(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type=MessageType.INGESTION_RESPONSE,
                    payload={
                        "status": "success",
                        "message": "pong",
                        "trace_id": trace_id
                    }
                )
            file_paths = message.payload.get("file_paths", [])
            if not file_paths:
                raise ValueError("No file paths provided in the ingestion request")
            return await self.process_files(file_paths, message.sender, trace_id)
        except Exception as e:
            error_msg = f"Failed to process ingestion request: {str(e)}"
            self.logger.error(f"{error_msg}, trace_id: {trace_id}", exc_info=True)
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                payload={
                    "status": "error",
                    "error": error_msg,
                    "trace_id": trace_id
                }
            )
    async def process_files(
        self, 
        file_paths: List[Union[str, Path]], 
        sender: str, 
        trace_id: str
    ) -> Message:
        """
        Process multiple files for ingestion.
        Args:
            file_paths: List of file paths to process.
            sender: ID of the message sender.
            trace_id: Trace ID for logging and tracking.
        Returns:
            Response message with the result of the processing.
        """
        processed_docs = []
        processing_errors = []
        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                self.logger.warning(f"{error_msg}, trace_id: {trace_id}")
                processing_errors.append({
                    "file": str(file_path),
                    "error": "File not found",
                    "trace_id": trace_id
                })
                continue
            try:
                file_docs, file_errors = await self.process_single_file(file_path, trace_id)
                processed_docs.extend(file_docs)
                processing_errors.extend(file_errors)
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                self.logger.error(f"{error_msg}, trace_id: {trace_id}", exc_info=True)
                processing_errors.append({
                    "file": str(file_path),
                    "error": str(e),
                    "trace_id": trace_id
                })
        if processed_docs:
            try:
                await self.store_documents(processed_docs, trace_id)
            except Exception as e:
                error_msg = f"Error storing documents in vector store: {str(e)}"
                self.logger.error(f"{error_msg}, trace_id: {trace_id}", exc_info=True)
                processing_errors.append({
                    "error": error_msg,
                    "trace_id": trace_id
                })
        response_payload = {
            "status": "success" if not processing_errors else "partial_success" if processed_docs else "error",
            "processed_count": len(processed_docs),
            "error_count": len(processing_errors),
            "trace_id": trace_id
        }
        if processing_errors:
            response_payload["errors"] = processing_errors
        return Message(
            sender=self.agent_id,
            receiver=sender,
            message_type=MessageType.INGESTION_RESPONSE,
            payload=response_payload
        )
    async def process_single_file(
        self, 
        file_path: Path, 
        trace_id: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a single file for ingestion.
        Args:
            file_path: Path to the file to process.
            trace_id: Trace ID for logging and tracking.
        Returns:
            Tuple of (processed_docs, processing_errors)
        """
        processed_docs = []
        processing_errors = []
        file_metadata = {
            "source": file_path.name,
            "ingestion_timestamp": datetime.datetime.utcnow().isoformat(),
            "file_size": file_path.stat().st_size,
            "trace_id": trace_id
        }
        start_time = datetime.datetime.now()
        result = self.document_processor.process_document(
            file_path=str(file_path),
            metadata=file_metadata
        )
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error processing document")
            self.logger.warning(f"Failed to process {file_path.name}: {error_msg}, trace_id: {trace_id}")
            processing_errors.append({
                "file": str(file_path),
                "error": error_msg,
                "trace_id": trace_id
            })
            return processed_docs, processing_errors
        content = result.get("content", "")
        if not content.strip():
            self.logger.warning(f"No content extracted from {file_path.name}, trace_id: {trace_id}")
            processing_errors.append({
                "file": str(file_path),
                "error": "No content extracted",
                "trace_id": trace_id
            })
            return processed_docs, processing_errors
        chunks = self.document_processor.chunk_text(
            content=content,
            chunk_size=1000,
            overlap=200
        )
        for chunk in chunks:
            chunk_metadata = {
                **result.get("metadata", {}),
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"],
                "chunk_id": f"{file_path.stem}_{chunk['start']}_{chunk['end']}",
                "trace_id": trace_id
            }
            processed_docs.append({
                "text": chunk["text"],
                "metadata": chunk_metadata
            })
        self.logger.info(
            f"Processed {len(chunks)} chunks from {file_path.name} "
            f"in {processing_time:.2f}s, trace_id: {trace_id}"
        )
        return processed_docs, processing_errors
    async def store_documents(self, documents: List[Dict[str, Any]], trace_id: str) -> None:
        """
        Store processed documents in the vector store.
        Args:
            documents: List of documents to store.
            trace_id: Trace ID for logging and tracking.
        """
        self.logger.info(f"Storing {len(documents)} document chunks in vector store, trace_id: {trace_id}")
        start_time = datetime.datetime.now()
        store_docs = []
        for doc in documents:
            store_docs.append({
                'text': doc.get('text', ''),
                'metadata': doc.get('metadata', {})
            })
        doc_ids = await self.vector_store.add_documents(store_docs)
        storage_time = (datetime.datetime.now() - start_time).total_seconds()
        self.logger.info(f"Stored {len(doc_ids)} document chunks in {storage_time:.2f}s, trace_id: {trace_id}")
    async def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the vector store.
        Raises:
            RuntimeError: If there's an error clearing the knowledge base.
        """
        try:
            self.logger.info("Clearing knowledge base")
            await self.vector_store.clear()
            self.logger.info("Knowledge base cleared successfully")
        except Exception as e:
            error_msg = f"Error clearing knowledge base: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    async def save_uploaded_file(self, file: Any, filename: str) -> Path:
        """
        Save an uploaded file to the upload directory.
        Args:
            file: The uploaded file object.
            filename: The original filename.
        Returns:
            Path to the saved file.
        Raises:
            RuntimeError: If there's an error saving the file.
        """
        try:
            file_ext = Path(filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = self.upload_dir / unique_filename
            with open(file_path, "wb") as f:
                content = file.read()
                if hasattr(content, 'decode'):
                    content = content.decode('utf-8')
                f.write(content)
            self.logger.info(f"Saved uploaded file: {file_path}")
            return file_path
        except Exception as e:
            error_msg = f"Error saving uploaded file {filename}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)