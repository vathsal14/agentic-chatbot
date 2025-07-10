from typing import Dict, Any, List, Optional
from core.mcp import Message, MessageType
from agents.base_agent import BaseAgent
from storage.vector_store import VectorStore
class RetrievalAgent(BaseAgent):
    """Agent responsible for retrieving relevant document chunks based on queries"""
    def __init__(self, mcp_server, vector_store: VectorStore):
        """
        Initialize the Retrieval Agent
        Args:
            mcp_server: Reference to the MCP server
            vector_store: Vector store instance for document retrieval
        """
        super().__init__("retrieval_agent", mcp_server)
        self.vector_store = vector_store
    def setup_handlers(self):
        """Set up message handlers for this agent"""
        self.register_handler(MessageType.RETRIEVAL_REQUEST, self.handle_retrieval_request)
    async def handle_retrieval_request(self, message: Message):
        """
        Handle document retrieval requests
        Args:
            message: MCP message containing retrieval request
        """
        try:
            payload = message.payload
            query = payload.get("query")
            filter_metadata = payload.get("filter_metadata", {})
            top_k = payload.get("top_k", 5)
            if not query:
                raise ValueError("No query provided in retrieval request")
            results = await self.vector_store.similarity_search(
                query_text=query,
                k=top_k,
                filter_condition=filter_metadata
            )
            retrieved_chunks = []
            for result in results:
                chunk = {
                    "text": result["text"],
                    "score": result["score"],
                    "metadata": result["metadata"]
                }
                retrieved_chunks.append(chunk)
            await self.send_message(
                receiver_id=message.sender,
                message_type=MessageType.RETRIEVAL_RESPONSE,
                payload={
                    "status": "success",
                    "query": query,
                    "retrieved_chunks": retrieved_chunks,
                    "trace_id": message.trace_id
                }
            )
        except Exception as e:
            print(f"Error in retrieval agent: {e}")
            await self.handle_error(e, message.trace_id)