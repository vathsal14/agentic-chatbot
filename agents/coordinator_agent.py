from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent, MCPMessage  
import logging
class CoordinatorAgent(BaseAgent):
    """Coordinates between different agents in the system"""
    def __init__(self, mcp_server, agents: Dict[str, Any]):
        """
        Initialize the Coordinator Agent
        Args:
            mcp_server: Reference to the MCP server for message routing
            agents: Dictionary of available agents
        """
        super().__init__("coordinator_agent", mcp_server)
        self.agents = agents
        self.logger = logging.getLogger(__name__)
        self.setup_handlers()
        self.logger.info("CoordinatorAgent initialized")
    def setup_handlers(self):
        """Set up message handlers for this agent"""
        self.register_handler("USER_QUERY", self.handle_user_query)
        self.register_handler("UPLOAD_DOCUMENT", self.handle_upload_document)
        self.register_handler("DOCUMENT_INGESTED", self.handle_document_ingested)
        self.register_handler("SEARCH_KNOWLEDGE_BASE", self.handle_search_knowledge_base)
        self.register_handler("SEARCH_RESULTS", self.handle_search_results)
        self.register_handler("GENERATE_RESPONSE", self.handle_generate_response)
        self.register_handler("RESPONSE_GENERATED", self.handle_response_generated)
        self.register_handler("ERROR", self.handle_error)
    async def handle_user_query(self, message: MCPMessage) -> None:
        """Handle a user query by searching the knowledge base and generating a response"""
        try:
            query = message.payload.get("query")
            if not query:
                raise ValueError("No query provided in the message payload")
            search_message = MCPMessage(
                sender=self.agent_id,
                receiver="retrieval_agent",
                message_type="SEARCH_KNOWLEDGE_BASE",
                trace_id=message.trace_id,
                payload={"query": query}
            )
            await self.send_message(search_message)
        except Exception as e:
            self.logger.error(f"Error handling user query: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_upload_document(self, message: MCPMessage) -> None:
        """Handle document upload by sending it to the ingestion agent"""
        try:
            file_path = message.payload.get("file_path")
            if not file_path:
                raise ValueError("No file path provided in the message payload")
            ingest_message = MCPMessage(
                sender=self.agent_id,
                receiver="ingestion_agent",
                message_type="INGEST_DOCUMENT",
                trace_id=message.trace_id,
                payload={"file_path": file_path}
            )
            await self.send_message(ingest_message)
        except Exception as e:
            self.logger.error(f"Error handling document upload: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_document_ingested(self, message: MCPMessage) -> None:
        """Handle the response from the ingestion agent"""
        try:
            self.logger.info(f"Document ingested: {message.payload}")
        except Exception as e:
            self.logger.error(f"Error handling document ingested: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_search_knowledge_base(self, message: MCPMessage) -> None:
        """Handle search knowledge base request"""
        try:
            search_message = MCPMessage(
                sender=self.agent_id,
                receiver="retrieval_agent",
                message_type="SEARCH_KNOWLEDGE_BASE",
                trace_id=message.trace_id,
                payload=message.payload
            )
            await self.send_message(search_message)
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_search_results(self, message: MCPMessage) -> None:
        """Handle search results from the retrieval agent"""
        try:
            response_message = MCPMessage(
                sender=self.agent_id,
                receiver="response_agent",
                message_type="GENERATE_RESPONSE",
                trace_id=message.trace_id,
                payload={
                    "query": message.payload.get("query"),
                    "context": message.payload.get("results", [])
                }
            )
            await self.send_message(response_message)
        except Exception as e:
            self.logger.error(f"Error handling search results: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_generate_response(self, message: MCPMessage) -> None:
        """Handle generate response request"""
        try:
            response_message = MCPMessage(
                sender=self.agent_id,
                receiver="response_agent",
                message_type="GENERATE_RESPONSE",
                trace_id=message.trace_id,
                payload=message.payload
            )
            await self.send_message(response_message)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_response_generated(self, message: MCPMessage) -> None:
        """Handle response from the response agent"""
        try:
            self.logger.info(f"Response generated: {message.payload.get('response')}")
            print(f"\nResponse: {message.payload.get('response')}")
        except Exception as e:
            self.logger.error(f"Error handling generated response: {str(e)}")
            await self.handle_error(e, message.trace_id)
    async def handle_error(self, error: Exception, trace_id: Optional[str] = None) -> None:
        """Handle errors in the coordinator"""
        error_message = f"Error in CoordinatorAgent: {str(error)}"
        self.logger.error(error_message)
        error_msg = MCPMessage(
            sender=self.agent_id,
            receiver="error_handler",
            message_type="ERROR",
            trace_id=trace_id or "unknown",
            payload={
                "error": str(error),
                "source": "CoordinatorAgent"
            }
        )
        print(f"\nERROR: {error_message}")