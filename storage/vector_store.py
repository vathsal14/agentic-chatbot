from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os
class VectorStore:
    """
    A vector store implementation using sentence-transformers for embeddings.
    This provides better semantic search capabilities than the dummy implementation.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector store with a sentence-transformers model.
        Args:
            model_name: Name of the sentence-transformers model to use for embeddings.
                       Default is 'all-MiniLM-L6-v2' which is a good general-purpose model.
        """
        self.vectors = {}
        self.metadata = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
    async def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to the store with associated metadata.
        Args:
            vectors: List of vector embeddings to add
            metadatas: List of metadata dictionaries corresponding to each vector
            ids: Optional list of IDs for each vector. If not provided, will generate UUIDs.
        Returns:
            List of IDs for the added vectors
        """
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        for vec_id, vector, metadata in zip(ids, vectors, metadatas):
            self.vectors[vec_id] = np.array(vector, dtype=np.float32)
            self.metadata[vec_id] = metadata
        return ids
    async def similarity_search(
        self, 
        query_text: str, 
        k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find the k most similar documents to the query text.
        Args:
            query_text: The query text to search for
            k: Number of results to return
            filter_condition: Optional filter to apply to metadata
        Returns:
            List of dictionaries containing 'id', 'text', 'metadata', and 'similarity' score
        Raises:
            ValueError: If no query text is provided
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        try:
            query_embedding = self.model.encode(
                query_text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            query_vector = query_embedding.tolist()
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}", exc_info=True)
            raise
        if not self.vectors:
            return []
        query_vec = np.array(query_vector, dtype=np.float32)
        results = []
        for vec_id, vector in self.vectors.items():
            metadata = self.metadata[vec_id]
            if filter_condition:
                if not all(
                    metadata.get(k) == v 
                    for k, v in filter_condition.items()
                ):
                    continue
            similarity = np.dot(query_vec, vector) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vector) + 1e-10
            )
            results.append({
                'id': vec_id,
                'vector': vector,
                'metadata': metadata,
                'similarity': float(similarity)
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by their IDs.
        Args:
            ids: List of vector IDs to delete
        Returns:
            True if all vectors were deleted successfully, False otherwise
        """
        success = True
        for vec_id in ids:
            try:
                del self.vectors[vec_id]
                del self.metadata[vec_id]
            except KeyError:
                success = False
        return success
    async def clear(self) -> None:
        """
        Clear all vectors and metadata from the store.
        Returns:
            None
        """
        self.vectors.clear()
        self.metadata.clear()
    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        Returns:
            int: Number of documents in the store
        """
        return len(self.vectors)
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store with automatic embedding generation.
        Args:
            documents: List of documents, where each document is a dict with 'text' and 'metadata'
        Returns:
            List of document IDs
        Raises:
            ValueError: If no text is provided in a document
        """
        if not documents:
            self.logger.warning("No documents provided to add_documents")
            return []
        texts = []
        metadatas = []
        for doc in documents:
            text = doc.get('text', '').strip()
            if not text:
                self.logger.warning("Skipping document with empty text")
                continue
            texts.append(text)
            metadatas.append(doc.get('metadata', {}))
        if not texts:
            self.logger.warning("No valid texts found in documents")
            return []
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} text chunks")
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            vectors = [vector.tolist() for vector in vectors]
            return await self.add_vectors(vectors, metadatas)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise