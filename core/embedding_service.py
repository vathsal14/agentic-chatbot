from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
class EmbeddingService:
    """Handles text embedding and similarity search."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """Initialize the embedding service with a pre-trained model."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings = []
        self.documents = []
    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        return self.model.encode(text, convert_to_numpy=True)
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector store."""
        if not text.strip():
            return
        embedding = self.embed_text(text)
        self.embeddings.append(embedding)
        self.documents.append({
            'text': text,
            'metadata': metadata or {}
        })
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents to the query."""
        if not self.embeddings:
            return []
        query_embedding = self.embed_text(query)
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['similarity'] = float(similarities[idx])
            results.append(doc)
        return results
    def clear(self):
        """Clear all stored documents and embeddings."""
        self.embeddings = []
        self.documents = []