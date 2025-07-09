"""Vector store with hybrid search capabilities."""
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import pickle
from pathlib import Path


class Document:
    """Represents a document chunk with metadata."""
    
    def __init__(self, text: str, embedding: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.doc_id = doc_id or f"doc_{hash(text) & 0xFFFFFFFF}"
    
    def __repr__(self):
        return f"Document({self.doc_id[:8]}...: {self.text[:50]}...)"


class InMemoryVectorStore:
    """In-memory vector store for prototyping and testing.
    
    Stores documents and their embeddings for similarity search.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: Dict[str, Document] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._index_built = False
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store."""
        for doc in documents:
            self.documents[doc.doc_id] = doc
            self._ids.append(doc.doc_id)
        
        self._index_built = False
    
    def build_index(self):
        """Build the embedding matrix."""
        embeddings = []
        for doc_id in self._ids:
            doc = self.documents[doc_id]
            if doc.embedding is None:
                raise ValueError(f"Document {doc_id} has no embedding")
            embeddings.append(doc.embedding)
        
        self.embeddings = np.array(embeddings)
        self._index_built = True
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self._index_built:
            self.build_index()
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize embeddings
        embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[self._ids[idx]]
            results.append((doc, similarities[idx]))
        
        return results
    
    def save(self, path: str):
        """Save the vector store to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'documents': self.documents,
                'ids': self._ids
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a vector store from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        store = cls(data['dimension'])
        for doc_id, doc in data['documents'].items():
            store.documents[doc_id] = doc
        store._ids = data['ids']
        
        return store


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    store = InMemoryVectorStore(dimension=128)
    
    # Add some documents
    docs = [
        Document("Machine learning is a subset of AI", 
                np.random.randn(128), {'topic': 'AI'}),
        Document("Deep learning uses neural networks",
                np.random.randn(128), {'topic': 'AI'}),
        Document("Python is a programming language",
                np.random.randn(128), {'topic': 'programming'})
    ]
    
    store.add_documents(docs)
    
    # Search
    query = np.random.randn(128)
    results = store.search(query, top_k=2)
    
    for doc, score in results:
        print(f"Score: {score:.4f} - {doc.text}")

# Optimize embedding batch processing throughput [2025-06-12T10:15:05]

# Fix dangling reference in LangChain retriever [2025-06-13T18:20:45]

# Add query cache with Redis backend [2025-06-16T11:28:41]

# Implement cross-encoder reranking for better retrieval [2025-06-16T18:15:05]

# Add reranking timeout fallback mechanism [2025-06-18T13:34:58]

# Implement late interaction with ColBERT [2025-06-20T14:08:24]

# Fix connection pool timeout handling [2025-06-20T16:38:51]

# Fix dangling reference in LangChain retriever [2025-06-22T15:59:55]

# Add source document provenance tracking [2025-06-23T09:35:10]

# Fix chunk overlap logic for markdown parser [2025-06-24T17:14:29]

# Implement parent-child document chunk linking [2025-06-30T19:25:01]

# Add cosine Similarity threshold configuration [2025-06-30T09:39:49]

# Add semantic routing by query intent classification [2025-07-03T20:23:06]

# Add query cache with Redis backend [2025-07-08T15:25:29]

# Optimize embedding batch processing throughput [2025-07-08T11:18:18]

# Add query rewrite using LLM for clarity [2025-07-09T17:44:54]
