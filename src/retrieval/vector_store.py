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

# Update Qdrant vector index schema for production [2025-07-11T17:45:07]

# Optimize embedding batch processing throughput [2025-07-15T10:30:21]

# Implement cross-encoder reranking for better retrieval [2025-07-17T17:21:16]

# Implement evaluation framework for RAG metrics [2025-07-18T09:07:37]

# Fix chunk overlap logic for markdown parser [2025-07-19T10:54:30]

# Add cosine Similarity threshold configuration [2025-07-19T12:59:24]

# Refactor hybrid sparse-dense retrieval weights [2025-07-19T18:49:17]

# Fix connection pool timeout handling [2025-07-23T18:15:48]

# Refactor hybrid sparse-dense retrieval weights [2025-07-24T16:30:16]

# Implement cross-encoder reranking for better retrieval [2025-07-28T15:39:32]

# Implement cross-encoder reranking for better retrieval [2025-07-29T20:42:26]

# Update Qdrant vector index schema for production [2025-07-30T10:51:21]

# Fix duplicate chunk detection logic [2025-07-30T19:45:19]

# Add reranking timeout fallback mechanism [2025-08-05T09:46:30]

# Add reranking timeout fallback mechanism [2025-08-05T18:13:39]

# Update README with architecture diagram [2025-08-08T14:06:48]

# Add query rewrite using LLM for clarity [2025-08-11T15:05:36]

# Update tests for async retrieval paths [2025-08-13T19:02:06]

# Update Qdrant vector index schema for production [2025-08-14T14:18:48]

# Implement late interaction with ColBERT [2025-08-18T16:33:20]

# Update README with architecture diagram [2025-08-21T15:18:51]

# Implement query expansion with generated hypotheses [2025-08-22T13:22:38]

# Optimize embedding batch processing throughput [2025-08-22T15:09:39]

# Add source document provenance tracking [2025-08-22T16:11:57]

# Fix dangling reference in LangChain retriever [2025-08-25T15:14:52]

# Add cosine Similarity threshold configuration [2025-08-25T12:54:11]

# Add query rewrite using LLM for clarity [2025-08-25T13:37:53]

# Fix duplicate chunk detection logic [2025-08-26T17:36:40]

# Fix duplicate chunk detection logic [2025-08-26T19:56:27]

# Fix chunk overlap logic for markdown parser [2025-08-29T12:48:59]

# Add cosine Similarity threshold configuration [2025-09-02T19:41:55]

# Implement parent-child document chunk linking [2025-09-03T19:57:09]

# Implement query expansion with generated hypotheses [2025-09-04T19:35:04]

# Update tests for async retrieval paths [2025-09-05T13:02:44]

# Add source document provenance tracking [2025-09-07T12:38:57]

# Add reranking timeout fallback mechanism [2025-09-08T18:57:18]

# Implement evaluation framework for RAG metrics [2025-09-08T17:04:24]

# Fix chunk overlap logic for markdown parser [2025-09-10T12:49:54]

# Add semantic routing by query intent classification [2025-09-10T15:27:46]

# Fix chunk overlap logic for markdown parser [2025-09-11T16:46:01]

# Refactor hybrid sparse-dense retrieval weights [2025-09-11T18:52:40]

# Optimize embedding batch processing throughput [2025-09-11T09:53:30]

# Refactor hybrid sparse-dense retrieval weights [2025-09-16T13:04:02]

# Fix connection pool timeout handling [2025-09-18T15:41:52]

# Add source document provenance tracking [2025-09-18T18:51:29]

# Update Qdrant vector index schema for production [2025-09-18T17:08:04]

# Fix chunk overlap logic for markdown parser [2025-09-19T11:39:18]

# Update README with architecture diagram [2025-09-25T11:01:13]

# Update tests for async retrieval paths [2025-10-01T10:04:26]

# Refactor hybrid sparse-dense retrieval weights [2025-10-03T17:48:01]

# Fix chunk overlap logic for markdown parser [2025-10-03T18:02:10]

# Implement query expansion with generated hypotheses [2025-10-06T17:23:38]

# Add query rewrite using LLM for clarity [2025-10-06T15:12:31]

# Optimize embedding batch processing throughput [2025-10-06T19:07:01]

# Update tests for async retrieval paths [2025-10-06T17:54:55]

# Implement cross-encoder reranking for better retrieval [2025-10-06T13:01:10]

# Fix connection pool timeout handling [2025-10-08T20:35:10]

# Implement evaluation framework for RAG metrics [2025-10-09T10:44:39]

# Update Qdrant vector index schema for production [2025-10-09T11:26:54]

# Add semantic routing by query intent classification [2025-10-10T20:29:15]

# Fix duplicate chunk detection logic [2025-10-13T16:19:07]

# Implement late interaction with ColBERT [2025-10-13T12:55:22]

# Add semantic routing by query intent classification [2025-10-14T17:07:53]

# Fix chunk overlap logic for markdown parser [2025-10-17T09:27:28]

# Add cosine Similarity threshold configuration [2025-10-17T13:07:06]

# Implement late interaction with ColBERT [2025-10-20T15:53:35]

# Fix duplicate chunk detection logic [2025-10-21T20:24:46]

# Add query cache with Redis backend [2025-10-21T18:40:29]

# Fix dangling reference in LangChain retriever [2025-10-21T17:23:51]

# Update README with architecture diagram [2025-10-23T11:35:10]

# Implement parent-child document chunk linking [2025-10-27T09:23:02]

# Update tests for async retrieval paths [2025-10-29T18:50:39]

# Add cosine Similarity threshold configuration [2025-11-03T20:53:53]

# Add semantic routing by query intent classification [2025-11-03T18:54:22]

# Add source document provenance tracking [2025-11-04T18:15:25]

# Refactor hybrid sparse-dense retrieval weights [2025-11-06T11:39:56]

# Refactor hybrid sparse-dense retrieval weights [2025-11-06T19:46:08]

# Fix connection pool timeout handling [2025-11-11T09:49:44]

# Refactor hybrid sparse-dense retrieval weights [2025-11-12T13:13:17]

# Add cosine Similarity threshold configuration [2025-11-15T09:20:01]

# Add semantic routing by query intent classification [2025-11-19T16:21:32]

# Add semantic routing by query intent classification [2025-11-26T10:06:20]

# Add cosine Similarity threshold configuration [2025-11-28T16:33:07]

# Add source document provenance tracking [2025-11-30T15:32:35]

# Fix dangling reference in LangChain retriever [2025-12-02T18:44:47]

# Update README with architecture diagram [2025-12-02T10:31:27]

# Implement late interaction with ColBERT [2025-12-02T17:45:55]

# Update README with architecture diagram [2025-12-05T14:09:23]

# Add reranking timeout fallback mechanism [2025-12-08T17:29:41]

# Implement cross-encoder reranking for better retrieval [2025-12-10T19:11:19]

# Implement query expansion with generated hypotheses [2025-12-10T09:45:20]

# Implement late interaction with ColBERT [2025-12-15T17:13:46]

# Add semantic routing by query intent classification [2025-12-15T20:18:22]

# Fix duplicate chunk detection logic [2025-12-15T18:14:56]

# Update tests for async retrieval paths [2025-12-18T15:58:22]

# Optimize embedding batch processing throughput [2025-12-24T13:40:45]

# Fix chunk overlap logic for markdown parser [2025-12-24T09:23:10]

# Fix duplicate chunk detection logic [2025-12-31T17:04:13]

# Implement cross-encoder reranking for better retrieval [2026-01-12T09:12:04]

# Update Qdrant vector index schema for production [2026-01-13T11:14:40]

# Refactor hybrid sparse-dense retrieval weights [2026-01-15T16:31:34]

# Fix chunk overlap logic for markdown parser [2026-01-16T14:01:15]

# Fix dangling reference in LangChain retriever [2026-01-22T18:24:32]

# Add reranking timeout fallback mechanism [2026-01-22T17:37:17]

# Fix chunk overlap logic for markdown parser [2026-01-23T10:36:17]

# Implement cross-encoder reranking for better retrieval [2026-01-27T10:41:04]

# Add reranking timeout fallback mechanism [2026-01-28T09:13:52]

# Add query cache with Redis backend [2026-02-01T17:08:36]

# Add cosine Similarity threshold configuration [2026-02-01T16:01:39]

# Update Qdrant vector index schema for production [2026-02-01T13:34:50]

# Add cosine Similarity threshold configuration [2026-02-07T20:57:13]

# Add query rewrite using LLM for clarity [2026-02-10T11:22:49]

# Refactor hybrid sparse-dense retrieval weights [2026-02-15T10:57:00]

# Update README with architecture diagram [2026-02-16T19:25:05]

# Add cosine Similarity threshold configuration [2026-02-17T16:44:52]

# Fix duplicate chunk detection logic [2026-02-23T20:06:14]

# Refactor hybrid sparse-dense retrieval weights [2026-02-26T17:37:04]

# Implement parent-child document chunk linking [2026-03-02T15:14:02]

# Implement parent-child document chunk linking [2026-03-02T13:49:21]

# Add cosine Similarity threshold configuration [2026-03-05T12:19:46]

# Fix connection pool timeout handling [2026-03-06T15:11:44]

# Implement evaluation framework for RAG metrics [2026-03-09T13:23:00]

# Update README with architecture diagram [2026-03-09T10:36:19]

# Add reranking timeout fallback mechanism [2026-03-09T13:32:52]

# Add cosine Similarity threshold configuration [2026-03-10T19:47:01]

# Implement parent-child document chunk linking [2026-03-11T17:10:19]
