"""An offline agentic retrieval-augmented generation pipeline.

Public entry points:

* :func:`src.ragpipe.corpus.build_corpus` -- documents, generated or read from disk.
* :class:`src.ragpipe.chunking.Chunker` -- fixed-size or sentence-aware chunking.
* :class:`src.ragpipe.bm25.BM25Index` -- lexical retrieval implemented from scratch.
* :class:`src.ragpipe.dense.DenseIndex` -- hashing TF-IDF vectors with cosine search.
* :class:`src.ragpipe.fusion.HybridIndex` -- reciprocal rank fusion of both.
* :class:`src.ragpipe.agent.RAGAgent` -- tool-using loop with a step budget.
* :func:`src.ragpipe.grounding.ground_answer` -- token-overlap grounding check.
* :func:`src.ragpipe.evaluation.evaluate_pipeline` -- recall@k and answer coverage.
"""

from .bm25 import BM25Index, tokenize
from .chunking import Chunk, Chunker, chunk_documents
from .corpus import Document, build_corpus
from .dense import DenseIndex, HashingTfIdfEmbedder
from .fusion import HybridIndex, reciprocal_rank_fusion

__all__ = [
    "BM25Index",
    "Chunk",
    "Chunker",
    "DenseIndex",
    "Document",
    "HashingTfIdfEmbedder",
    "HybridIndex",
    "build_corpus",
    "chunk_documents",
    "reciprocal_rank_fusion",
    "tokenize",
]
