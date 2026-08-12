"""
Embedding infrastructure for Famou 2.0.

This module provides embedding generation capabilities for semantic similarity,
diversity-based selection, and novelty search in evolutionary optimization.

Available Clients:
    - OpenAIEmbeddingClient: Production OpenAI embeddings (text-embedding-3-*)
    - MockEmbeddingClient: Testing with deterministic embeddings
    - SimpleMockEmbeddingClient: Simple one-hot style embeddings for basic tests

Protocol:
    - EmbeddingClient: Interface that all clients implement

Example:
    >>> from famou.infrastructure.embedding import OpenAIEmbeddingClient
    >>> client = OpenAIEmbeddingClient(api_key="sk-...", model="text-embedding-3-small")
    >>> response = client.embed("def sort(arr): return sorted(arr)")
    >>> print(response.dimensions)  # 1536
    >>> vector = response.to_numpy()  # numpy array
"""

from famou.infrastructure.embedding.base import (
    BatchEmbeddingResponse,
    EmbeddingClient,
    EmbeddingResponse,
)
from famou.infrastructure.embedding.mock_embedding import (
    MockEmbeddingClient,
)
from famou.infrastructure.embedding.openai_embedding import OpenAIEmbeddingClient

__all__ = [
    # Protocol and data models
    "EmbeddingClient",
    "EmbeddingResponse",
    "BatchEmbeddingResponse",
    # Implementations
    "OpenAIEmbeddingClient",
    "MockEmbeddingClient",
]
