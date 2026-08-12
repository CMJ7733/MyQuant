"""
Base embedding client protocol and data models.

Defines the interface that all embedding clients must implement, along with
standard response/request models.
"""

from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """
    Standardized response from embedding generation.

    This model encapsulates all data returned from an embedding call,
    providing a uniform interface regardless of the underlying provider.

    Fields:
        embedding: The embedding vector
        raw_response: Original response dict from the provider
        usage: Token usage statistics (for providers that charge per token)
        model: Model identifier that was used
        dimensions: Dimensionality of the embedding vector
    """

    embedding: List[float] = Field(description="Embedding vector as list")
    raw_response: Dict[str, Any] = Field(
        default_factory=dict, description="Raw provider response"
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage (prompt_tokens, total_tokens)",
    )
    model: str = Field(description="Model identifier used")
    dimensions: int = Field(description="Dimensionality of embedding vector")

    model_config = {"arbitrary_types_allowed": True}

    def to_numpy(self) -> np.ndarray:
        """Convert embedding list to numpy array."""
        return np.array(self.embedding, dtype=np.float32)

    def __repr__(self) -> str:
        """Concise representation."""
        return f"EmbeddingResponse(model={self.model}, dimensions={self.dimensions})"


class BatchEmbeddingResponse(BaseModel):
    """
    Response for batch embedding generation.

    Fields:
        embeddings: List of embedding vectors (2D array)
        raw_response: Original response dict from the provider
        usage: Aggregated token usage statistics
        model: Model identifier that was used
        dimensions: Dimensionality of embedding vectors
        count: Number of embeddings in batch
    """

    embeddings: List[List[float]] = Field(description="List of embedding vectors")
    raw_response: Dict[str, Any] = Field(
        default_factory=dict, description="Raw provider response"
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Aggregated token usage",
    )
    model: str = Field(description="Model identifier used")
    dimensions: int = Field(description="Dimensionality of embedding vectors")
    count: int = Field(description="Number of embeddings")

    model_config = {"arbitrary_types_allowed": True}

    def to_numpy(self) -> np.ndarray:
        """Convert embeddings to numpy array (shape: [count, dimensions])."""
        return np.array(self.embeddings, dtype=np.float32)

    def __repr__(self) -> str:
        """Concise representation."""
        return f"BatchEmbeddingResponse(model={self.model}, count={self.count}, dimensions={self.dimensions})"


class EmbeddingClient(Protocol):
    """
    Protocol for embedding client implementations.

    All embedding clients (OpenAI, local models, Mock, etc.) must implement this interface.
    This ensures consistent API across different providers.

    The interface is designed to be:
    - Simple: Single embed() method for most use cases
    - Efficient: Batch processing support via embed_batch()
    - Type-safe: Returns structured response objects
    - Observable: Includes usage statistics for cost tracking

    Example Implementation:
        >>> class MyEmbeddingClient:
        ...     def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        ...         # Call provider API
        ...         vector = my_provider.embed(text)
        ...         return EmbeddingResponse(
        ...             embedding=vector,
        ...             model="my-model",
        ...             dimensions=len(vector),
        ...             usage={...}
        ...         )
        ...
        ...     def embed_batch(self, texts: List[str], **kwargs) -> BatchEmbeddingResponse:
        ...         # Batch processing for efficiency
        ...         vectors = my_provider.embed_many(texts)
        ...         return BatchEmbeddingResponse(
        ...             embeddings=vectors,
        ...             model="my-model",
        ...             dimensions=len(vectors[0]),
        ...             count=len(vectors),
        ...             usage={...}
        ...         )

    Example Usage:
        >>> client: EmbeddingClient = OpenAIEmbeddingClient(api_key="...")
        >>> response = client.embed("def sort(arr): return sorted(arr)")
        >>> print(response.embedding[:5])  # First 5 dimensions
        >>> print(f"Dimensions: {response.dimensions}")
        >>> print(f"Cost: {response.usage['total_tokens']} tokens")
        >>>
        >>> # Batch processing
        >>> batch_response = client.embed_batch([
        ...     "code snippet 1",
        ...     "code snippet 2",
        ...     "code snippet 3"
        ... ])
        >>> embeddings = batch_response.to_numpy()  # Shape: (3, dimensions)
    """

    def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        """
        Generate embedding for a single text.

        This is the primary method for generating embeddings.
        All implementations must support this method.

        Args:
            text: The text to embed (code, prompt, or any string)
            **kwargs: Provider-specific options
                - model: Override default model
                - encoding_format: Embedding format (float, base64, etc.)
                - user: User identifier for tracking
                - etc.

        Returns:
            EmbeddingResponse with embedding vector and metadata

        Raises:
            Exception: Implementation-specific errors (network, quota, etc.)

        Example:
            >>> response = client.embed("def hello(): return 'world'")
            >>> vector = response.to_numpy()  # numpy array
            >>> print(vector.shape)  # (1536,) for text-embedding-3-small
        """
        ...

    def embed_batch(
        self, texts: List[str], batch_size: Optional[int] = None, **kwargs
    ) -> BatchEmbeddingResponse:
        """
        Generate embeddings for multiple texts efficiently.

        Batch processing is more efficient than calling embed() multiple times,
        especially for API-based providers that support batch requests.

        Args:
            texts: List of texts to embed
            batch_size: Maximum batch size (if provider has limits)
            **kwargs: Provider-specific options

        Returns:
            BatchEmbeddingResponse with list of embedding vectors

        Raises:
            Exception: Implementation-specific errors

        Example:
            >>> texts = ["code1", "code2", "code3"]
            >>> response = client.embed_batch(texts)
            >>> embeddings = response.to_numpy()  # Shape: (3, dimensions)
            >>> print(f"Generated {response.count} embeddings")
        """
        ...

    @property
    def model_name(self) -> str:
        """
        Get the model name used by this client.

        Returns:
            Model identifier (e.g., "text-embedding-3-small")
        """
        ...

    @property
    def dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this client.

        Returns:
            Embedding dimensions (e.g., 1536 for text-embedding-3-small)
        """
        ...
