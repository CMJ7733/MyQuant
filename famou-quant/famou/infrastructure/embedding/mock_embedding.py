"""
Mock embedding client for testing.

Provides deterministic embeddings without making real API calls.
Useful for:
- Unit testing
- Integration testing
- Local development without API costs
- Reproducible experiments
"""

import hashlib
from typing import List, Optional

import numpy as np

from famou.infrastructure.embedding.base import (
    BatchEmbeddingResponse,
    EmbeddingResponse,
)


class MockEmbeddingClient:
    """
    Mock embedding client that generates deterministic random embeddings.

    This client generates embeddings based on hash of input text, making it
    perfect for testing without incurring API costs or network latency.

    Features:
    - Deterministic: Same input text → same embedding
    - Fast: No network calls, pure computation
    - Free: No API costs
    - Configurable: Adjustable dimensions
    - Realistic: Normalized vectors similar to real embeddings

    Examples:
        >>> # Create mock client
        >>> client = MockEmbeddingClient(dimensions=1536)
        >>> response = client.embed("def hello(): return 'world'")
        >>> response.dimensions
        1536
        >>> vector = response.to_numpy()
        >>> np.linalg.norm(vector)  # Normalized to unit length
        1.0
        >>>
        >>> # Batch processing
        >>> batch_response = client.embed_batch(["code1", "code2", "code3"])
        >>> embeddings = batch_response.to_numpy()  # Shape: (3, 1536)
    """

    def __init__(
        self,
        dimensions: int = 1536,
        model: str = "mock-embedding",
        seed: Optional[int] = None,
    ):
        """
        Initialize mock embedding client.

        Args:
            dimensions: Dimensionality of embeddings
            model: Model name to report in responses
            seed: Random seed for reproducibility
        """
        self._dimensions = dimensions
        self.model = model
        self.seed = seed
        self.call_count = 0

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._dimensions

    def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        """
        Generate mock embedding for a single text.

        Args:
            text: Text to embed
            **kwargs: Additional args (ignored)

        Returns:
            EmbeddingResponse with deterministic embedding

        Example:
            >>> client = MockEmbeddingClient(dimensions=128)
            >>> response = client.embed("def sort(arr): return sorted(arr)")
            >>> vector = response.to_numpy()
            >>> vector.shape
            (128,)
        """
        self.call_count += 1

        # Generate deterministic embedding based on text hash
        embedding_array = self._generate_embedding(text)
        embedding_list = embedding_array.tolist()

        return EmbeddingResponse(
            embedding=embedding_list,
            raw_response={
                "id": f"mock-emb-{self.call_count}",
                "model": self.model,
                "object": "embedding",
            },
            usage={
                "prompt_tokens": len(text) // 4,
                "total_tokens": len(text) // 4,
            },
            model=self.model,
            dimensions=len(embedding_list),
        )

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> BatchEmbeddingResponse:
        """
        Generate mock embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Ignored (mock processes all at once)
            **kwargs: Additional args (ignored)

        Returns:
            BatchEmbeddingResponse with list of embeddings

        Raises:
            ValueError: If texts is empty

        Example:
            >>> client = MockEmbeddingClient(dimensions=128)
            >>> response = client.embed_batch(["code1", "code2", "code3"])
            >>> embeddings = response.to_numpy()  # Shape: (3, 128)
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        # Generate embeddings for all texts
        embeddings = []
        total_tokens = 0

        for text in texts:
            embedding_array = self._generate_embedding(text)
            embeddings.append(embedding_array.tolist())
            total_tokens += len(text) // 4

        self.call_count += len(texts)

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            raw_response={
                "id": f"mock-batch-{self.call_count}",
                "model": self.model,
                "object": "list",
            },
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
            model=self.model,
            dimensions=len(embeddings[0]) if embeddings else self._dimensions,
            count=len(embeddings),
        )

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding vector for text based on hash.

        Args:
            text: Input text

        Returns:
            Numpy array of shape (dimensions,), normalized to unit length
        """
        # Generate deterministic random vector based on hash of text
        # This ensures same text always gets same embedding
        text_hash = hashlib.sha256(text.encode()).digest()

        # Use hash as seed for deterministic randomness
        hash_int = int.from_bytes(text_hash[:8], byteorder='big')
        rng = np.random.RandomState((hash_int + (self.seed or 0)) % (2**32))

        # Generate random vector
        embedding = rng.randn(self._dimensions).astype(np.float32)

        # Normalize to unit length (like real embeddings)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def __repr__(self) -> str:
        """Concise representation."""
        return f"MockEmbeddingClient(dimensions={self._dimensions}, calls={self.call_count})"

