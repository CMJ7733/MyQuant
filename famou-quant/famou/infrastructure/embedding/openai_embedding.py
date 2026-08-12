"""
OpenAI Embedding client implementation.

Provides production-ready OpenAI Embeddings API integration with:
- Support for latest embedding models (text-embedding-3-small, text-embedding-3-large)
- Automatic retry logic for transient failures
- Rate limit handling
- Batch processing for efficiency
- Token usage tracking
"""

from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from famou.infrastructure.embedding.base import (
    BatchEmbeddingResponse,
    EmbeddingResponse,
)


class OpenAIEmbeddingClient:
    """
    OpenAI Embeddings API client with robust error handling.
    """
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        api_base: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **default_kwargs,
    ):
        """
        Initialize OpenAI embedding client.

        Args:
            api_key: OpenAI API key
            model: Model identifier (text-embedding-3-small, text-embedding-3-large, etc.)
            api_base: Optional API base URL (for proxies or compatible APIs)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **default_kwargs: Additional default parameters for API calls

        """

        # Initialize OpenAI client
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)
        self._api_key = api_key
        self._api_base = api_base
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_kwargs = default_kwargs

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model

    def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        """
        Generate embedding for a single text.

        Automatically retries on:
        - Rate limit errors (429)
        - Server errors (500, 502, 503, 504)
        - Timeout errors

        Args:
            text: Text to embed (code, prompt, or any string)
            **kwargs: Additional OpenAI API parameters
                - model: Override default model
                - encoding_format: "float" or "base64"
                - user: User identifier for tracking

        Returns:
            EmbeddingResponse with embedding vector and metadata

        Raises:
            OpenAI API errors after retries exhausted

        Example:
            >>> response = client.embed("def hello(): return 'world'")
            >>> vector = response.to_numpy()
            >>> print(vector.shape)  # (1536,)
        """
        # Merge parameters
        params = {
            "model": self.model,
            "input": text,
            **self.default_kwargs,
            **kwargs,
        }


        # Call API with retry logic
        response = self._call_with_retry(**params)

        # Extract embedding data
        data = response.data[0]
        embedding = data.embedding

        return EmbeddingResponse(
            embedding=embedding,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            dimensions=len(embedding),
        )

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> BatchEmbeddingResponse:
        """
        Generate embeddings for multiple texts efficiently.

        OpenAI supports up to 2048 inputs per request. If more texts are provided,
        they will be automatically split into multiple batches.

        Args:
            texts: List of texts to embed
            batch_size: Maximum batch size (default: 2048, OpenAI's limit)
            **kwargs: Additional OpenAI API parameters

        Returns:
            BatchEmbeddingResponse with list of embedding vectors

        Raises:
            ValueError: If texts is empty
            OpenAI API errors after retries exhausted

        Example:
            >>> texts = ["code1", "code2", "code3"]
            >>> response = client.embed_batch(texts)
            >>> embeddings = response.to_numpy()  # Shape: (3, 1536)
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        # Default batch size (OpenAI limit is 2048)
        batch_size = batch_size or 2048

        # If texts fit in single batch, call API once
        if len(texts) <= batch_size:
            return self._embed_batch_single(texts, **kwargs)

        # Otherwise, split into multiple batches
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        model_name = None

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_response = self._embed_batch_single(batch, **kwargs)

            all_embeddings.extend(batch_response.embeddings)
            total_usage["prompt_tokens"] += batch_response.usage.get("prompt_tokens", 0)
            total_usage["total_tokens"] += batch_response.usage.get("total_tokens", 0)
            model_name = batch_response.model

        return BatchEmbeddingResponse(
            embeddings=all_embeddings,
            raw_response={"batched": True, "num_batches": (len(texts) + batch_size - 1) // batch_size},
            usage=total_usage,
            model=model_name or self.model,
            dimensions=len(all_embeddings[0]),
            count=len(all_embeddings),
        )

    def _embed_batch_single(self, texts: List[str], **kwargs) -> BatchEmbeddingResponse:
        """
        Generate embeddings for a batch of texts (single API call).

        Args:
            texts: List of texts (must be <= batch_size)
            **kwargs: Additional API parameters

        Returns:
            BatchEmbeddingResponse with embeddings
        """
        # Merge parameters
        params = {
            "model": self.model,
            "input": texts,
            **self.default_kwargs,
            **kwargs,
        }

        # Call API with retry logic
        response = self._call_with_retry(**params)

        # Extract embeddings (preserve order)
        embeddings = [data.embedding for data in response.data]

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            dimensions=len(embeddings[0]),
            count=len(embeddings),
        )

    def _call_with_retry(self, **params):
        """
        Call OpenAI Embeddings API with automatic retry.

        Uses exponential backoff starting at 1 second and doubles on each retry.

        Args:
            **params: Parameters to pass to embeddings.create()

        Returns:
            OpenAI Embeddings response

        Raises:
            OpenAI API errors after max retries
        """
        retrying = Retrying(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(max(1, int(self.max_retries))),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self.client.embeddings.create(**params)
        raise RuntimeError("Embedding retry loop terminated unexpectedly")

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"OpenAIEmbeddingClient(model={self.model}, "
        )

    def __getstate__(self):
        """Pickle: exclude the httpx-backed OpenAI client (contains _thread.RLock)."""
        state = self.__dict__.copy()
        del state["client"]
        return state

    def __setstate__(self, state):
        """Unpickle: restore all fields then rebuild the OpenAI client."""
        self.__dict__.update(state)
        client_kwargs: Dict[str, Any] = {"api_key": self._api_key, "timeout": self.timeout}
        if self._api_base:
            client_kwargs["base_url"] = self._api_base
        self.client = OpenAI(**client_kwargs)
