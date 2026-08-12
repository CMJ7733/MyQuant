"""Embedding-based feature extraction module for Famou 2.0."""

from typing import TYPE_CHECKING

from famou.core.data import Context, RolloutResult
from famou.core.protocol import RequiresEmbedding
from famou.modules.judge.base import JudgeModule

if TYPE_CHECKING:
    from famou.infrastructure.embedding.base import EmbeddingClient


class EmbeddingFeature(JudgeModule, RequiresEmbedding):
    """
    Extract features using embedding models.

    Uses an embedding client (OpenAI, local model, etc.) to generate
    semantic embeddings of the program code and writes to program.feature_vector.

    This is a JudgeModule that enriches programs with feature vectors.
    """

    embedding_client: "EmbeddingClient"  # Injected by RolloutEngine

    def judge(self, context: Context, result: RolloutResult) -> None:
        """
        Extract embedding and write to program.feature_vector.

        Uses key_features from LLM feedback if available, otherwise falls back
        to the program's source code.
        """
        _ = context  # Unused but required by protocol
        program = result.generated_program

        # Try to get key_features from llm_feedback, fallback to code
        text = None
        if program.llm_feedback and isinstance(program.llm_feedback, dict):
            text = program.llm_feedback.get("key_features")

        # Fallback to program code if no key_features
        if not text:
            text = program.code

        # Final fallback for empty code
        if not text:
            self.log_warning("Empty text for embedding, using placeholder")
            text = "# empty"

        # Generate embedding
        self.log_info("Generating embedding")
        response = self.embedding_client.embed(text)

        # Write to program.feature_vector (enrichment in-place)
        program.feature_vector = response.embedding

        self.log_info(
            f"Extracted embedding features for {program.id}",
            program_id=program.id,
            feature_dim=len(response.embedding),
        )
