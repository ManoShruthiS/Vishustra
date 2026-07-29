import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGenerator(BaseNode):
    """
    A Vishustra node responsible for generating numerical embeddings from text data.

    This node takes text input (either a single string or a list of strings)
    and simulates the process of converting them into high-dimensional vector
    representations, suitable for downstream machine learning tasks like similarity
    search, clustering, or input to other LLM components.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> List[List[float]]:
        """
        Generates embeddings for the provided text data.

        The `data` input can be a single string or a list of strings. Each string
        will be processed to produce a corresponding embedding vector.

        The `context` dictionary can optionally specify parameters, such as
        `embedding_dimension`, which dictates the size of the generated vectors.

        Args:
            data: The text data to be embedded. Expected types are `str` for a single
                  document or `List[str]` for multiple documents.
            context: A dictionary containing operational parameters for the node.
                     E.g., `{"embedding_dimension": 768}`.

        Returns:
            A `List[List[float]]` where each inner list represents an embedding
            vector for a corresponding input text. If the input `data` was a
            single string, the output list will contain one embedding.

        Raises:
            ValueError: If the `data` is not a string or a list of strings,
                        or if a list contains non-string elements.
            RuntimeError: If an unexpected issue occurs during the embedding
                          generation process.
        """
        logger.debug(f"[{self.node_name}] Initiating embedding process for input of type: {type(data).__name__}")

        texts_to_embed: List[str]

        # --- Input Validation ---
        if isinstance(data, str):
            texts_to_embed = [data]
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                error_msg = (
                    f"[{self.node_name}] Invalid input list. Expected all elements "
                    f"to be strings, but found non-string types."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            texts_to_embed = data
        else:
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected `str` or "
                f"`List[str]`, received `{type(data).__name__}`."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not texts_to_embed:
            logger.warning(f"[{self.node_name}] No texts provided for embedding. Returning an empty list.")
            return []

        embeddings: List[List[float]] = []
        try:
            # --- Simulated Embedding Generation ---
            # In a production environment, this section would typically involve:
            # 1. Loading or selecting an appropriate embedding model (e.g., from context).
            # 2. Tokenizing the input texts.
            # 3. Performing inference using the chosen model to generate actual embeddings.
            # 4. Handling batching for efficiency if multiple texts are provided.

            # For this simulation, we generate a dummy embedding vector.
            # The dimension of the embedding can be configured via the context.
            embedding_dimension = context.get("embedding_dimension", 384)  # Common embedding dimension

            for i, text in enumerate(texts_to_embed):
                # Generate a simple, repeatable dummy embedding for demonstration.
                # In a real system, this would be a sophisticated model output.
                dummy_embedding = [(float(j) + (len(text) % 10) * 0.1) / embedding_dimension
                                   for j in range(embedding_dimension)]
                embeddings.append(dummy_embedding)
                logger.debug(
                    f"[{self.node_name}] Generated dummy embedding for text {i+1}/{len(texts_to_embed)} "
                    f"(first 50 chars): '{text[:50]}...'"
                )

            logger.info(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings.")
            return embeddings

        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during embedding generation: {e}"
            logger.exception(error_msg)  # Logs the full traceback for debugging
            raise RuntimeError(error_msg) from e
