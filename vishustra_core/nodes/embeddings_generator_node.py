import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of vector embeddings for textual data.

    This node takes either a single string or a list of strings and returns
    corresponding mock embedding vectors. The embedding dimension is configurable
    via the node's initialization, or falls back to a default.
    """

    def __init__(self, embedding_dim: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dim (int): The dimension of the mock embedding vectors to generate.
                                 Defaults to 768, a common dimension for many models.
        Raises:
            ValueError: If `embedding_dim` is not a positive integer.
        """
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.error(f"Initialization failed: Invalid embedding_dim provided: {embedding_dim}. Must be a positive integer.")
            raise ValueError("Embedding dimension must be a positive integer.")
        self._embedding_dim = embedding_dim
        logger.debug(f"EmbeddingsGeneratorNode initialized with embedding_dim={self._embedding_dim}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generates a mock embedding vector for a given text.
        In a real scenario, this private method would encapsulate the call to an actual
        embedding model (e.g., from Hugging Face, OpenAI, Cohere, etc.) and handle its specifics.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: A list of floats representing the mock embedding vector.
        """
        # For simulation, we generate a list of random floats within a common range.
        # This range (-1.0 to 1.0) is arbitrary but often seen with normalized embeddings.
        return [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dim)]

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[List[float]]:
        """
        Processes the input data (text or list of texts) to generate mock embeddings.

        Args:
            data (Union[str, List[str]]): The input text(s) to generate embeddings for.
                                          Can be a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing additional context information.
                                      Currently not directly used for embedding generation parameters,
                                      but available for future extensions (e.g., model overrides,
                                      batching configurations, retry policies).

        Returns:
            List[List[float]]: A list of embedding vectors. Each vector is a list of floats.
                               If a single string was provided, a list containing one vector is returned.
                               Returns an empty list if no valid texts could be processed.

        Raises:
            ValueError: If the input data is `None`.
            TypeError: If the input data is not a string or a list of strings.
        """
        if data is None:
            logger.error("Input data received for EmbeddingsGeneratorNode is None. Cannot process.")
            raise ValueError("Input data cannot be None.")

        embeddings: List[List[float]] = []

        if isinstance(data, str):
            if not data.strip():
                logger.warning("Received an empty or whitespace-only string for embedding generation. Returning an empty list.")
                return []
            logger.info(f"Generating embedding for a single text input (length: {len(data)} characters).")
            embeddings.append(self._generate_mock_embedding(data))
        elif isinstance(data, list):
            if not data:
                logger.warning("Received an empty list for embedding generation. Returning an empty list.")
                return []

            valid_items_processed = 0
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.warning(f"Batch item at index {i} is not a string (type: {type(item).__name__}). Skipping this item.")
                    continue
                if not item.strip():
                    logger.warning(f"Batch item at index {i} is an empty or whitespace-only string. Skipping this item.")
                    continue

                logger.debug(f"Generating embedding for batch item {i} (length: {len(item)} characters).")
                try:
                    embeddings.append(self._generate_mock_embedding(item))
                    valid_items_processed += 1
                except Exception as e:
                    # Catch potential errors from the mock embedding function (though unlikely for random.uniform)
                    logger.error(f"Failed to generate embedding for item at index {i}: {e}", exc_info=True)
            
            logger.info(f"Processed batch of {len(data)} items. Generated {valid_items_processed} embeddings.")
            if valid_items_processed == 0 and any(isinstance(item, str) and item.strip() for item in data):
                logger.warning("Despite input containing valid strings, no embeddings were generated. This might indicate an internal issue.")

        else:
            logger.error(f"Unsupported data type for EmbeddingsGeneratorNode: {type(data).__name__}. Expected str or List[str].")
            raise TypeError(f"Unsupported data type: {type(data).__name__}. Expected str or List[str].")

        return embeddings