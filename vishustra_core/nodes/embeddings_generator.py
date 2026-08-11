import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node responsible for generating embeddings for textual data.

    This node simulates the process of converting input text (either a single
    string or a list of strings) into high-dimensional numerical vectors
    (embeddings). The generated embeddings can then be used by downstream
    nodes for tasks such as similarity search, classification, or clustering.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGeneratorNode"

    def _generate_single_embedding(self, text: str, embedding_dim: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a production environment, this method would interface with a real
        embedding model (e.g., from Hugging Face, OpenAI, etc.).
        """
        logger.debug(
            f"{self.node_name}: Simulating embedding for text (first 50 chars): '{text[:50]}...'"
        )
        # Simulate a fixed-size vector of random floats,
        # typically between -1.0 and 1.0, representing an embedding.
        return [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]

    def process(
        self, data: Union[str, List[str]], context: Dict[str, Any]
    ) -> Union[List[float], List[List[float]]]:
        """
        Processes input text (or a list of texts) to generate embeddings.

        The `context` dictionary can optionally specify:
        - `embedding_dim` (int): The desired dimension of the embedding vectors.
                                 Defaults to 768 if not provided, a common dimension
                                 for many pre-trained models.

        Args:
            data: The input text as a single string or a list of strings to be
                  converted into embeddings. Empty strings or lists will be
                  handled gracefully.
            context: A dictionary containing operational context, which may
                     include configuration parameters like `embedding_dim`.

        Returns:
            A `List[float]` if the input `data` was a single string, or a
            `List[List[float]]` if the input `data` was a list of strings.
            Each inner list represents an embedding vector.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if a list contains non-string elements.
            RuntimeError: If an unexpected error occurs during the simulated
                          embedding generation process.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"{self.node_name}: Invalid input data type. Expected str or list[str], "
                f"got {type(data).__name__}."
            )
            raise ValueError(
                f"Input data for {self.node_name} must be a string or a list of strings, "
                f"but received {type(data).__name__}."
            )

        embedding_dim = context.get("embedding_dim", 768)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.warning(
                f"{self.node_name}: Invalid 'embedding_dim' in context: {embedding_dim}. "
                f"Using default dimension of 768."
            )
            embedding_dim = 768

        try:
            if isinstance(data, str):
                if not data.strip():
                    logger.warning(
                        f"{self.node_name}: Received empty or whitespace-only string. "
                        f"Returning a zero vector of dimension {embedding_dim}."
                    )
                    return [0.0] * embedding_dim
                return self._generate_single_embedding(data, embedding_dim)
            elif isinstance(data, list):
                embeddings: List[List[float]] = []
                for idx, item in enumerate(data):
                    if not isinstance(item, str):
                        logger.error(
                            f"{self.node_name}: List input contains non-string element at index {idx}. "
                            f"Expected str, got {type(item).__name__}."
                        )
                        raise ValueError(
                            f"List input for {self.node_name} must contain only strings. "
                            f"Found {type(item).__name__} at index {idx}."
                        )
                    if not item.strip():
                        logger.warning(
                            f"{self.node_name}: Received empty or whitespace-only string in list "
                            f"at index {idx}. Appending a zero vector of dimension {embedding_dim}."
                        )
                        embeddings.append([0.0] * embedding_dim)
                    else:
                        embeddings.append(self._generate_single_embedding(item, embedding_dim))
                return embeddings
        except Exception as e:
            logger.exception(
                f"{self.node_name}: An unexpected error occurred during embedding generation."
            )
            raise RuntimeError(f"Failed to generate embeddings in {self.node_name}: {e}") from e
