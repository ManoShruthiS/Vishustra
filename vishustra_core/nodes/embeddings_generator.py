import logging
import hashlib
import random
from typing import Any, Dict, List, Union

# Assuming the base_node module is located within vishustra_core.nodes
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A processing node that generates simulated embeddings for text data.

    This node takes text (either a single string or a list of strings)
    and converts it into a numerical vector representation (embedding).
    For demonstration purposes, a reproducible mock embedding is generated.
    """

    _DEFAULT_EMBEDDING_DIMENSIONS = 128

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Generates simulated embeddings for the input data.

        The `data` can be a single string or a list of strings.
        The `context` can optionally specify `embedding_dimensions`.

        Args:
            data (Union[str, List[str]]): The text data to embed.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       Expected keys:
                                       - 'embedding_dimensions' (int, optional): Desired
                                         dimension of the embedding vectors. Defaults to 128.

        Returns:
            Union[List[float], List[List[float]]]: The generated embedding(s).
                                                  A single list of floats if input was a string,
                                                  or a list of lists of floats if input was a list.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if a list contains non-string items.
            RuntimeError: If an unexpected error occurs during embedding generation.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                "Invalid input data type for EmbeddingsGeneratorNode. Expected str or list[str], got %s.",
                type(data),
            )
            raise ValueError("Input data must be a string or a list of strings.")

        embedding_dimensions = context.get(
            "embedding_dimensions", self._DEFAULT_EMBEDDING_DIMENSIONS
        )

        if not isinstance(embedding_dimensions, int) or embedding_dimensions <= 0:
            logger.warning(
                "Invalid 'embedding_dimensions' in context: %s. "
                "Using default dimensions: %d.",
                embedding_dimensions,
                self._DEFAULT_EMBEDDING_DIMENSIONS,
            )
            embedding_dimensions = self._DEFAULT_EMBEDDING_DIMENSIONS

        try:
            if isinstance(data, str):
                if not data:
                    logger.warning("Received empty string for embedding generation.")
                return self._generate_mock_embedding(data, embedding_dimensions)
            elif isinstance(data, list):
                embeddings = []
                for item in data:
                    if not isinstance(item, str):
                        logger.error(
                            "List contains non-string item: %s. "
                            "All items in the input list must be strings.",
                            type(item),
                        )
                        raise ValueError("All items in the input list must be strings.")
                    if not item:
                        logger.warning("Received empty string in list for embedding generation.")
                    embeddings.append(self._generate_mock_embedding(item, embedding_dimensions))
                return embeddings
        except Exception as e:
            logger.exception("An unexpected error occurred during embedding generation.")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def _generate_mock_embedding(self, text: str, dimensions: int) -> List[float]:
        """
        Generates a reproducible mock embedding for a given text.
        This simulates a fixed-size vector representation.
        The generation is deterministic based on the input text.
        """
        # Use SHA256 hash to create a robust seed for reproducibility across runs and systems
        hasher = hashlib.sha256()
        hasher.update(text.encode('utf-8'))
        # Ensure seed fits within typical integer limits for random.Random
        seed = int(hasher.hexdigest(), 16) % (2**32 - 1)

        rng = random.Random(seed)
        # Generate a list of floats, typically between -1.0 and 1.0, rounded for cleanliness
        return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dimensions)]