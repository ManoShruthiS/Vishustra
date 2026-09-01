import logging
import random
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate vector embeddings for text data.

    This node takes a string as input and returns a list of floats representing
    its high-dimensional vector embedding. For demonstration and modularity
    within Vishustra, the embedding generation is simulated. In a production
    environment, this would integrate with an actual embedding model API or library.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> List[float]:
        """
        Generates a simulated embedding vector for the input text data.

        The `data` input is expected to be a string (the text content to be embedded).
        The `context` dictionary can optionally specify parameters for embedding
        generation, such as the desired vector `embedding_dimension`.

        Args:
            data: The input text (str) for which to generate an embedding.
            context: A dictionary providing contextual information for the process.
                     Expected keys:
                     - 'embedding_dimension' (optional, int): The desired
                       dimensionality of the output embedding vector. Defaults to 768.

        Returns:
            A list of floats representing the simulated embedding vector.

        Raises:
            ValueError: If the input `data` is not a string, as embeddings
                        are typically generated from textual content.
            RuntimeError: If an unexpected issue occurs during the simulated
                          embedding generation process.
        """
        if not isinstance(data, str):
            error_msg = (
                f"EmbeddingsGeneratorNode requires 'data' to be a string for embedding. "
                f"Received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Retrieve embedding_dimension from context, defaulting if not provided
        embedding_dimension = context.get('embedding_dimension', 768)

        # Validate embedding_dimension
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"Invalid 'embedding_dimension' specified in context: {embedding_dimension}. "
                f"It must be a positive integer. Falling back to default dimension 768."
            )
            embedding_dimension = 768

        try:
            # Simulate embedding generation: create a list of random floats.
            # In a real-world implementation, this section would invoke an
            # external embedding service or a local model inference.
            embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]

            logger.debug(
                f"Successfully generated simulated embedding of dimension {embedding_dimension} "
                f"for input string (first 50 chars): '{data[:50]}{'...' if len(data) > 50 else ''}'"
            )
            return embedding

        except Exception as e:
            # Catch any unexpected errors during the simulation or potential
            # future integration with actual embedding models.
            error_msg = f"An unexpected error occurred during embedding generation for input data. Details: {e}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e