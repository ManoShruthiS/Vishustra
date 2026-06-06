import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of embeddings
    for textual data.

    This node expects a string as input and produces a fixed-dimension list
    of floats, simulating a dense vector embedding. For demonstration purposes,
    the embedding values are derived deterministically from the input string's hash.
    """

    # Default embedding dimension if not specified in context
    DEFAULT_EMBEDDING_DIM = 768

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> List[float]:
        """
        Generates a simulated embedding for the input data.

        Expects 'data' to be a string. If 'data' is not a string, it logs an
        error and raises a TypeError.

        Context can optionally provide 'embedding_dim' (int) to specify the output
        vector dimension. If 'embedding_dim' is invalid, a ValueError is raised.

        Args:
            data: The input text (str) for which to generate embeddings.
            context: A dictionary containing runtime context, potentially including
                     'embedding_dim' (int).

        Returns:
            A list of floats representing the simulated embedding vector.

            Example: `[0.123, 0.456, ..., 0.789]`

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'embedding_dim' in context is not a positive integer.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            )
            raise TypeError(
                f"EmbeddingsGeneratorNode requires string input, but received {type(data).__name__}."
            )

        embedding_dim = context.get("embedding_dim", self.DEFAULT_EMBEDDING_DIM)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.error(
                f"[{self.node_name}] Invalid 'embedding_dim' in context. Expected a positive integer, got '{embedding_dim}'."
            )
            raise ValueError(
                f"Invalid 'embedding_dim' in context: {embedding_dim}. Must be a positive integer."
            )

        try:
            # Simulate embedding generation based on the input string's properties.
            # This is a deterministic but simple simulation.
            # In a real scenario, this would involve calling an external embedding model API or library.
            data_hash = hash(data)
            
            simulated_embedding: List[float] = []
            for i in range(embedding_dim):
                # A simple deterministic function to generate float values
                # ensures that the same input string produces the same "embedding"
                # Values are normalized between 0.0 and 1.0 for plausibility.
                value = (abs(data_hash + i * 7) % 997) / 996.0 # Using 997 as a prime for distribution
                simulated_embedding.append(value)

            logger.info(
                f"[{self.node_name}] Successfully generated simulated embedding "
                f"of dimension {embedding_dim} for input (first 50 chars): '{data[:50]}{'...' if len(data) > 50 else ''}'"
            )
            return simulated_embedding

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during embedding generation for data (first 50 chars): '{data[:50]}{'...' if len(data) > 50 else ''}'"
            )
            # Re-raise the exception after logging for upstream handling
            raise
