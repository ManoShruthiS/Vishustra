import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of embeddings for given text data.

    This node processes input text (a single string or a list of strings)
    and returns corresponding embedding vectors. For demonstration purposes,
    embeddings are simulated using randomly generated float values.
    """

    DEFAULT_EMBEDDING_DIM = 768  # A common default dimension for embedding vectors

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "Embeddings Generator"

    def _generate_single_embedding(self, text: str, embedding_dim: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a production environment, this would involve calling an external
        embedding service or a local ML model.
        """
        # Log a debug message to show the operation, without exposing sensitive data
        logger.debug(f"Simulating embedding for text (first 50 chars): '{text[:50]}...' "
                     f"with target dimension {embedding_dim}.")
        # Generate a list of random floats between -1.0 and 1.0 to simulate an embedding vector
        return [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for the input data.

        The input `data` can be a single string or a list of strings.
        The `context` dictionary can optionally specify the 'embedding_dimension'
        for the generated vectors. If not provided, `DEFAULT_EMBEDDING_DIM` is used.

        Args:
            data: The input text data (str or list[str]) for which to generate embeddings.
            context: A dictionary containing execution context and configuration.
                     Recognizes 'embedding_dimension' (int) to set the output vector size.

        Returns:
            A list of floats if `data` was a single string, or a list of lists of floats
            if `data` was a list of strings. Each inner list represents an embedding vector.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if any element within the list is not a string.
            ValueError: If 'embedding_dimension' in the `context` is not a positive integer.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"Input data for {self.node_name} is of invalid type. "
                f"Expected str or list[str], but received {type(data).__name__}."
            )
            raise TypeError(
                f"Invalid input data type for {self.node_name}. Expected str or list[str], "
                f"but got {type(data).__name__}."
            )

        embedding_dim = context.get('embedding_dimension', self.DEFAULT_EMBEDDING_DIM)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.error(
                f"Invalid 'embedding_dimension' in context for {self.node_name}. "
                f"Expected a positive integer, but received '{embedding_dim}' ({type(embedding_dim).__name__})."
            )
            raise ValueError(
                f"'embedding_dimension' in context must be a positive integer, "
                f"but received {embedding_dim}."
            )
        
        logger.info(f"Initiating embedding generation with {self.node_name}. "
                    f"Target dimension: {embedding_dim}.")

        if isinstance(data, str):
            embedding = self._generate_single_embedding(data, embedding_dim)
            logger.debug(f"Successfully generated a single embedding of dimension {len(embedding)}.")
            return embedding
        else:  # data is a list
            embeddings: List[List[float]] = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(
                        f"Invalid item type in list input for {self.node_name} at index {i}. "
                        f"Expected str, but received {type(item).__name__}."
                    )
                    raise TypeError(
                        f"All items in the input list for {self.node_name} must be strings, "
                        f"but item at index {i} is of type {type(item).__name__}."
                    )
                embeddings.append(self._generate_single_embedding(item, embedding_dim))
            logger.info(f"Successfully generated {len(embeddings)} embeddings, each of dimension {embedding_dim}.")
            return embeddings