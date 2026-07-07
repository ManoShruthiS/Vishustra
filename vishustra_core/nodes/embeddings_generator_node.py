import logging
from typing import Any, Dict, List, Union
# Assuming vishustra_core.nodes.base_node is available as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate embeddings for text data.

    This node accepts either a single string or a list of strings and returns
    a corresponding list of float vectors, representing the embeddings.
    The embedding generation is simulated for demonstration and testing purposes,
    providing deterministic, fixed-dimension vectors based on input characteristics.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _simulate_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of an embedding vector for a given text.
        This is a deterministic simulation, suitable for testing and development.

        Args:
            text: The input string for which to generate a simulated embedding.
            dimension: The desired dimension of the output embedding vector.

        Returns:
            A list of floats representing the simulated embedding vector.

        Raises:
            TypeError: If the input `text` is not a string.
        """
        if not isinstance(text, str):
            logger.error(f"Input for _simulate_embedding must be a string, got {type(text).__name__}.")
            raise TypeError("Input for embedding simulation must be a string.")
        
        # Generate a simple hash-like value from the text content.
        # This ensures determinism and some variability across different texts.
        text_seed = sum(ord(c) for c in text) % 1000 if text else 0
        
        # Construct a vector of the specified dimension.
        # The values are derived from the text_seed and an incremental factor,
        # ensuring a unique (for simulation purposes) vector for each unique text
        # while maintaining the requested dimension.
        return [
            (text_seed / 1000.0) + (i / float(dimension * 20.0))
            for i in range(dimension)
        ]

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates simulated embeddings for the input text data.

        The `data` input can be a single string or a list of strings.
        The `context` dictionary can specify the `embedding_dimension`.

        Args:
            data: The input text(s) to generate embeddings for. Can be a single
                  string or a list of strings.
            context: A dictionary containing operational context.
                     Expected keys:
                     - 'embedding_dimension' (int, optional): The desired dimension
                       of the output embeddings. Defaults to 768.

        Returns:
            A list of floats (for a single string input) or a list of lists of floats
            (for a list of strings input), representing the simulated embeddings.

        Raises:
            TypeError: If `data` is not a string or a list of strings, or if
                       elements within a list are not strings.
            ValueError: If `embedding_dimension` in context is not a positive integer.
        """
        logger.info("EmbeddingsGeneratorNode initiated processing.")

        if not isinstance(data, (str, list)):
            logger.error(f"Invalid data type received. Expected str or list[str], got {type(data).__name__}.")
            raise TypeError(f"Invalid data type. Expected str or list[str], got {type(data).__name__}.")

        embedding_dimension = context.get("embedding_dimension", 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"Invalid 'embedding_dimension' in context. Expected a positive integer, got {embedding_dimension}.")
            raise ValueError(f"Invalid 'embedding_dimension' in context. Must be a positive integer, got {embedding_dimension}.")

        if isinstance(data, str):
            logger.debug(f"Generating embedding for a single text item with dimension {embedding_dimension}.")
            result = self._simulate_embedding(data, embedding_dimension)
            logger.info("Successfully generated embedding for single text item.")
            return result
        else:  # data is assumed to be list
            if not all(isinstance(item, str) for item in data):
                logger.error("Input list contains non-string elements. All elements must be strings.")
                raise TypeError("All elements in the input list must be strings.")
            
            if not data:
                logger.warning("Input list is empty. Returning an empty list of embeddings.")
                return []

            logger.debug(f"Generating embeddings for {len(data)} text items with dimension {embedding_dimension}.")
            embeddings = [self._simulate_embedding(item, embedding_dimension) for item in data]
            logger.info(f"Successfully generated embeddings for {len(data)} text items.")
            return embeddings

