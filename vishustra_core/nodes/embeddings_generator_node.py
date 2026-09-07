import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path within the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of vector embeddings
    for given text inputs.

    This node expects input data to be a string or a list of strings.
    It returns a list of floats (representing an embedding vector) for a single
    string input, or a list of such vectors for a list of strings.
    """

    _DEFAULT_EMBEDDING_DIMENSION = 768 # Common dimension for many models like BERT, OpenAI Ada-002

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embeddings.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     Can optionally contain 'embedding_dimension' for custom output
                     vector size.

        Returns:
            A list of floats (for single string input) or a list of lists of floats
             (for list of strings input), representing the generated embeddings.

        Raises:
            ValueError: If the input data is not a string or a list of strings.
        """
        embedding_dimension = context.get('embedding_dimension', self._DEFAULT_EMBEDDING_DIMENSION)
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"Invalid 'embedding_dimension' '{embedding_dimension}' in context. "
                f"Using default dimension {self._DEFAULT_EMBEDDING_DIMENSION}."
            )
            embedding_dimension = self._DEFAULT_EMBEDDING_DIMENSION

        if isinstance(data, str):
            logger.debug(f"Generating embedding for a single string input.")
            # Simulate embedding: return a vector of zeros
            return [0.0] * embedding_dimension
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(f"Input data list contains non-string items: {data}")
                raise ValueError("All items in the input data list must be strings for embedding generation.")
            
            logger.debug(f"Generating embeddings for a list of {len(data)} strings.")
            # Simulate embeddings for each string
            return [[0.0] * embedding_dimension for _ in data]
        else:
            logger.error(f"Invalid input data type for EmbeddingsGeneratorNode: {type(data)}. Expected string or list of strings.")
            raise ValueError("Input data must be a string or a list of strings to generate embeddings.")
