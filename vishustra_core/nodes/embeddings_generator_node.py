import logging
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure.
# This relative import ensures compatibility within the Vishustra framework.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating text embeddings.

    This node simulates the process of transforming input text (either a single string
    or a list of strings) into a numerical vector representation (embedding).
    In a production environment, this would interface with a dedicated embedding model
    or service. For this implementation, it provides a deterministic, simulated vector
    for demonstration and testing purposes.
    """

    # Default dimension for the simulated embeddings
    _DEFAULT_EMBEDDING_DIMENSION: int = 128

    def __init__(self, embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension (int): The desired dimension for the generated embeddings.
                                       Must be a positive integer. Defaults to 128.

        Raises:
            ValueError: If `embedding_dimension` is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"Invalid embedding_dimension: {embedding_dimension}. Must be a positive integer.")
            raise ValueError("embedding_dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.debug(f"EmbeddingsGeneratorNode initialized with embedding dimension: {self._embedding_dimension}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated text embeddings.

        This method accepts either a single string or a list of strings.
        For each string, it generates a simulated embedding vector of a predefined dimension.
        The `context` dictionary is provided for potential future use (e.g., passing model
        parameters or client-specific configurations), but is not actively used in
        this simulated embedding generation logic.

        Args:
            data (Union[str, List[str]]): The text input(s) for which to generate embeddings.
                                          Can be a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing.

        Returns:
            Union[List[float], List[List[float]]]:
                If `data` is a single string, returns a `List[float]` representing
                its embedding.
                If `data` is a list of strings, returns a `List[List[float]]`,
                where each inner list is an embedding for the corresponding input string.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if any item within an input list is not a string.
        """
        logger.info(f"Node '{self.node_name}' received data for embedding generation.")
        # Logging context for debugging and traceability, even if not directly used
        logger.debug(f"Processing context: {context}")

        if isinstance(data, str):
            if not data.strip():
                logger.warning("Received an empty or whitespace-only string for embedding generation. Returning an embedding of zeros.")
                return [0.0] * self._embedding_dimension
            # Simulate a deterministic embedding for a single string.
            # In a real system, this would involve calling a sophisticated model.
            simulated_embedding = [
                (i % 10 + 1) * 0.01 + len(data) * 0.00001
                for i in range(self._embedding_dimension)
            ]
            logger.debug(f"Generated simulated embedding for single string (first 5 elements): {simulated_embedding[:5]}...")
            return simulated_embedding
        elif isinstance(data, list):
            if not data:
                logger.warning("Received an empty list for embedding generation. Returning an empty list of embeddings.")
                return []

            embeddings: List[List[float]] = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(f"Input list contains non-string item at index {i} (type: {type(item)}). Raising TypeError.")
                    raise TypeError(f"All items in the input list must be strings, but found type {type(item)} at index {i}.")

                if not item.strip():
                    logger.warning(f"Received an empty or whitespace-only string in the list at index {i}. Returning an embedding of zeros for it.")
                    embeddings.append([0.0] * self._embedding_dimension)
                else:
                    # Simulate a deterministic embedding for each string in the list.
                    simulated_embedding = [
                        (j % 10 + 1) * 0.01 + len(item) * 0.00001
                        for j in range(self._embedding_dimension)
                    ]
                    embeddings.append(simulated_embedding)
                    logger.debug(f"Generated simulated embedding for string at index {i} (first 5 elements): {simulated_embedding[:5]}...")
            
            logger.info(f"Successfully generated {len(embeddings)} simulated embeddings for a list of strings.")
            return embeddings
        else:
            logger.error(f"Invalid data type received for embedding generation: {type(data)}. Expected str or List[str]. Raising TypeError.")
            raise TypeError(f"Input data must be a string or a list of strings, but received {type(data)}.")