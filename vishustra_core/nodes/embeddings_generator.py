import logging
import random
from typing import Any, Dict, List, Union

# Assuming the specified import path for BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGenerator(BaseNode):
    """
    A Vishustra node that simulates the generation of embedding vectors for text.

    This node takes text input (either a single string or a list of strings)
    and produces simulated embedding vectors (lists of floats). In a production
    environment, this would integrate with a real embedding model service or library
    to convert textual data into a numerical representation suitable for machine
    learning tasks.
    """

    _DEFAULT_EMBEDDING_DIMENSION = 768
    _EMBEDDING_VALUE_RANGE = (-1.0, 1.0)  # Simulating values typically between -1 and 1

    def __init__(self, embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION):
        """
        Initializes the EmbeddingsGenerator node.

        Args:
            embedding_dimension: The desired dimension for the simulated embedding vectors.
                                 Must be a positive integer. Defaults to 768.

        Raises:
            ValueError: If the provided `embedding_dimension` is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"Invalid embedding_dimension '{embedding_dimension}'. Must be a positive integer.")
            raise ValueError("Embedding dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.debug(f"EmbeddingsGenerator initialized with dimension: {self._embedding_dimension}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self) -> List[float]:
        """
        Simulates generating a single embedding vector of the configured dimension.
        Values are randomized within a predefined range to mimic real embeddings.
        """
        min_val, max_val = self._EMBEDDING_VALUE_RANGE
        return [random.uniform(min_val, max_val) for _ in range(self._embedding_dimension)]

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embeddings.

        The node expects the `data` parameter to be either a single string
        or a list of strings. Each string will be processed into a
        simulated embedding vector. The `context` dictionary is currently
        not utilized for core embedding generation but provides a mechanism
        for future extensions, such as dynamic model configuration or batching
        parameters.

        Args:
            data: The input text(s) to be embedded.
                  Expected types: `str` for a single text, or `List[str]` for multiple texts.
            context: A dictionary containing runtime context information, which can include
                     configuration specific to the embedding process.

        Returns:
            A `List[float]` if `data` was a single string, representing one embedding vector.
            A `List[List[float]]` if `data` was a list of strings, representing multiple
            embedding vectors, one for each input string.

        Raises:
            ValueError: If the input `data` is not of an expected type (`str` or `List[str]`).
            RuntimeError: For any unexpected operational failures during embedding generation.
        """
        logger.info(f"Node '{self.node_name}' received data of type '{type(data).__name__}' for processing.")

        try:
            if isinstance(data, str):
                # Process a single text string
                embedding = self._generate_single_embedding()
                logger.debug(f"Successfully generated a single embedding (dimension: {self._embedding_dimension}).")
                return embedding
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                # Process a list of text strings
                embeddings = [self._generate_single_embedding() for _ in data]
                logger.debug(f"Successfully generated {len(embeddings)} embeddings (dimension: {self._embedding_dimension}).")
                return embeddings
            else:
                error_msg = (
                    f"Invalid input data type for '{self.node_name}'. "
                    f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        except ValueError as ve:
            # Re-raise ValueErrors as they indicate issues with the provided input data.
            raise ve
        except Exception as e:
            # Catch any other unexpected exceptions and wrap them in a RuntimeError
            # for consistent error propagation in the framework.
            logger.critical(
                f"An unexpected error occurred during embedding generation in '{self.node_name}': {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to generate embeddings due to an unexpected internal error: {e}") from e