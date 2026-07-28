import logging
import random
from typing import Any, Dict, List, Union

# Assuming the core BaseNode is located here
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node responsible for generating embedding vectors from text data.

    This node simulates the process of converting input text (or a list of texts)
    into numerical vector representations (embeddings). The dimension of the
    generated embeddings can be configured via the 'embedding_dimension' key
    in the operational context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates simulated embeddings for the given text data.

        This method expects either a single string or a list of strings as input.
        It then generates a corresponding simulated embedding vector or a list
        of embedding vectors.

        Args:
            data: The input text as a `str` or a `List[str]` for which to generate embeddings.
            context: A dictionary containing operational parameters. Expected keys include:
                     - 'embedding_dimension' (int, optional): The desired dimension for
                       the generated embedding vectors. Defaults to 768 if not provided.

        Returns:
            A `List[float]` representing a single embedding vector if `data` was a string,
            or a `List[List[float]]` if `data` was a list of strings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If 'embedding_dimension' in `context` is not a positive integer.
            RuntimeError: For unexpected errors during the embedding generation process.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"Invalid input data type for EmbeddingsGeneratorNode. Expected str or list[str], "
                f"received {type(data).__name__}. Input data: {data}"
            )
            raise TypeError(
                f"EmbeddingsGeneratorNode requires 'data' to be a string or a list of strings, "
                f"but received type: {type(data).__name__}"
            )

        embedding_dimension = context.get('embedding_dimension', 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Invalid 'embedding_dimension' in context: {embedding_dimension}. "
                f"Must be a positive integer."
            )
            raise ValueError(
                f"'embedding_dimension' in context must be a positive integer, "
                f"but received: {embedding_dimension}"
            )

        try:
            if isinstance(data, str):
                logger.info(
                    f"Generating embedding for single text (length: {len(data)}) "
                    f"with dimension {embedding_dimension}."
                )
                return self._generate_single_embedding(data, embedding_dimension)
            else:  # data is a list
                if not all(isinstance(item, str) for item in data):
                    logger.error(
                        f"Input list 'data' contains non-string elements. "
                        f"All elements must be strings."
                    )
                    raise TypeError(
                        f"When 'data' is a list, all its elements must be strings. "
                        f"Found non-string elements."
                    )
                logger.info(
                    f"Generating embeddings for {len(data)} texts "
                    f"with dimension {embedding_dimension}."
                )
                return [self._generate_single_embedding(text, embedding_dimension) for text in data]
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during embedding generation in EmbeddingsGeneratorNode."
            )
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        In a production environment, this method would interface with an actual
        embedding model (e.g., via an API call or a local model inference).
        For this simulation, it produces a list of random floats.

        Args:
            text: The text string to embed.
            dimension: The desired dimensionality of the embedding vector.

        Returns:
            A list of floats representing the simulated embedding vector.
        """
        # Simulate an embedding vector with random float values.
        # The specific values are arbitrary for the purpose of this simulation.
        return [random.uniform(-1.0, 1.0) for _ in range(dimension)]
