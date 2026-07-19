import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

# Initialize a logger for this module.
logger = logging.getLogger(__name__)

class EmbeddingsGenerator(BaseNode):
    """
    A Vishustra processing node responsible for generating numerical embeddings
    from input text data. This node simulates the process of transforming
    human-readable text into a dense vector representation, suitable for
    machine learning tasks like similarity search, clustering, or input to other models.

    It can process single strings or lists of strings for batch embedding.
    """

    # A common embedding dimension, chosen for illustrative purposes.
    # In a real implementation, this might be configurable via context or an __init__ parameter.
    _EMBEDDING_DIMENSION: int = 768

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text string.

        In a production system, this method would interface with an actual
        embedding model (e.g., from Hugging Face Transformers, OpenAI API, etc.).
        For this simulation, it produces a list of random floats.

        Args:
            text: The input string to be embedded.

        Returns:
            A list of floats representing the embedding vector.
        """
        logger.debug(f"[{self.node_name}] Simulating embedding for text snippet: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        # Generate a list of random floats within a typical embedding range (-1.0 to 1.0)
        return [random.uniform(-1.0, 1.0) for _ in range(self._EMBEDDING_DIMENSION)]

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        This method expects either a single string or a list of strings as input.
        It uses a simulated embedding generation mechanism to return corresponding
        embedding vectors.

        Args:
            data: The input text data to embed.
                  Can be a single `str` or a `List[str]` for batch processing.
            context: A dictionary containing contextual information for the node.
                     This could include model names, batching strategies,
                     API keys, or other configuration specific to the embedding process.
                     (Currently not utilized by the simulated embedding logic but available).

        Returns:
            If `data` is a `str`, returns a `List[float]` representing the single embedding.
            If `data` is a `List[str]`, returns a `List[List[float]]`, where each inner
            list is an embedding for the corresponding input string.

        Raises:
            TypeError: If the input `data` is not a `str` or `List[str]`,
                       or if a list contains non-string elements.
            Exception: Catches and re-raises any unexpected errors during the embedding process.
        """
        logger.info(f"[{self.node_name}] Initiating embedding generation process.")

        if not isinstance(data, (str, list)):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if isinstance(data, list) and not all(isinstance(item, str) for item in data):
            error_msg = (
                f"[{self.node_name}] Invalid list content. "
                f"All elements within the input list must be strings. "
                f"Detected non-string elements in the list."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            if isinstance(data, str):
                logger.debug(f"[{self.node_name}] Processing a single text string.")
                result = self._generate_single_embedding(data)
            else: # Must be List[str] due to prior type checks
                logger.debug(f"[{self.node_name}] Processing a list of text strings (batch mode).")
                result = [self._generate_single_embedding(text) for text in data]

            logger.info(f"[{self.node_name}] Successfully generated embeddings for {len(data) if isinstance(data, list) else 'single'} item(s).")
            return result
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during embedding generation.")
            # Re-raise the exception to propagate it up the call stack for further handling
            raise e