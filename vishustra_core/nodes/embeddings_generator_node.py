import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that generates embeddings for input text.

    This node simulates the process of converting textual data (strings)
    into numerical vector representations (embeddings). It can handle
    either a single string or a list of strings.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension (int): The desired dimension of the
                                       simulated embedding vectors.
        Raises:
            ValueError: If `embedding_dimension` is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError("`embedding_dimension` must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.debug(f"EmbeddingsGeneratorNode initialized with dimension: {self._embedding_dimension}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        In a production system, this method would typically interface with
        an external embedding model service (e.g., OpenAI, Hugging Face).
        For this simulation, it generates a vector of random floats.
        The randomness is seeded by the input text's hash for consistent
        (though still random-looking) output for identical inputs within
        a single process run.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        # Ensure determinism for simulation purposes for identical inputs.
        # Note: hash() can vary between Python runs/processes, but for a single run
        # it provides a stable seed for the same string.
        # Max value for random seed to avoid OS-specific issues with large integers
        MAX_SEED = 2**32 - 1
        random.seed(hash(text) % MAX_SEED)

        # Generate a list of random floats within a typical embedding range.
        embedding = [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        The node expects input `data` to be either a single string or a
        list of strings. It returns a single embedding vector (List[float])
        for a single string input, or a list of embedding vectors
        (List[List[float]]) for a list of strings.

        Args:
            data (Any): The input data, expected to be `str` or `List[str]`.
            context (Dict[str, Any]): A dictionary containing runtime context
                                       and configuration. Not directly used by
                                       this simulated node's logic but available.

        Returns:
            Union[List[float], List[List[float]]]: The generated embedding(s).

        Raises:
            TypeError: If the input `data` is not a `str` or `List[str]`,
                       or if a list contains non-string elements.
            ValueError: If an unexpected error occurs during embedding generation
                        for a single item or any item in a batch.
        """
        logger.info(f"EmbeddingsGeneratorNode received data for processing. Context keys: {list(context.keys()) if context else []}")

        if isinstance(data, str):
            try:
                embedding = self._generate_single_embedding(data)
                logger.debug(f"Generated single embedding (dimension: {len(embedding)}) for input string.")
                return embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding for single string data: '{data[:100]}...'. Error: {e}", exc_info=True)
                raise ValueError(f"Failed to generate embedding for single string: {e}") from e

        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error("Input list contains non-string elements. All elements must be strings for embedding generation.")
                raise TypeError("EmbeddingsGeneratorNode expects a list of strings, but found non-string elements.")

            embeddings: List[List[float]] = []
            for i, text in enumerate(data):
                try:
                    embeddings.append(self._generate_single_embedding(text))
                except Exception as e:
                    logger.error(f"Failed to generate embedding for item {i} in batch ('{text[:50]}...'). Error: {e}", exc_info=True)
                    # For batch processing, the strategy for errors can vary (e.g., skip, return partial, raise).
                    # Here, we raise to indicate a full failure for any element's processing.
                    raise ValueError(f"Failed to generate embedding for item {i} in batch: {e}") from e
            
            logger.debug(f"Generated {len(embeddings)} embeddings for a list of {len(data)} strings.")
            return embeddings

        else:
            logger.error(f"Unsupported data type for EmbeddingsGeneratorNode: {type(data)}. Expected str or List[str].")
            raise TypeError("EmbeddingsGeneratorNode expects data of type `str` or `List[str]`.")