import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is correctly located here based on project structure
from vishustra_core.nodes.base_node import BaseNode

# Configure a module-level logger
logger = logging.getLogger(__name__)


class TextEmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating text embeddings.

    This node simulates the interaction with an external or internal embedding model
    to convert input text (strings) into numerical vector representations.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initializes the TextEmbeddingsGeneratorNode.

        Args:
            embedding_dimension: The desired dimension of the output embedding vectors.
                                 Defaults to 768 for common embedding models.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.info(
            f"Initialized TextEmbeddingsGeneratorNode with embedding dimension: {self._embedding_dimension}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextEmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of an embedding vector for a single string.
        In a production system, this would involve calling a specific embedding model
        (e.g., from OpenAI, Hugging Face, or a local service).

        For simulation, it creates a deterministic, random-like vector
        based on a seed derived from the input text, ensuring consistent
        output for the same input while demonstrating vector structure.
        """
        # A simple, deterministic way to seed randomness based on text content.
        # This is for simulation purposes only. Real models are more complex.
        text_hash = sum(ord(c) for c in text) % 10000
        random.seed(text_hash)

        # Generate a list of floats as a simulated embedding vector
        # Values are typically floats, often normalized, but for simulation,
        # a simple range is sufficient.
        embedding = [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]
        return embedding

    def process(
        self, data: Union[str, List[str]], context: Dict[str, Any]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for the input text data.

        The node accepts either a single string or a list of strings as input.
        It returns a corresponding embedding vector or a list of embedding vectors.

        Args:
            data: The input text(s) to be embedded. Can be a single string or a list of strings.
            context: A dictionary containing contextual information relevant to the processing,
                     though not directly used in this simulated node beyond logging.

        Returns:
            A list of floats representing the embedding vector if `data` was a single string,
            or a list of lists of floats if `data` was a list of strings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If an unexpected error occurs during the embedding generation process.
        """
        logger.debug(
            f"[{self.node_name}] Starting process for input data type: {type(data).__name__}"
        )

        if not isinstance(data, (str, list)):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected `str` or `List[str]`, got `{type(data).__name__}`."
            )
            raise TypeError(
                f"Input data must be a string or a list of strings, but got `{type(data).__name__}`."
            )

        try:
            if isinstance(data, str):
                logger.debug(f"[{self.node_name}] Generating embedding for a single string.")
                embedding = self._generate_single_embedding(data)
                logger.debug(
                    f"[{self.node_name}] Generated single embedding of dimension {len(embedding)}."
                )
                return embedding
            else:  # data is a list
                if not all(isinstance(item, str) for item in data):
                    logger.error(
                        f"[{self.node_name}] List input contains non-string elements. "
                        f"Found element of type: {type([item for item in data if not isinstance(item, str)][0]).__name__}"
                    )
                    raise TypeError("All elements in the input list must be strings.")

                logger.debug(f"[{self.node_name}] Generating embeddings for {len(data)} strings.")
                embeddings = [self._generate_single_embedding(text) for text in data]
                logger.debug(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings.")
                return embeddings
        except Exception as e:
            # Catching generic exceptions for robust error reporting
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during embedding generation: {e}"
            )
            raise ValueError(f"Failed to generate embeddings: {e}") from e

