import logging
import hashlib
from typing import Any, Dict, List, Union

# Assuming vishustra_core is installed and available in the project's Python path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating text embeddings.

    This node takes either a single string or a list of strings as input
    and returns a simulated embedding vector (or a list of vectors) for each.
    It's designed to be a placeholder for integration with actual embedding models.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension (int): The simulated dimension of the embedding vectors.
                                       This parameter is used to define the size of
                                       the mock embedding vectors.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Invalid embedding_dimension provided: {embedding_dimension}. "
                "Must be a positive integer. Defaulting to 768."
            )
            self._embedding_dimension = 768
        else:
            self._embedding_dimension = embedding_dimension
        logger.debug(f"{self.node_name} initialized with simulated dimension: {self._embedding_dimension}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of an embedding vector for a single text string.

        This method provides a deterministic, mock embedding by hashing the input text
        and converting it into a fixed-size list of floats. This is purely for
        demonstration and testing within the Vishustra framework without external
        model dependencies.

        Args:
            text (str): The input text string to generate an embedding for.

        Returns:
            List[float]: A list of floats representing the simulated embedding vector.
        """
        # Use SHA256 hash to create a deterministic "seed" for the embedding.
        # This ensures the same text always yields the same simulated embedding.
        text_hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)

        embedding = []
        for i in range(self._embedding_dimension):
            # A simple pseudo-random generation based on hash and index.
            # Values are scaled to be between 0.0 and 1.0.
            val = ((text_hash_val + i * 12345) % 1000000) / 1000000.0
            embedding.append(val)
            
        # Introduce a slight variation based on text length to make it slightly more dynamic
        # for different text lengths, purely for simulation purposes.
        if self._embedding_dimension > 0:
            embedding[0] = (embedding[0] + (len(text) % 100) / 100.0) / 2.0

        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embeddings.

        This method expects `data` to be either a single string or a list of strings.
        It generates a simulated embedding for each text item.

        Args:
            data (Any): The input data. Expected types are `str` or `List[str]`.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. Not directly used
                                       by this node's logic but available for future
                                       enhancements (e.g., model configuration,
                                       API keys, etc.).

        Returns:
            Union[List[float], List[List[float]]]:
                - A list of floats representing a single embedding if `data` was a `str`.
                - A list of lists of floats (each inner list being an embedding)
                  if `data` was a `List[str]`.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if a list contains non-string elements.
            RuntimeError: If an unexpected error occurs during the embedding generation.
        """
        logger.info(f"[{self.node_name}] Starting process for data of type: {type(data)}")

        if not isinstance(data, (str, list)):
            error_msg = (
                f"Invalid input data type for {self.node_name}. Expected 'str' or 'List[str]', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if isinstance(data, str):
            try:
                embedding = self._generate_single_embedding(data)
                logger.debug(
                    f"[{self.node_name}] Generated embedding for single string "
                    f"(first 5 values): {embedding[:5]}..."
                )
                return embedding
            except Exception as e:
                error_msg = f"[{self.node_name}] Failed to generate embedding for single string: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                error_msg = (
                    f"Invalid list item type for {self.node_name}. Expected 'List[str]', "
                    "but found non-string elements within the list."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            embeddings = []
            for i, text in enumerate(data):
                try:
                    embedding = self._generate_single_embedding(text)
                    embeddings.append(embedding)
                    logger.debug(
                        f"[{self.node_name}] Generated embedding for list item {i} "
                        f"(first 5 values): {embedding[:5]}..."
                    )
                except Exception as e:
                    # Decide on error handling strategy:
                    # 1. Raise immediately (as implemented here).
                    # 2. Log warning and append a placeholder (e.g., None, empty list) for the failed item.
                    # For critical processing, raising immediately is often preferred.
                    error_msg = (
                        f"[{self.node_name}] Failed to generate embedding for list item {i} "
                        f"('{(text[:50] + '...') if len(text) > 50 else text}'): {e}"
                    )
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
            
            logger.info(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings for list input.")
            return embeddings