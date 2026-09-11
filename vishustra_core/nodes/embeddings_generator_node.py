import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of text embeddings.

    This node takes text (or a list of texts) as input and returns a
    list of embedding vectors. For simulation purposes, the embeddings
    are generated deterministically based on input text content (via a seed)
    and are of a configurable dimension.

    The actual embedding model invocation logic would reside within
    the `_generate_single_embedding` method in a real-world scenario.
    """

    _DEFAULT_EMBEDDING_DIMENSION = 384  # Common embedding dimension in many models

    def __init__(self, embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension (int): The default dimension of the simulated embedding vectors.
                                       This can be overridden by `context` during `process` execution.

        Raises:
            ValueError: If `embedding_dimension` is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"[{self.node_name}] Invalid embedding_dimension provided during initialization: "
                f"{embedding_dimension}. Must be a positive integer."
            )
            raise ValueError("Embedding dimension must be a positive integer.")
        self._initialized_embedding_dimension = embedding_dimension
        logger.debug(
            f"[{self.node_name}] Initialized with default dimension: "
            f"{self._initialized_embedding_dimension}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGeneratorNode"

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a real system, this method would integrate with an actual embedding model API.

        For simulation, it generates a list of floats of the specified dimension.
        The values are pseudo-random but deterministic based on the text content
        to allow for consistent testing.

        Args:
            text (str): The input text to embed.
            dimension (int): The desired dimension for the output embedding vector.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        # Create a simple deterministic seed based on the text content.
        # This ensures the same text always produces the same simulated embedding.
        seed = sum(ord(c) for c in text) % 1_000_000
        random.seed(seed)

        # Generate 'dimension' number of floats between -1.0 and 1.0
        embedding = [round(random.uniform(-1.0, 1.0), 6) for _ in range(dimension)]
        return embedding

    def process(
        self, data: Union[str, List[str]], context: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Processes the input data (a string or a list of strings) and generates
        simulated embedding vectors for each text.

        The embedding dimension can be specified during node initialization or
        overridden by providing an `embedding_dimension` key in the `context` dictionary.

        Args:
            data (Union[str, List[str]]): The text or list of texts to embed.
            context (Dict[str, Any]): A dictionary containing additional runtime
                                      information, such as:
                                      - "embedding_dimension" (int, optional): Overrides the
                                        default embedding dimension for this specific call.

        Returns:
            List[List[float]]: A list of embedding vectors, where each vector
                               is a list of floats, corresponding to the input texts.

        Raises:
            TypeError: If the input data is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If the input data is `None`.
            RuntimeError: If an unexpected error occurs during embedding generation
                          for any of the texts.
        """
        logger.info(f"[{self.node_name}] Starting embedding generation process.")

        texts_to_embed: List[str] = []

        if data is None:
            logger.error(f"[{self.node_name}] Input data is None. Cannot generate embeddings.")
            raise ValueError("Input data cannot be None.")
        elif isinstance(data, str):
            texts_to_embed = [data]
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. Expected List[str]."
                )
                raise TypeError("All elements in the input list must be strings.")
            texts_to_embed = data
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data)}. Expected str or List[str]."
            )
            raise TypeError("Input data must be a string or a list of strings.")

        if not texts_to_embed:
            logger.warning(
                f"[{self.node_name}] Received an empty list of texts to embed. Returning empty list."
            )
            return []

        # Determine the embedding dimension to use for this process call
        configured_dimension = context.get("embedding_dimension")
        current_embedding_dimension = self._initialized_embedding_dimension

        if configured_dimension is not None:
            if isinstance(configured_dimension, int) and configured_dimension > 0:
                current_embedding_dimension = configured_dimension
                logger.debug(
                    f"[{self.node_name}] Using embedding_dimension from context: "
                    f"{current_embedding_dimension}"
                )
            else:
                logger.warning(
                    f"[{self.node_name}] Invalid 'embedding_dimension' in context ({configured_dimension}). "
                    f"Using initialized dimension: {current_embedding_dimension}."
                )

        embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_embed):
            try:
                # Call the simulated embedding generation function with the determined dimension
                embedding_vector = self._generate_single_embedding(
                    text, current_embedding_dimension
                )
                embeddings.append(embedding_vector)
                logger.debug(
                    f"[{self.node_name}] Generated embedding for text index {i} "
                    f"(dimension: {current_embedding_dimension})."
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Error generating embedding for text '{text[:50]}...': {e}",
                    exc_info=True,
                )
                # Re-raise as a RuntimeError to signal a critical failure
                raise RuntimeError(
                    f"Failed to generate embedding for text: '{text[:50]}...'"
                ) from e

        logger.info(
            f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings "
            f"with dimension {current_embedding_dimension}."
        )
        return embeddings