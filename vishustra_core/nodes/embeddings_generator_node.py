import logging
import hashlib
import random
from typing import Any, Dict, List, Union

# Assuming this path is correctly set up in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of text embeddings.

    This node takes text (either a single string or a list of strings)
    and produces a list of simulated embedding vectors. The embedding
    generation is deterministic for a given text and configuration, making
    it suitable for testing and development environments where a full
    embedding service is not required.

    Configuration can be provided via the context dictionary:
    - 'embedding_dimension': int, desired dimension of the embedding vectors (default: 768).
    - 'model_name': str, a descriptive name for the simulated embedding model (default: "simulated-embedding-model").
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates generating a single embedding vector for a given text.
        The generation is deterministic based on the text hash, ensuring
        consistent outputs for identical inputs.
        """
        if not text:
            logger.warning("Attempted to generate embedding for empty text. Returning zero vector.")
            return [0.0] * dimension

        # Use SHA-256 hash of the text to seed the random number generator.
        # This makes the "embedding" deterministic for a given text, crucial for reproducibility.
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        # Cap the seed to prevent very large integers which might cause issues with some RNG implementations.
        seed = int(text_hash, 16) % (2**32 - 1)

        rng = random.Random(seed)
        # Generate 'dimension' number of floats between -1.0 and 1.0.
        # This range provides a plausible simulation for embedding values,
        # though a real model's distribution would be more complex.
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        return embedding

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[List[float]]:
        """
        Processes the input data (text or list of texts) to generate simulated embeddings.

        Args:
            data (Union[str, List[str]]): The input text(s) for which to generate embeddings.
                                          Accepts either a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing runtime context and
                                      configuration parameters for the embedding process.
                                      Expected keys:
                                      - 'embedding_dimension': int (optional, default: 768)
                                        The desired length of the output embedding vectors.
                                      - 'model_name': str (optional, default: "simulated-embedding-model")
                                        A label for the simulated model, useful for logging and metadata.

        Returns:
            List[List[float]]: A list of embedding vectors. Each inner list represents an
                               embedding vector corresponding to one input text. If the
                               input 'data' was a single string, the returned list will
                               contain exactly one vector.

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If 'embedding_dimension' in the context is not a positive integer.
            RuntimeError: If an unexpected error occurs during the simulated embedding
                          generation for a specific text item.
        """
        embedding_dimension = context.get('embedding_dimension', 768)
        model_name = context.get('model_name', "simulated-embedding-model")

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Configuration error in {self.node_name}: 'embedding_dimension' must be a positive integer. "
                f"Received: {embedding_dimension} (type: {type(embedding_dimension).__name__})."
            )
            raise ValueError(
                f"Invalid 'embedding_dimension' in context for {self.node_name}. "
                f"Must be a positive integer, but got {embedding_dimension}."
            )

        texts_to_embed: List[str]
        # Keep track if the original input was a single string for potential future handling,
        # though current output always returns a list of lists.
        # is_single_input = False

        if isinstance(data, str):
            texts_to_embed = [data]
            # is_single_input = True
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    f"Invalid input data for {self.node_name}: List contains non-string elements. "
                    f"Elements types: {[type(item).__name__ for item in data if not isinstance(item, str)]}"
                )
                raise TypeError(
                    f"Input for {self.node_name} must be a string or a list of strings. "
                    f"Found non-string elements in list input."
                )
            texts_to_embed = data
        else:
            logger.error(
                f"Invalid input data type for {self.node_name}: Expected str or List[str], "
                f"but got {type(data).__name__}."
            )
            raise TypeError(
                f"Input for {self.node_name} must be a string or a list of strings, "
                f"but got {type(data).__name__}."
            )

        if not texts_to_embed:
            logger.warning(f"No texts provided for embedding in {self.node_name}. Returning an empty list of embeddings.")
            return []

        logger.info(
            f"Generating {embedding_dimension}-dimensional embeddings for {len(texts_to_embed)} text(s) "
            f"using simulated model '{model_name}'."
        )

        embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_embed):
            try:
                embedding = self._generate_single_embedding(text, embedding_dimension)
                embeddings.append(embedding)
            except Exception as e:
                # Log the specific failure and re-raise to indicate a critical processing issue.
                logger.exception(
                    f"Failed to generate simulated embedding for text item {i} (first 50 chars: '{text[:50]}...'). "
                    f"Error: {e}"
                )
                raise RuntimeError(
                    f"Critical error in {self.node_name}: Failed to generate embedding for a text item."
                ) from e

        logger.debug(f"Successfully generated {len(embeddings)} embeddings for {self.node_name}.")
        return embeddings