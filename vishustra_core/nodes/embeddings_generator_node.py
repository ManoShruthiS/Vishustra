import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A processing node designed to generate vector embeddings from text data.

    This node takes either a single string or a list of strings and
    simulates the creation of corresponding fixed-dimension vector embeddings.
    The simulation ensures deterministic output for identical inputs and
    supports configurable embedding dimensions via the processing context.
    """

    DEFAULT_EMBEDDING_DIMENSION: int = 768
    """The default dimension for generated embeddings if not specified in the context."""

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]], None]:
        """
        Generates vector embeddings for the provided input data.

        The `context` dictionary can optionally specify the desired `embedding_dim`
        (an integer greater than 0) for the output vectors. If not provided or
        invalid, `DEFAULT_EMBEDDING_DIMENSION` will be used.

        Args:
            data (Any): The input data, expected to be a `str` for a single text
                        or a `List[str]` for multiple texts.
            context (Dict[str, Any]): A dictionary containing runtime configuration
                                      and contextual information for the node.

        Returns:
            Union[List[float], List[List[float]], None]:
                - A `List[float]` if the input `data` was a single string.
                - A `List[List[float]]` if the input `data` was a list of strings.
                - `None` if the input data is `None`, of an unsupported type,
                  or if an error occurred during processing.
        """
        embedding_dim = context.get('embedding_dim', self.DEFAULT_EMBEDDING_DIMENSION)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.warning(
                f"Invalid or non-positive 'embedding_dim' '{embedding_dim}' provided in context. "
                f"Using default dimension: {self.DEFAULT_EMBEDDING_DIMENSION}."
            )
            embedding_dim = self.DEFAULT_EMBEDDING_DIMENSION

        if data is None:
            logger.debug("Received None as input data. Returning None.")
            return None
        elif isinstance(data, str):
            logger.debug(f"Processing single text input (first 20 chars: '{data[:20]}').")
            return self._generate_single_embedding(data, embedding_dim)
        elif isinstance(data, list):
            logger.debug(f"Processing list of {len(data)} text inputs.")
            results = []
            for item_idx, item in enumerate(data):
                if isinstance(item, str):
                    results.append(self._generate_single_embedding(item, embedding_dim))
                else:
                    logger.warning(
                        f"Skipping non-string item at index {item_idx} in input list "
                        f"(type: {type(item).__name__}). Only strings are supported."
                    )
            return results
        else:
            logger.error(
                f"Unsupported input data type for EmbeddingsGenerator: '{type(data).__name__}'. "
                "Expected str or List[str]."
            )
            return None

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        This method uses a hash-based pseudo-random number generator to produce
        a deterministic list of floats. This ensures that the same text input
        always yields the same simulated embedding vector for a given dimension.

        Args:
            text (str): The text string for which to generate an embedding.
            dimension (int): The desired dimension of the output embedding vector.

        Returns:
            List[float]: A list of floats representing the generated embedding vector.
                         Returns a vector of zeros if the input `text` is empty.
        """
        if not text:
            logger.debug("Received an empty string for embedding generation. Returning a zero vector.")
            return [0.0] * dimension

        # Use a seed derived from the text's hash to ensure deterministic "embeddings"
        # The seed is capped to fit common PRNG requirements (e.g., Python's random.seed takes an int).
        seed = hash(text) % (2**32 - 1)
        rng = random.Random(seed)

        embedding = [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dimension)]
        logger.debug(
            f"Successfully generated simulated embedding of dimension {dimension} "
            f"for text (first 20 chars: '{text[:20]}...')"
        )
        return embedding
