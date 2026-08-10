import logging
import random
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the environment
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of vector embeddings
    for given text data.

    This node expects input `data` to be either a single string or a list of strings.
    It simulates calling an embedding model and returns corresponding fixed-dimension
    vector embeddings.

    Configuration can be provided via the `context` dictionary:
    - `embedding_dimension` (int, optional): The desired dimension of the output embeddings.
      Defaults to 768 if not provided or invalid.
    - `embedding_model_name` (str, optional): A descriptive name for the simulated
      embedding model. Defaults to "simulated-bge-large".
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGeneratorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Generates vector embeddings for the input data.

        Args:
            data (str or List[str]): The text or list of texts to embed.
            context (Dict[str, Any]): A dictionary containing runtime context and
                                       configuration for the node, e.g.,
                                       'embedding_dimension' and 'embedding_model_name'.

        Returns:
            List[float] or List[List[float]]: A single embedding vector if input was a string,
                                               or a list of embedding vectors if input was a list of strings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
            ValueError: If `embedding_dimension` in context is invalid.
        """
        # Determine embedding dimension from context, defaulting to a common size
        embedding_dimension = context.get("embedding_dimension", 768)
        model_name = context.get("embedding_model_name", "simulated-bge-large")

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"Invalid 'embedding_dimension' in context: {embedding_dimension}. "
                f"Must be a positive integer. Using default {768}."
            )
            embedding_dimension = 768

        def _generate_single_embedding(text: str) -> List[float]:
            """
            Simulates the generation of a single embedding vector.
            In a real-world scenario, this would involve calling an actual
            embedding model (e.g., via an API or a local model inference).
            """
            logger.debug(
                f"Generating {embedding_dimension}-dim embedding for text (first 50 chars): "
                f"'{text[:50]}...' using model '{model_name}'"
            )
            # Simulate embedding by generating random floats between -1 and 1
            return [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]

        if isinstance(data, str):
            if not data.strip():
                logger.warning(
                    "Received an empty or whitespace-only string for embedding generation. "
                    "Returning a zero vector."
                )
                return [0.0] * embedding_dimension
            return _generate_single_embedding(data)
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                non_str_items = [item for item in data if not isinstance(item, str)]
                error_msg = (
                    f"Expected list of strings, but received a list containing "
                    f"non-string elements. First few non-strings: "
                    f"{[type(item).__name__ for item in non_str_items][:3]}."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

            embeddings = []
            for i, item in enumerate(data):
                if not item.strip():
                    logger.warning(
                        f"Received an empty or whitespace-only string at index {i} "
                        f"in the list for embedding generation. Returning a zero vector for this item."
                    )
                    embeddings.append([0.0] * embedding_dimension)
                else:
                    embeddings.append(_generate_single_embedding(item))
            return embeddings
        else:
            error_msg = (
                f"Unsupported data type for embedding generation: {type(data).__name__}. "
                "Expected str or List[str]."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)