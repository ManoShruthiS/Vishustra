import logging
import random
from typing import Any, Dict, List, Union

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate text embeddings.

    This node takes text (or a list of texts) as input and returns
    a fixed-size vector representation (embedding) for each text.
    The embedding generation process is simulated for demonstration,
    and its dimension can be configured via the processing context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Embeddings Generator"

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        In a production environment, this method would interface with a pre-trained
        embedding model (e.g., from Hugging Face, OpenAI, Cohere) or an internal
        service to compute actual embeddings. For this simulation, it produces
        a list of random floating-point numbers.

        Args:
            text: The input string for which to generate an embedding.
            dimension: The desired dimensionality of the embedding vector.

        Returns:
            A list of floats representing the simulated embedding vector.
        """
        # A real implementation would involve a model inference call here.
        # For simulation, we generate random floats.
        return [random.uniform(-1.0, 1.0) for _ in range(dimension)]

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (text or list of texts) to generate embeddings.

        Args:
            data: The input for which embeddings are to be generated.
                  Expected types:
                  - `str`: A single text string.
                  - `list[str]`: A list of text strings.
            context: A dictionary containing operational parameters for the node.
                     Expected keys:
                     - 'embedding_dimension' (int, optional): The desired dimension
                       of the output embedding vectors. Defaults to 768.

        Returns:
            `List[float]` if `data` was a single string, representing one embedding.
            `List[List[float]]` if `data` was a list of strings, representing
            a list of embeddings.

        Raises:
            TypeError: If the input `data` type is not `str` or `list[str]`,
                       or if context is not a dictionary, or if list elements are not strings.
            ValueError: If `embedding_dimension` in context is not a positive integer.
            Exception: Propagates any underlying errors during embedding generation.
        """
        logger.info(f"[{self.node_name}] Starting process for input data type: {type(data)}")

        if not isinstance(context, dict):
            logger.error(f"[{self.node_name}] Invalid context type. Expected dict, got {type(context)}.")
            raise TypeError("Node context must be a dictionary.")

        embedding_dimension = context.get('embedding_dimension', 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"[{self.node_name}] Invalid 'embedding_dimension' in context: {embedding_dimension}. "
                         "Must be a positive integer.")
            raise ValueError("Parameter 'embedding_dimension' must be a positive integer.")

        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing a single text input.")
            try:
                embedding = self._generate_single_embedding(data, embedding_dimension)
                logger.info(f"[{self.node_name}] Successfully generated embedding for single text.")
                return embedding
            except Exception as e:
                logger.error(f"[{self.node_name}] Failed to generate embedding for single text: {e}", exc_info=True)
                raise
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(f"[{self.node_name}] Input list contains non-string elements. All elements must be strings.")
                raise TypeError("All elements in the input list must be strings.")

            logger.debug(f"[{self.node_name}] Processing a list of {len(data)} texts.")
            embeddings: List[List[float]] = []
            for i, text_item in enumerate(data):
                try:
                    embedding = self._generate_single_embedding(text_item, embedding_dimension)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"[{self.node_name}] Failed to generate embedding for item {i}: {e}", exc_info=True)
                    # For strict pipelines, re-raise. For fault-tolerant, consider logging and skipping/returning partial.
                    raise
            logger.info(f"[{self.node_name}] Successfully generated embeddings for {len(data)} texts.")
            return embeddings
        else:
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str or list[str], got {type(data)}.")
            raise TypeError("Input 'data' must be a string or a list of strings.")