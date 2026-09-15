import logging
import random
from typing import Any, Dict, List, Union

# Assuming vishustra_core is a package and base_node.py is within vishustra_core/nodes/
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node responsible for generating text embeddings.

    This node takes a string or a list of strings as input and produces
    a corresponding embedding vector (list of floats) or a list of embedding
    vectors. It allows for configuration of embedding dimensions and a
    simulated model name via the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embeddings.

        The `data` input is expected to be either a single string or a list of strings.
        The `context` dictionary can optionally specify:
        - `embedding_dimensions` (int): The desired dimensionality of the embedding vectors.
                                        Defaults to 1536. Must be a positive integer.
        - `model_name` (str): A descriptive name for the embedding model being simulated.
                              Defaults to "simulated-embedding-model".

        Args:
            data: The text content or a list of text contents for which embeddings are to be generated.
            context: A dictionary containing operational parameters for the node.

        Returns:
            A list of floats representing the embedding vector if `data` was a string.
            A list of lists of floats (each inner list being an embedding vector) if `data` was a list of strings.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if a list contains non-string elements, or if it's an empty list.
            RuntimeError: If an unexpected issue occurs during the embedding generation process.
        """
        logger.debug(f"[{self.node_name}] Initiating process for input data of type: {type(data).__name__}.")

        # --- Input Data Validation ---
        if not isinstance(data, (str, list)):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str' or 'List[str]', "
                f"received '{type(data).__name__}'."
            )
            raise ValueError(f"Input 'data' must be a string or a list of strings, but got {type(data).__name__}.")

        if isinstance(data, list):
            if not data:
                logger.warning(f"[{self.node_name}] Received an empty list for embedding generation. Returning an empty list.")
                return []
            if not all(isinstance(item, str) for item in data):
                logger.error(f"[{self.node_name}] Invalid list content. All elements in 'data' list must be strings.")
                raise ValueError("If 'data' is a list, all its elements must be strings.")

        # --- Context Parameter Configuration ---
        embedding_dimensions = context.get('embedding_dimensions', 1536)
        model_name = context.get('model_name', "simulated-embedding-model")

        if not isinstance(embedding_dimensions, int) or embedding_dimensions <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'embedding_dimensions' in context ({embedding_dimensions}). "
                f"Expected a positive integer. Defaulting to 1536 dimensions."
            )
            embedding_dimensions = 1536
        
        logger.info(
            f"[{self.node_name}] Using simulated model '{model_name}' "
            f"to generate embeddings with {embedding_dimensions} dimensions."
        )

        # --- Embedding Generation Simulation ---
        try:
            if isinstance(data, str):
                # Process a single string
                embedding = self._generate_single_embedding(data, embedding_dimensions)
                logger.debug(f"[{self.node_name}] Generated a single embedding vector of length {len(embedding)}.")
                return embedding
            else: # data is List[str]
                # Process a list of strings
                embeddings = [self._generate_single_embedding(item, embedding_dimensions) for item in data]
                logger.debug(f"[{self.node_name}] Generated {len(embeddings)} embedding vectors.")
                return embeddings
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unhandled exception occurred during embedding generation. "
                f"This indicates a fault in the simulation logic or environment setup."
            )
            raise RuntimeError(f"Failed to generate embeddings due to an internal error: {e}") from e

    def _generate_single_embedding(self, text: str, dimensions: int) -> List[float]:
        """
        Helper method to simulate the generation of a single embedding vector.
        
        In a production system, this method would typically interface with an
        external embedding model API (e.g., OpenAI, Hugging Face, local ONNX model).
        For this simulation, it produces a vector of random floats.

        Args:
            text: The input string for which to generate an embedding.
            dimensions: The desired dimensionality of the embedding vector.

        Returns:
            A list of floats representing the simulated embedding vector.
        """
        # A common practice for empty strings is to return a zero vector or a special token's embedding.
        # Here, we opt for a zero vector to maintain vector space properties.
        if not text.strip(): # Check for functionally empty string (whitespace only)
            logger.debug(f"[{self.node_name}] Generating a zero vector for an effectively empty text input.")
            return [0.0] * dimensions
        
        # Simulate an embedding vector with random float values between -1.0 and 1.0
        return [random.uniform(-1.0, 1.0) for _ in range(dimensions)]
