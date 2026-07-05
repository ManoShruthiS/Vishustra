import logging
import random
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node is available in the project structure.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of embeddings
    for text input.

    This node is designed to take a single string or a list of strings and
    return a simulated embedding vector (list of floats) or a list of
    such vectors, respectively. It allows for configurable embedding
    dimensions and reproducible simulations via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input text data to generate simulated embeddings.

        This method simulates an embedding model by generating random float
        vectors. In a production environment, this would interface with a
        real embedding service (e.g., OpenAI, Hugging Face, custom model).

        Args:
            data: The input text to embed. Can be a single string or a list of strings.
            context: A dictionary containing operational context and parameters.
                     Expected keys:
                     - 'embedding_dimension' (int, optional): The desired dimension of the
                       output embedding vectors. Defaults to 768. Must be a positive integer.
                     - 'random_seed' (int, optional): An integer seed for the random number
                       generator to ensure reproducible simulated embeddings.

        Returns:
            A list of floats representing the simulated embedding for a single
            string input, or a list of lists of floats for a list of string inputs.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
            ValueError: If an empty list of strings is provided, or if any element
                        in the list is not a string.
        """
        # Determine embedding dimension from context, default to 768
        embedding_dimension = context.get('embedding_dimension', 768)
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"{self.node_name}: Invalid 'embedding_dimension' '{embedding_dimension}' "
                f"in context. Must be a positive integer. Falling back to default: 768."
            )
            embedding_dimension = 768

        # Configure random seed for reproducibility if provided
        random_seed = context.get('random_seed')
        if random_seed is not None:
            if isinstance(random_seed, int):
                random.seed(random_seed)
                logger.debug(f"{self.node_name}: Set random seed to {random_seed} for reproducibility.")
            else:
                logger.warning(
                    f"{self.node_name}: Invalid 'random_seed' '{random_seed}' in context. "
                    f"Expected an integer. Ignoring seed and using system randomness."
                )
                random.seed(None) # Reset to system time-based seeding

        def _generate_single_embedding(text_input: str) -> List[float]:
            """Helper function to generate a single simulated embedding vector."""
            # Generates a list of random floats between -1.0 and 1.0.
            # This simulates a typical normalized embedding vector.
            return [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]

        if isinstance(data, str):
            logger.debug(f"{self.node_name}: Processing single string input.")
            return _generate_single_embedding(data)
        elif isinstance(data, list):
            if not data:
                logger.warning(f"{self.node_name}: Received an empty list for embedding. Returning empty list.")
                return []

            embeddings: List[List[float]] = []
            for item in data:
                if not isinstance(item, str):
                    logger.error(
                        f"{self.node_name}: List input contains non-string element. "
                        f"Expected str, got {type(item).__name__} at index {data.index(item)}."
                    )
                    raise ValueError(
                        f"All elements in the input list for '{self.node_name}' must be strings. "
                        f"Found type {type(item).__name__}."
                    )
                embeddings.append(_generate_single_embedding(item))
            logger.debug(f"{self.node_name}: Generated {len(embeddings)} embeddings for a list of strings.")
            return embeddings
        else:
            logger.error(
                f"{self.node_name}: Invalid input data type. Expected str or list[str], "
                f"got {type(data).__name__}."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string or a list of strings. "
                f"Received type: {type(data).__name__}"
            )