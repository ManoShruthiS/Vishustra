import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is correctly available at this path in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that generates vector embeddings for text data.

    This node is designed to interface with an underlying embedding model (simulated
    in this implementation) to convert input text into numerical vector representations.
    It supports processing single strings or a list of strings.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGeneratorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[List[float]]:
        """
        Processes the input text data to generate embeddings.

        This method simulates the generation of embeddings. In a production
        environment, it would interact with an actual embedding model API
        or a local model instance.

        Args:
            data: The input text(s) for which to generate embeddings. Can be a single
                  string or a list of strings.
            context: A dictionary containing operational parameters for the node.
                     Expected keys:
                     - 'embedding_dimension' (int, optional): The desired dimensionality
                       of the output embeddings. Must be a positive integer. Defaults to 768.

        Returns:
            A list of lists of floats, where each inner list represents the
            generated embedding vector for a corresponding input text.

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If 'embedding_dimension' in the context is not a positive integer.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"Invalid input data type for EmbeddingsGeneratorNode. Expected str or List[str], "
                f"but received {type(data)}."
            )
            raise TypeError("Input 'data' must be a string or a list of strings.")

        # Normalize input to always be a list of strings for consistent processing
        if isinstance(data, str):
            texts = [data]
        else:
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    "Invalid input list for EmbeddingsGeneratorNode. All elements "
                    "in the input 'data' list must be strings."
                )
                raise TypeError("All elements in the input 'data' list must be strings.")
            texts = data

        # Retrieve and validate embedding_dimension from context
        embedding_dimension = context.get("embedding_dimension", 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Invalid 'embedding_dimension' in context. Expected a positive integer, "
                f"but received {embedding_dimension} (type: {type(embedding_dimension)})."
            )
            raise ValueError("Context parameter 'embedding_dimension' must be a positive integer.")

        generated_embeddings: List[List[float]] = []
        for text in texts:
            # Simulate embedding generation:
            # A deterministic pseudo-random generator is used based on the text hash.
            # This ensures that calling `process` with the same text will yield
            # an identical simulated embedding for testing and consistency.
            # In a real scenario, this would be replaced by an actual model inference call.
            seed_value = hash(text) % (2**32 - 1)  # Ensure seed is within common int range
            text_rng = random.Random(seed_value)
            
            simulated_embedding = [text_rng.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]
            generated_embeddings.append(simulated_embedding)

        logger.debug(
            f"Successfully generated {len(generated_embeddings)} simulated embeddings "
            f"of dimension {embedding_dimension}."
        )
        return generated_embeddings

