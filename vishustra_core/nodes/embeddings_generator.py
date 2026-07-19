import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGenerator(BaseNode):
    """
    A Vishustra node that generates numerical embeddings for text data.

    This node takes text input (either a single string or a list of strings)
    and, in a real-world scenario, would leverage an external or internal
    embedding model to convert these texts into fixed-size numerical vectors.
    For demonstration purposes, it simulates this process by generating
    random float vectors.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for the input text data.

        The `process` method expects `data` to be a string or a list of strings.
        It looks for 'embedding_dimension' (an integer) in the `context` dictionary
        to determine the size of the generated embedding vectors. Optionally,
        'embedding_model_name' (a string) can be provided in the `context` for
        logging and future integration with specific models.

        Args:
            data: The input text (str) or a list of texts (list[str]) to embed.
            context: A dictionary containing operational parameters, including:
                     - 'embedding_dimension' (int): The desired dimension for the embeddings.
                     - 'embedding_model_name' (str, optional): Name of the embedding model
                                                               for logging/future integration.

        Returns:
            A list of floats if a single string was provided as input, or a list of
            lists of floats if a list of strings was provided. Each inner list
            represents an embedding vector.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if elements within an input list are not strings.
            ValueError: If 'embedding_dimension' is missing from `context` or is
                        not a positive integer.
        """
        logger.debug(f"EmbeddingsGenerator node received data for processing. Type: {type(data)}")

        if not isinstance(data, (str, list)):
            logger.error(
                f"Invalid input data type for EmbeddingsGenerator. Expected str or list[str], "
                f"but received {type(data)}."
            )
            raise TypeError("EmbeddingsGenerator expects data to be a string or a list of strings.")

        if isinstance(data, list) and not all(isinstance(item, str) for item in data):
            logger.error(
                "Invalid input data: list contains non-string elements. "
                "All elements in the input list must be strings."
            )
            raise TypeError("EmbeddingsGenerator expects all elements in the input list to be strings.")

        embedding_dimension = context.get('embedding_dimension')
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Configuration error: Missing or invalid 'embedding_dimension' in context. "
                f"Received: {embedding_dimension} (expected a positive integer)."
            )
            raise ValueError(
                "Context must contain a positive integer 'embedding_dimension' for EmbeddingsGenerator."
            )

        embedding_model_name = context.get('embedding_model_name', 'simulated_embedding_model')
        logger.info(
            f"Generating embeddings using '{embedding_model_name}' with a dimension of {embedding_dimension}."
        )

        def _generate_single_embedding(text: str) -> List[float]:
            """
            Simulates the generation of an embedding vector for a single text string.
            In a production environment, this would interface with a real embedding model.
            """
            # For simulation, we generate random floats between -1.0 and 1.0.
            # This range is common for many embedding models.
            logger.debug(f"Simulating embedding for text (first 50 chars): '{text[:50]}...'")
            return [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]

        if isinstance(data, str):
            # Process a single string
            result = _generate_single_embedding(data)
            logger.debug(f"Successfully generated a single embedding of dimension {len(result)}.")
            return result
        else:
            # Process a list of strings
            results = [_generate_single_embedding(item) for item in data]
            logger.debug(
                f"Successfully generated {len(results)} embeddings, "
                f"each with dimension {embedding_dimension}."
            )
            return results