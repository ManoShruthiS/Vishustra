
import logging
import random
from typing import Any, Dict, List, Union

# Assuming this import path based on the project context and requirements
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of vector embeddings
    for input text data.

    This node expects text input (a single string or a list of strings) and
    produces a list of corresponding embedding vectors. The dimensions of the
    embeddings can be configured via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> List[List[float]]:
        """
        Generates simulated embeddings for the input text data.

        The `data` input is expected to be a single string or a list of strings.
        The `context` dictionary can optionally specify 'embedding_dimensions'
        (an integer > 0, default: 768).

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing runtime context and
                                       configuration, potentially including 'embedding_dimensions'.

        Returns:
            List[List[float]]: A list of embedding vectors, where each vector
                               is a list of floats.

        Raises:
            TypeError: If the input data is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If 'embedding_dimensions' in the context is not a positive integer.
        """
        logger.debug(f"[{self.node_name}] Starting process for input data of type: {type(data)}")

        # Validate input data type
        if not isinstance(data, (str, list)):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected str or list of str, but received {type(data).__name__}."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string or a list of strings, "
                f"but got {type(data).__name__}."
            )

        # Determine embedding dimensions from context, default to 768
        embedding_dimensions = context.get("embedding_dimensions", 768)

        if not isinstance(embedding_dimensions, int) or embedding_dimensions <= 0:
            logger.error(
                f"[{self.node_name}] Invalid 'embedding_dimensions' in context. "
                f"Expected a positive integer, but received {embedding_dimensions}."
            )
            raise ValueError(
                f"'embedding_dimensions' in context must be a positive integer, "
                f"but got {embedding_dimensions} of type {type(embedding_dimensions).__name__}."
            )

        # Standardize input to a list of strings for uniform processing
        input_texts: List[str]
        if isinstance(data, str):
            input_texts = [data]
        else: # data is a list
            # Validate that all elements in the list are strings
            if not all(isinstance(item, str) for item in data):
                non_string_types = {type(item).__name__ for item in data if not isinstance(item, str)}
                logger.error(
                    f"[{self.node_name}] Invalid list elements. "
                    f"All elements in data list must be strings, but found non-string types: {', '.join(non_string_types)}."
                )
                raise TypeError(
                    f"If input data is a list, all its elements must be strings. "
                    f"Found non-string types: {', '.join(non_string_types)}."
                )
            input_texts = data

        generated_embeddings: List[List[float]] = []
        for i, text in enumerate(input_texts):
            # Simulate embedding generation: create a vector of random floats.
            # In a production system, this would involve calling a robust
            # embedding model (e.g., from an LLM provider or a local model).
            embedding_vector = [random.uniform(-1.0, 1.0) for _ in range(embedding_dimensions)]
            generated_embeddings.append(embedding_vector)
            logger.debug(
                f"[{self.node_name}] Generated embedding for text snippet {i+1} "
                f"(first 30 chars: '{text[:30]}...') with {embedding_dimensions} dimensions."
            )

        logger.debug(
            f"[{self.node_name}] Finished process. "
            f"Generated {len(generated_embeddings)} embedding vectors."
        )
        return generated_embeddings
