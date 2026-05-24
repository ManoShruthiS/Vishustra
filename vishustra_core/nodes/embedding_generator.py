import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of embeddings for input text.
    It generates random float vectors as a placeholder for actual embeddings,
    useful for testing and development pipelines where a real embedding model
    might not be available or needed.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes input text (or list of texts) and simulates generating
        fixed-size embedding vectors.

        Args:
            data: The input text (str) or a list of texts (List[str]) to embed.
            context: A dictionary containing operational context.
                     Must include 'embedding_dimension' (int) to define the
                     size of the simulated embedding vector.

        Returns:
            A list of floats representing a single embedding if a single string was
            provided, or a list of lists of floats if a list of texts was provided.

        Raises:
            ValueError: If 'embedding_dimension' is missing, invalid, or non-positive
                        in the context.
            TypeError: If input 'data' is not a string or a list of strings,
                       or if the list contains non-string elements.
        """
        logger.debug(f"[{self.node_name}] Starting process for data of type: {type(data).__name__}")

        embedding_dimension = context.get('embedding_dimension')
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"[{self.node_name}] Configuration error: 'embedding_dimension' "
                f"in context must be a positive integer. Got: {embedding_dimension}"
            )
            raise ValueError(
                f"Context for {self.node_name} must contain a positive integer "
                f"'embedding_dimension'. Received: {embedding_dimension}"
            )

        texts_to_process: List[str]
        is_single_input_string = False

        if isinstance(data, str):
            texts_to_process = [data]
            is_single_input_string = True
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(f"[{self.node_name}] Input list contains non-string elements.")
                raise TypeError(
                    f"Input 'data' for {self.node_name} must be a string or a list of strings. "
                    f"Detected non-string elements within the input list."
                )
            texts_to_process = data
        else:
            logger.error(f"[{self.node_name}] Invalid input data type: {type(data).__name__}")
            raise TypeError(
                f"Input 'data' for {self.node_name} must be a string or a list of strings. "
                f"Received type: {type(data).__name__}"
            )

        generated_embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_process):
            # Simulate embedding generation: a list of random floats
            # Values are typically between -1 and 1 or 0 and 1 for common embedding spaces.
            embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]
            generated_embeddings.append(embedding)
            logger.debug(
                f"[{self.node_name}] Generated simulated embedding (dim={embedding_dimension}) "
                f"for text item {i+1}/{len(texts_to_process)} (first 30 chars): '{text[:30]}...'"
            )

        logger.debug(
            f"[{self.node_name}] Finished processing. Generated {len(generated_embeddings)} "
            f"simulated embeddings."
        )

        if is_single_input_string:
            return generated_embeddings[0]  # Return single embedding for single string input
        else:
            return generated_embeddings  # Return list of embeddings for list of strings input
