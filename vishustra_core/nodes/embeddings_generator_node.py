import logging
import random
from typing import Any, Dict, List, Union

# Assuming the core BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node responsible for generating text embeddings.

    This node accepts either a single string or a list of strings as input
    and outputs a simulated embedding vector for each text item.
    The embedding generation is a placeholder simulation using random values
    and does not involve actual machine learning models or API calls.
    It demonstrates the node's capability to transform textual data into
    vector representations.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to generate simulated embeddings.

        The method expects text data and configures the embedding dimension
        via the `context` dictionary.

        Args:
            data (Any): The input data to be embedded.
                        Expected types: `str` (single text) or `List[str]` (multiple texts).
            context (Dict[str, Any]): A dictionary containing runtime configuration
                                      parameters.
                                      Expected key:
                                      - 'embedding_dimension' (int, optional):
                                        The desired dimension of the output embedding vectors.
                                        Defaults to 768 if not provided or invalid.

        Returns:
            Any: A list of floats (if `data` was a single string) representing
                 one embedding, or a list of lists of floats (if `data` was
                 a list of strings) representing multiple embeddings.

        Raises:
            ValueError: If the input `data` is not of the expected type (str or List[str]),
                        or if a list input contains non-string elements.
        """
        logger.info(f"[{self.node_name}] Starting embedding generation process.")
        logger.debug(f"[{self.node_name}] Received data type: {type(data)}")

        if not isinstance(data, (str, list)):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' or 'List[str]', but received '{type(data)}'."
            )
            raise ValueError(
                f"Invalid data type for EmbeddingsGeneratorNode. "
                f"Expected 'str' or 'List[str]', but received '{type(data)}'."
            )

        # Determine embedding dimension from context, or use a sensible default
        embedding_dimension = context.get('embedding_dimension')
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            default_dim = 768
            if embedding_dimension is not None:
                logger.warning(
                    f"[{self.node_name}] Invalid or missing 'embedding_dimension' "
                    f"in context (received: {embedding_dimension}). "
                    f"Falling back to default dimension: {default_dim}."
                )
            embedding_dimension = default_dim
        
        logger.debug(f"[{self.node_name}] Configured embedding dimension: {embedding_dimension}")

        is_single_input = isinstance(data, str)
        texts_to_process: List[str] = [data] if is_single_input else data

        # Validate all elements in the list if it's a list input
        if not all(isinstance(text, str) for text in texts_to_process):
            first_invalid = next((type(t) for t in texts_to_process if not isinstance(t, str)), None)
            logger.error(
                f"[{self.node_name}] List input contains non-string elements. "
                f"First problematic element type: '{first_invalid}'."
            )
            raise ValueError(
                "List input for EmbeddingsGeneratorNode must exclusively contain string elements."
            )

        results: List[List[float]] = []
        for i, text_item in enumerate(texts_to_process):
            # Simulate an embedding vector: a list of random floats
            # In a production environment, this would involve calling a specific
            # embedding model (e.g., via an API or a local ML library).
            simulated_embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]
            results.append(simulated_embedding)
            logger.debug(
                f"[{self.node_name}] Generated simulated embedding for item "
                f"{i + 1}/{len(texts_to_process)} (dim: {embedding_dimension})."
            )

        logger.info(
            f"[{self.node_name}] Successfully generated {len(results)} simulated embedding(s)."
        )

        # Return a single embedding if the input was a single string,
        # otherwise return the list of embeddings.
        return results[0] if is_single_input else results