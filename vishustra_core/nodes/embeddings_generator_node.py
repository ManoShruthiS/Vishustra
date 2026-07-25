import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that generates simulated text embeddings.

    This node takes text input (single string or list of strings) and
    produces a corresponding list of numerical embedding vectors.
    The embedding generation is simulated for demonstration purposes.
    """

    _DEFAULT_EMBEDDING_DIMENSION = 768  # A common dimension for text embeddings

    def __init__(self, embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension: The desired dimension for the generated embedding vectors.
                                 Must be a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Initialization error: 'embedding_dimension' must be a positive integer. "
                f"Received: {embedding_dimension}"
            )
            raise ValueError("Embedding dimension must be a positive integer.")

        self._embedding_dimension = embedding_dimension
        logger.info(
            f"{self.node_name} initialized with default embedding dimension: {self._embedding_dimension}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGeneratorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[List[float]]:
        """
        Processes the input data to generate simulated embeddings.

        The `data` can be a single string or a list of strings.
        The `context` dictionary can optionally specify an `embedding_dimension`
        (int, > 0) to temporarily override the node's configured dimension for this
        specific processing call.

        Args:
            data: The text data (str or list[str]) for which to generate embeddings.
            context: A dictionary containing contextual information.
                     Can include 'embedding_dimension' (int) to override the
                     node's default/configured dimension.

        Returns:
            A list of lists of floats, where each inner list represents an
            embedding vector for the corresponding input text.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if an invalid embedding dimension is provided in context.
            RuntimeError: If an unexpected error occurs during simulated
                          embedding generation for any text item.
        """
        current_embedding_dim = self._embedding_dimension
        if 'embedding_dimension' in context:
            context_dim = context['embedding_dimension']
            if isinstance(context_dim, int) and context_dim > 0:
                current_embedding_dim = context_dim
                logger.debug(
                    f"{self.node_name} overriding embedding dimension for this call: {current_embedding_dim}"
                )
            else:
                logger.warning(
                    f"Invalid 'embedding_dimension' in context: '{context_dim}'. "
                    f"Expected a positive integer. Using node's configured dimension: "
                    f"{self._embedding_dimension}."
                )

        if isinstance(data, str):
            texts_to_embed = [data]
            is_single_input = True
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            texts_to_embed = data
            is_single_input = False
        else:
            logger.error(
                f"Invalid input data type for {self.node_name}. "
                f"Expected str or list[str], received {type(data)}."
            )
            raise ValueError("Input data must be a string or a list of strings.")

        generated_embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_embed):
            try:
                # Simulate embedding generation: return a fixed-size vector of random floats.
                # In a production environment, this would involve calling a real embedding model
                # (e.g., via an API or an in-process library).
                embedding_vector = [random.uniform(-1.0, 1.0) for _ in range(current_embedding_dim)]
                generated_embeddings.append(embedding_vector)
                logger.debug(
                    f"Generated simulated embedding for item {i+1}/{len(texts_to_embed)} "
                    f"(dimension: {current_embedding_dim})"
                )
            except Exception as e:
                logger.exception(
                    f"An unexpected error occurred during simulated embedding generation "
                    f"for text item {i+1} in {self.node_name}: {e}"
                )
                # Depending on requirements, one might log the error and skip,
                # return a partial result, or re-raise. For robustness, we re-raise.
                raise RuntimeError(
                    f"Failed to generate embedding for text item {i+1} due to an internal error."
                ) from e

        logger.info(
            f"Successfully processed {len(texts_to_embed)} text(s) and generated "
            f"{len(generated_embeddings)} embedding(s) of dimension {current_embedding_dim}."
        )

        return generated_embeddings

