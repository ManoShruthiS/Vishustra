import logging
import random
from typing import Any, Dict, List, Union

# Assuming this import path is consistent with the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate or simulate text embeddings.

    This node accepts either a single string or a list of strings as input
    and processes them to return corresponding numerical vector representations.
    For this implementation, the embedding generation is simulated using random
    vectors to demonstrate the node's interface and error handling capabilities.
    In a production environment, this would integrate with a concrete embedding
    model or an external API.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text(s) to generate simulated embeddings.

        The `context` dictionary can specify 'vector_dim' to control the
        dimension of the output embedding vectors. If not provided or invalid,
        it defaults to 768.

        Args:
            data: The input data, expected to be a `str` (single text) or
                  `List[str]` (multiple texts) for embedding.
            context: A dictionary containing operational parameters.
                     - `vector_dim` (int, optional): The desired dimension for
                       the generated embedding vectors. Defaults to 768.

        Returns:
            A `List[float]` representing the embedding for a single input string.
            A `List[List[float]]` representing embeddings for a list of input strings.
            Returns an empty list or list of empty lists if the input is empty
            or contains no valid data for embedding.

        Raises:
            TypeError: If the input `data` is not a `str` or `List[str]`.
            ValueError: If an unexpected issue occurs during the embedding
                        generation process (e.g., in `_generate_single_embedding`).
        """
        vector_dim = context.get("vector_dim", 768)
        
        if not isinstance(vector_dim, int) or vector_dim <= 0:
            logger.warning(
                "Invalid 'vector_dim' in context. Expected a positive integer, "
                "received '%s' (%s). Defaulting to 768.", vector_dim, type(vector_dim).__name__
            )
            vector_dim = 768

        if isinstance(data, str):
            if not data.strip():
                logger.warning("Received an empty or whitespace-only string for embedding. Returning an empty embedding vector.")
                return []
            return self._generate_single_embedding(data, vector_dim)
        
        elif isinstance(data, list):
            if not data:
                logger.info("Received an empty list for embedding. Returning an empty list of embeddings.")
                return []
            
            embeddings = []
            for item_idx, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(
                        "List item at index %d is not a string (type: %s). Skipping this item.",
                        item_idx, type(item).__name__
                    )
                    continue
                
                if not item.strip():
                    logger.warning("Skipping embedding for an empty string at index %d within the list.", item_idx)
                    continue 
                
                try:
                    embeddings.append(self._generate_single_embedding(item, vector_dim))
                except ValueError as ve:
                    logger.error(
                        "Failed to generate embedding for item at index %d (first 50 chars: '%s'): %s",
                        item_idx, item[:50], ve
                    )
                except Exception as e: # Catch any other unexpected errors
                    logger.error(
                        "An unexpected error occurred while processing item at index %d (first 50 chars: '%s'): %s",
                        item_idx, item[:50], e
                    )
            return embeddings
        
        else:
            logger.error(
                "Invalid data type for EmbeddingsGeneratorNode. Expected 'str' or 'List[str]', received '%s' (%s).",
                data, type(data).__name__
            )
            raise TypeError(
                f"Invalid data type for EmbeddingsGeneratorNode. Expected 'str' or 'List[str]', "
                f"but received {type(data).__name__}."
            )

    def _generate_single_embedding(self, text: str, dim: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        Args:
            text: The input string for which to generate an embedding.
            dim: The desired dimension of the embedding vector.

        Returns:
            A `List[float]` representing the simulated embedding vector.

        Raises:
            ValueError: If `dim` is not a positive integer or if an error
                        occurs during random vector generation (unlikely).
        """
        if not isinstance(text, str):
            logger.error("Internal error: _generate_single_embedding received non-string input type: %s", type(text).__name__)
            raise TypeError("Input 'text' must be a string.")
        if not isinstance(dim, int) or dim <= 0:
            logger.error("Internal error: _generate_single_embedding received invalid dimension: %s", dim)
            raise ValueError("Embedding dimension 'dim' must be a positive integer.")

        try:
            # Simulate an embedding vector with random floats between -1.0 and 1.0
            embedding = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            logger.debug(
                "Successfully generated simulated embedding of dimension %d for text (first 20 chars): '%s'",
                dim, text[:20]
            )
            return embedding
        except Exception as e:
            logger.error(
                "Failed to generate simulated embedding for text (first 20 chars: '%s') with dimension %d: %s",
                text[:20], dim, e
            )
            raise ValueError(f"Failed to generate simulated embedding: {e}") from e