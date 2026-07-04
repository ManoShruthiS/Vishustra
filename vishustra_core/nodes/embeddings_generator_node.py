import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of embeddings for text data.
    This node expects input data to be a single string or a list of strings
    and returns a corresponding embedding vector (list of floats) or a list
    of embedding vectors.

    Configuration for embedding dimension can be provided via the 'context' dictionary
    under 'embedding_config.dimension'.
    """

    _DEFAULT_EMBEDDING_DIM = 768
    """The default dimensionality for generated embeddings if not specified in context."""

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to generate simulated embeddings.

        This method will:
        1. Log the start of processing.
        2. Determine the embedding dimension, prioritizing `context['embedding_config']['dimension']`
           and falling back to `_DEFAULT_EMBEDDING_DIM`.
        3. Validate the input `data` to ensure it is a string or a list of strings.
        4. Generate a simulated embedding (a list of random floats) for each text.
        5. Log successful completion or any errors encountered.

        Args:
            data (Union[str, List[str]]): The text or list of texts to embed.
                                          Each text is expected to be a string.
            context (Dict[str, Any]): A dictionary containing runtime context
                                       and configuration. Expected to optionally
                                       contain a 'embedding_config' key, which
                                       can hold a 'dimension' key (e.g.,
                                       `{'embedding_config': {'dimension': 512}}`).

        Returns:
            Union[List[float], List[List[float]]]: A list of floats representing
                                                    the embedding for a single text,
                                                    or a list of such lists if multiple
                                                    texts were provided.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if a list contains non-string elements.
            Exception: For any unexpected errors during the embedding generation process.
        """
        logger.debug(f"[{self.node_name}] Starting processing. Input data type: {type(data)}")

        try:
            # Determine embedding dimension from context, falling back to default
            embedding_config = context.get('embedding_config', {})
            embedding_dim = embedding_config.get('dimension', self._DEFAULT_EMBEDDING_DIM)

            if not isinstance(embedding_dim, int) or embedding_dim <= 0:
                logger.warning(
                    f"[{self.node_name}] Invalid embedding_config dimension "
                    f"'{embedding_dim}' in context. Must be a positive integer. "
                    f"Using default: {self._DEFAULT_EMBEDDING_DIM}"
                )
                embedding_dim = self._DEFAULT_EMBEDDING_DIM

            def _generate_single_embedding(text_segment: str) -> List[float]:
                """
                Generates a single simulated embedding vector for a given text segment.
                In a production system, this would invoke a real embedding model.
                """
                # Validate that the segment is indeed a string before processing
                if not isinstance(text_segment, str):
                    raise TypeError(f"Expected text segment to be string, got {type(text_segment)}")
                
                # Simulate an embedding by generating a list of random floats
                return [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]

            if isinstance(data, str):
                logger.info(f"[{self.node_name}] Generating embedding for a single text.")
                embedding = _generate_single_embedding(data)
                logger.debug(f"[{self.node_name}] Generated single embedding of dimension {len(embedding)}.")
                return embedding
            elif isinstance(data, list):
                # Ensure all elements in the list are strings
                if not all(isinstance(item, str) for item in data):
                    raise ValueError("Input list for embedding generation must contain only strings.")
                
                logger.info(f"[{self.node_name}] Generating embeddings for a list of {len(data)} texts.")
                embeddings = [_generate_single_embedding(item) for item in data]
                if embeddings:
                    logger.debug(f"[{self.node_name}] Generated {len(embeddings)} embeddings, each of dimension {len(embeddings[0])}.")
                else:
                    logger.debug(f"[{self.node_name}] Generated an empty list of embeddings for empty input list.")
                return embeddings
            else:
                raise ValueError("Input data must be a string or a list of strings.")

        except (ValueError, TypeError) as validation_error:
            logger.error(f"[{self.node_name}] Input validation error: {validation_error}")
            raise # Re-raise the specific validation error for upstream handling
        except Exception as e:
            # Catch any other unexpected errors during the simulation
            logger.exception(f"[{self.node_name}] An unexpected error occurred during embedding generation.")
            raise # Re-raise for upstream orchestration to handle