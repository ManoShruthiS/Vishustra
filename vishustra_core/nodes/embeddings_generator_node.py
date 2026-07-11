import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating numerical embeddings
    from input text data.

    This node converts text (or a list of texts) into fixed-size numerical vectors,
    which can capture semantic meaning. The dimensionality of the generated
    embeddings can be configured via the 'embedding_dimensions' key in the
    processing context.

    In a production environment, this node would interface with a real embedding
    model (e.g., from an LLM provider, a local model, or a vector database
    integration). For demonstration, it simulates embedding generation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGeneratorNode"

    def _generate_single_embedding(self, text: str, dimensions: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a real implementation, this method would call an actual embedding model.
        """
        # Simulate an embedding operation by generating a list of random floats
        # within a typical embedding range.
        logger.debug(
            f"Simulating {dimensions}-dimensional embedding generation for text snippet: "
            f"'{text[:100]}{'...' if len(text) > 100 else ''}'"
        )
        return [random.uniform(-1.0, 1.0) for _ in range(dimensions)]

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        The `data` input can be:
        - `str`: A single text string to be embedded.
        - `List[str]`: A list of text strings, where each string will be embedded.

        The `context` dictionary can optionally contain:
        - `embedding_dimensions` (int): The desired dimensionality for the output
          embedding vectors. If not provided or invalid, a default of 768 is used.

        Returns:
            Union[List[float], List[List[float]]]:
            - If `data` was a `str`, returns a single `List[float]` representing its embedding.
            - If `data` was a `List[str]`, returns `List[List[float]]`, where each
              inner list is an embedding for the corresponding input string.
            - Returns an empty list or a zero vector for problematic inputs,
              depending on the specific input type and issue.

        Raises:
            ValueError: If the input `data` is not a `str` or `List[str]`.
            RuntimeError: If an unexpected issue prevents the successful (simulated)
                          embedding generation.
        """
        if not isinstance(data, (str, list)):
            error_msg = (
                f"Invalid input data type for '{self.node_name}'. Expected 'str' or 'List[str]', "
                f"but received '{type(data).__name__}'. Data: {data!r}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Retrieve embedding dimensions from context, default if not provided or invalid
        embedding_dimensions = context.get('embedding_dimensions', 768)
        if not isinstance(embedding_dimensions, int) or embedding_dimensions <= 0:
            logger.warning(
                f"Invalid or missing 'embedding_dimensions' in context for '{self.node_name}'. "
                f"Received '{embedding_dimensions}'. Defaulting to 768 dimensions."
            )
            embedding_dimensions = 768

        try:
            if isinstance(data, str):
                if not data.strip():
                    logger.warning(
                        f"Received an empty or whitespace-only string in '{self.node_name}'. "
                        f"Returning a zero-vector of {embedding_dimensions} dimensions."
                    )
                    return [0.0] * embedding_dimensions
                
                logger.info(f"Generating a single embedding for text using '{self.node_name}'.")
                return self._generate_single_embedding(data, embedding_dimensions)

            elif isinstance(data, list):
                if not data:
                    logger.warning(
                        f"Received an empty list of texts in '{self.node_name}'. Returning an empty list of embeddings."
                    )
                    return []
                
                generated_embeddings: List[List[float]] = []
                for i, item in enumerate(data):
                    if not isinstance(item, str):
                        logger.warning(
                            f"Skipping non-string item at index {i} in list for '{self.node_name}'. "
                            f"Expected 'str', got '{type(item).__name__}'. Appending a zero-vector placeholder."
                        )
                        generated_embeddings.append([0.0] * embedding_dimensions)
                        continue
                    
                    if not item.strip():
                        logger.warning(
                            f"Skipping empty or whitespace-only string item at index {i} in '{self.node_name}'. "
                            f"Appending a zero-vector placeholder."
                        )
                        generated_embeddings.append([0.0] * embedding_dimensions)
                        continue
                    
                    try:
                        generated_embeddings.append(self._generate_single_embedding(item, embedding_dimensions))
                    except Exception as e:
                        logger.error(
                            f"Failed to generate embedding for item at index {i} in '{self.node_name}': {e}",
                            exc_info=True
                        )
                        # On individual item failure, append a zero-vector to maintain list length
                        generated_embeddings.append([0.0] * embedding_dimensions)
                
                logger.info(
                    f"Generated {len(generated_embeddings)} embeddings (out of {len(data)} inputs) "
                    f"from a list of texts using '{self.node_name}'."
                )
                return generated_embeddings

        except Exception as e:
            # Catch any unexpected errors during the process and re-raise as a RuntimeError
            logger.exception(f"An unexpected error occurred during embedding generation in '{self.node_name}'.")
            raise RuntimeError(f"Failed to generate embeddings in '{self.node_name}': {e}") from e

