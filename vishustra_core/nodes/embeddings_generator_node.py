import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node responsible for generating text embeddings.

    This node is designed to process string input data, simulating the creation
    of a high-dimensional vector representation (embedding). It includes robust
    input validation and error handling to ensure stable operation within the
    LLM orchestration framework. The actual embedding logic is simulated here;
    in a production environment, this would interface with a dedicated embedding
    model or service.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> List[float]:
        """
        Generates a simulated embedding vector for the provided text data.

        This method expects a string as its primary input (`data`). It consults
        the `context` dictionary for potential configuration parameters, such as
        the desired `embedding_dimension`.

        Args:
            data (Any): The input data to be embedded. This node specifically
                        expects a `str` type.
            context (Dict[str, Any]): A dictionary providing contextual information,
                                       which may include configuration settings
                                       like 'embedding_dimension' or flags for
                                       simulated errors.

        Returns:
            List[float]: A list of floats representing the generated embedding vector.
                         The length of this list corresponds to the embedding dimension.

        Raises:
            ValueError: If the input `data` is not a string, or if an issue
                        occurs during the simulated embedding generation process.
        """
        if not isinstance(data, str):
            error_msg = (f"EmbeddingsGeneratorNode received invalid input type. Expected 'str', "
                         f"but got '{type(data).__name__}'.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"EmbeddingsGeneratorNode initiated processing for data (first 50 chars): '{data[:50]}...'")

        # Determine the embedding dimension from context, defaulting to a common size
        embedding_dimension = context.get("embedding_dimension", 768)
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"Invalid 'embedding_dimension' '{embedding_dimension}' provided in context. "
                "Falling back to default dimension: 768."
            )
            embedding_dimension = 768

        try:
            # Simulate a failure condition if specified in the context for testing resilience
            if context.get("simulate_embedding_error", False):
                raise RuntimeError("Simulated external embedding service unavailability or error.")

            # --- Simulated Embedding Generation Logic ---
            # In a real-world scenario, this section would involve calling an
            # external embedding model (e.g., via an API or a local model inference).
            # For this simulation, we generate a deterministic vector based on input data.
            seed_value = sum(ord(char) for char in data) % 1000 / 1000.0
            embedding = [
                seed_value + (i * 0.00001 % 0.01) for i in range(embedding_dimension)
            ]
            # Ensure values are within a reasonable range for embeddings (e.g., -1 to 1 or 0 to 1)
            embedding = [(val - 0.5) * 2 for val in embedding]
            # --- End Simulated Embedding Generation Logic ---

            logger.debug(f"Successfully simulated embedding generation. Vector length: {len(embedding)}")
            return embedding

        except Exception as e:
            error_msg = (f"An unexpected error occurred during simulated embedding generation for "
                         f"data: '{data[:50]}...'. Error: {e}")
            logger.exception(error_msg)  # Log full traceback for critical issues
            raise ValueError(f"Embedding generation failed: {e}") from e