import logging
import hashlib
import random
from typing import Any, Dict, List, Union

# Assuming vishustra_core is a package and nodes is a subpackage
# and base_node.py contains BaseNode.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of text embeddings.

    This node takes text (or a list of texts) as input and returns
    a simulated embedding vector (or a list of vectors). The simulation
    is deterministic based on the input text to ensure consistency.
    """

    _EMBEDDING_DIMENSION = 768  # A common embedding dimension for simulation purposes

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        The generation is deterministic based on the text content to ensure
        reproducible results for the same input.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the simulated embedding vector.

        Raises:
            Exception: If an unexpected error occurs during the simulation process.
        """
        try:
            # Create a deterministic seed from the text content
            seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
            rng = random.Random(seed)

            # Generate a vector of floats within a common range [-1.0, 1.0]
            embedding = [rng.uniform(-1.0, 1.0) for _ in range(self._EMBEDDING_DIMENSION)]
            return embedding
        except Exception as e:
            logger.error(
                f"[{self.node_name}] Failed to simulate embedding for text (first 50 chars): '{text[:50]}...': {e}",
                exc_info=True
            )
            raise

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings, either for a single text
        or a batch of texts.

        Args:
            data: The input data, expected to be a string (single text) or
                  a list of strings (multiple texts).
            context: A dictionary containing contextual information.
                     Expected keys:
                     - 'embedding_model_name' (str, optional): The name of the
                       embedding model to simulate (e.g., "text-embedding-ada-002").
                       Used for logging purposes in this simulation.

        Returns:
            A list of floats if a single string was provided as input, or a list of
            lists of floats if a list of strings was provided. Each inner list
            represents an embedding vector.

        Raises:
            ValueError: If the input data is not a string or a list of strings.
            Exception: For any issues encountered during the embedding generation simulation
                       for one or more items.
        """
        model_name = context.get('embedding_model_name', 'simulated-embedding-model')
        logger.debug(f"[{self.node_name}] Starting embedding generation using model: '{model_name}'")

        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing a single text for embedding.")
            return self._generate_single_embedding(data)
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            logger.debug(f"[{self.node_name}] Processing a batch of {len(data)} texts for embeddings.")
            embeddings = []
            for i, text in enumerate(data):
                try:
                    embeddings.append(self._generate_single_embedding(text))
                except Exception as e:
                    # Log the specific failure for an item and re-raise to indicate batch processing failure.
                    # Depending on system requirements, one might choose to skip the item or return a placeholder.
                    # For robustness, failing the entire batch on an item error is a common strategy to prevent
                    # incomplete or misleading results.
                    logger.error(
                        f"[{self.node_name}] Error processing text at index {i} in batch: {e}",
                        exc_info=True
                    )
                    raise # Re-raise the exception to indicate a problem with the batch.
            return embeddings
        else:
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str' or 'List[str]', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)