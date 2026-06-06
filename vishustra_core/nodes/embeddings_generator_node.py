import logging
import random
from typing import Any, Dict, List, Union

# Assuming vishustra_core is installed and available in the environment
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of text embeddings.

    This node takes a string or a list of strings as input and
    produces a list of simulated embedding vectors. It allows
    configuration of the embedding dimension during initialization.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension: The desired dimension of the simulated embedding vectors.
                                 Defaults to 768, a common dimension for many embedding models.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"Invalid embedding_dimension provided: {embedding_dimension}. Must be a positive integer.")
            raise ValueError("embedding_dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.info(
            f"EmbeddingsGeneratorNode initialized with embedding_dimension={self._embedding_dimension}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a real-world scenario, this would involve calling a specific
        embedding model (e.g., from Hugging Face, OpenAI, Cohere, etc.).

        For demonstration, a deterministic 'random' vector is generated based on the text's hash.
        This approach ensures consistency for identical texts within the same process execution,
        without affecting the global random state.
        """
        # Save the current state of the random number generator
        original_random_state = random.getstate()
        try:
            # Seed the random generator deterministically for the given text
            # This makes the "embedding" consistent for the same input text
            random.seed(hash(text))
            embedding = [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]
            return embedding
        finally:
            # Restore the original state of the random number generator
            random.setstate(original_random_state)

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        The node expects input `data` to be either a single string or a list of strings.
        It generates a simulated embedding vector for each string.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing shared context or configuration
                     for the processing pipeline. While available, this node
                     does not actively use the context for its core embedding
                     generation logic in this simulated implementation.

        Returns:
            If input `data` is a single string, returns a single embedding (List[float]).
            If input `data` is a list of strings, returns a list of embeddings (List[List[float]]).

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings,
                       or if a list contains non-string elements.
            Exception: For any unexpected errors during the simulated embedding generation.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data).__name__}")

        texts_to_process: List[str]
        is_single_string_input = False

        if isinstance(data, str):
            texts_to_process = [data]
            is_single_string_input = True
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                non_string_types = {type(item).__name__ for item in data if not isinstance(item, str)}
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. "
                    f"Expected list of strings, but found types: {', '.join(non_string_types)}."
                )
                raise TypeError(
                    f"Input 'data' must be a string or a list of strings. "
                    f"Received list containing non-string elements (e.g., {non_string_types.pop() if non_string_types else 'unknown type'})."
                )
            texts_to_process = data
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data).__name__}. "
                "Expected str or List[str]."
            )
            raise TypeError(
                f"Input 'data' must be a string or a list of strings. "
                f"Received type: {type(data).__name__}"
            )

        try:
            embeddings_list: List[List[float]] = []
            for text in texts_to_process:
                embedding = self._generate_single_embedding(text)
                embeddings_list.append(embedding)

            logger.info(f"[{self.node_name}] Successfully generated {len(embeddings_list)} embedding(s).")
            logger.debug(f"[{self.node_name}] Context received keys: {list(context.keys()) if context else 'None'}")

            if is_single_string_input:
                # If the original input was a single string, return just the single embedding vector.
                return embeddings_list[0]
            else:
                # If the original input was a list of strings, return the list of embedding vectors.
                return embeddings_list

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during embedding generation for input data."
            )
            # Re-raise the exception to allow upstream nodes or the orchestrator to handle it.
            raise
