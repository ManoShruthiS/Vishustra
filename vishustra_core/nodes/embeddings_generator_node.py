import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node designed to simulate the generation of embeddings
    from input text data.

    This node expects its 'data' input to be either a single string or a list of strings.
    It produces a single embedding (a list of floats) if the input is a string,
    or a list of embeddings (a list of lists of floats) if the input is a list of strings.

    The actual embedding generation is simulated using random float values to represent
    dummy embeddings. In a production deployment, this node would integrate with
    a genuine embedding model service or library (e.g., via an API call to a cloud
    provider or by utilizing a local machine learning model).
    """

    # A fixed dimension for the simulated embeddings. In a real-world scenario,
    # this might be configurable via the node's initialization or the 'context' dictionary.
    _EMBEDDING_DIMENSION = 768 

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "EmbeddingsGenerator"

    def _generate_dummy_embedding(self, text_input: str) -> List[float]:
        """
        Simulates the generation of an embedding for a given text input.
        This private helper method abstracts the embedding logic, which would
        be replaced by a call to an actual embedding model in a real system.

        Args:
            text_input: The string of text to generate an embedding for.

        Returns:
            A list of floats representing the generated (dummy) embedding.
        """
        # Basic validation for the text input, even for simulation purposes.
        if not isinstance(text_input, str) or not text_input.strip():
            logger.warning(
                f"Attempted to generate dummy embedding for non-string or empty input. "
                f"Returning an embedding of zeros. Input type: {type(text_input).__name__}"
            )
            return [0.0] * self._EMBEDDING_DIMENSION
            
        # Simulate an embedding by generating a list of random float values within a typical range.
        embedding = [random.uniform(-1.0, 1.0) for _ in range(self._EMBEDDING_DIMENSION)]
        logger.debug(
            f"Generated dummy embedding of dimension {len(embedding)} "
            f"for a text snippet (first 30 chars: '{text_input[:30]}...')"
        )
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings. The method handles both
        single string inputs and lists of strings.

        The 'context' dictionary is provided for potential future extensions,
        allowing for dynamic configuration such as specifying a particular
        embedding model, API keys, or custom embedding dimensions. This
        simulated implementation does not currently utilize the 'context'
        for these purposes, relying on internal defaults.

        Args:
            data: The input text data to be embedded. Expected types are `str`
                  for a single text, or `List[str]` for multiple texts.
            context: A dictionary containing operational context or configuration
                     parameters for the node.

        Returns:
            A `List[float]` if the input 'data' was a single string, or a
            `List[List[float]]` if the input 'data' was a list of strings.
            Each inner list represents an embedding.

        Raises:
            ValueError: If the input 'data' is not a string or a list of strings,
                        or if a list input contains non-string elements.
            RuntimeError: For any other unexpected errors encountered during
                          the embedding generation process.
        """
        logger.info(f"Node '{self.node_name}' initiated processing for embedding generation.")
        
        try:
            if isinstance(data, str):
                logger.debug(f"Processing single string input of length {len(data)} for embedding.")
                return self._generate_dummy_embedding(data)
            elif isinstance(data, list):
                # Validate that all elements in the list are strings.
                if not all(isinstance(item, str) for item in data):
                    error_msg = (
                        f"List input for EmbeddingsGeneratorNode must contain only strings. "
                        f"Detected non-string elements."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.debug(f"Processing a list of {len(data)} strings for embeddings.")
                return [self._generate_dummy_embedding(item) for item in data]
            else:
                error_msg = (
                    f"Invalid data type for EmbeddingsGeneratorNode. "
                    f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        except ValueError as ve:
            # Re-raise specific ValueErrors directly as they indicate invalid input.
            raise ve
        except Exception as e:
            # Catch any other unforeseen issues during the embedding process.
            error_msg = f"An unexpected error occurred during embedding generation in '{self.node_name}': {e}"
            logger.critical(error_msg, exc_info=True)
            # Wrap unexpected exceptions in a RuntimeError for consistency.
            raise RuntimeError(error_msg) from e
