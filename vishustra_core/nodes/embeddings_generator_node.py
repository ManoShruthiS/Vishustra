import logging
from typing import Any, Dict, List, Union

# Assuming the BaseNode class is located here within the framework structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for simulating the generation of
    text embeddings.

    This node takes a single string or a list of strings as input and produces
    a corresponding embedding vector (a list of floats) or a list of such vectors.
    The embedding generation is simulated for demonstration purposes, producing
    deterministic vectors based on the input text.
    """

    # Define a common embedding dimension for simulation
    _EMBEDDING_DIMENSION = 128

    def __init__(self):
        """
        Initializes the EmbeddingsGeneratorNode.
        """
        logger.debug(f"Initializing '{self.node_name}' node.")

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this node.
        """
        return "embeddings_generator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        This method is a placeholder for actual embedding model inference.

        Args:
            text: The input string for which to generate an embedding.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            TypeError: If the input 'text' is not a string.
        """
        if not isinstance(text, str):
            logger.error(
                f"[{self.node_name}] Expected text to be a string, but got type {type(text)}."
            )
            raise TypeError(f"Invalid input type: Expected str, got {type(text)}")

        if not text:
            logger.warning(
                f"[{self.node_name}] Attempted to generate embedding for an empty string. "
                "Returning a zero vector of dimension {self._EMBEDDING_DIMENSION}."
            )
            return [0.0] * self._EMBEDDING_DIMENSION

        # Simple, deterministic simulation of an embedding vector.
        # In a real-world scenario, this would involve calling an actual
        # embedding model (e.g., OpenAI, HuggingFace, local model).
        text_seed = sum(ord(char) for char in text) % 1_000_000 # Generate a numeric seed from text
        embedding = [
            ((text_seed + i * 7) % 1000 / 999.0 - 0.5) # Scale to roughly -0.5 to 0.5
            for i in range(self._EMBEDDING_DIMENSION)
        ]
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        Args:
            data: The input text(s) to embed. Can be a single string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     This node does not directly use the context, but it's passed
                     through the BaseNode interface.

        Returns:
            If 'data' was a single string, returns a list of floats representing
            its embedding vector.
            If 'data' was a list of strings, returns a list of lists of floats,
            where each inner list is the embedding for the corresponding input string.

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings,
                       or if any element within a list of data is not a string.
            Exception: Propagates any exceptions encountered during single embedding
                       generation for robust error handling.
        """
        logger.debug(
            f"[{self.node_name}] Initiating embedding generation. "
            f"Input data type: {type(data)}. Context keys: {list(context.keys())}"
        )

        if isinstance(data, str):
            logger.info(f"[{self.node_name}] Generating embedding for a single string input.")
            return self._generate_single_embedding(data)
        elif isinstance(data, list):
            # Validate all elements in the list are strings
            if not all(isinstance(item, str) for item in data):
                first_non_str = next((item for item in data if not isinstance(item, str)), None)
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. "
                    f"First invalid item: '{first_non_str}' (type: {type(first_non_str)})."
                )
                raise TypeError("Input list for embedding generation must contain only strings.")
            
            logger.info(f"[{self.node_name}] Generating embeddings for a list of {len(data)} strings.")
            
            results = []
            for i, text in enumerate(data):
                try:
                    results.append(self._generate_single_embedding(text))
                except Exception as e:
                    # Log the specific error and re-raise to ensure upstream nodes are aware.
                    logger.critical(
                        f"[{self.node_name}] Failed to generate embedding for item {i} "
                        f"(text: '{text[:100]}...') due to an error: {e}", exc_info=True
                    )
                    raise # Re-raise the exception after logging for robust error propagation
            return results
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: "
                f"Expected 'str' or 'List[str]', got '{type(data)}'."
            )
            raise TypeError(
                f"Invalid input data type: Expected 'str' or 'List[str]', got '{type(data)}'."
            )

