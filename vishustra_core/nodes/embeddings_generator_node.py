import logging
import random
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure
# This import path is specified by the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of vector embeddings from text data.
    
    This node expects either a single string or a list of strings as input data.
    It processes the input to generate a corresponding vector embedding (or list of embeddings).
    
    In a production environment, this node would interface with a concrete embedding model
    (e.g., from OpenAI, HuggingFace, local models, etc.). For this implementation, it
    generates dummy float vectors to represent the embedding output.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension (int): The desired dimension for the simulated embeddings.
                                       Defaults to 768, a common embedding size.
                                       Must be a positive integer.
        Raises:
            ValueError: If `embedding_dimension` is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError(
                f"Invalid embedding_dimension '{embedding_dimension}'. "
                "Must be a positive integer."
            )
        self._embedding_dimension = embedding_dimension
        logger.info(f"[{self.node_name}] Initialized with embedding dimension: {self._embedding_dimension}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """
        Generates a dummy embedding vector for a given text.
        This method simulates the output of an actual embedding model.

        Args:
            text (str): The input text for which to generate an embedding.

        Returns:
            List[float]: A list of floats representing the simulated embedding vector.
        """
        # For simulation, we create a deterministic-ish dummy vector based on text content.
        # In a real scenario, this would involve calling an external or local embedding model.
        seed_value = sum(ord(c) for c in text) % 10000 + len(text)
        random.seed(seed_value)
        
        # Generate floats within a typical embedding range, e.g., -1.0 to 1.0
        return [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate text embeddings.

        This method expects `data` to be either a single string or a list of strings.
        It generates a dummy embedding vector for each string provided.

        Args:
            data (Any): The input data. Expected types are `str` (for a single text)
                        or `List[str]` (for multiple texts).
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       While not directly used by this simulated generator,
                                       it's available for passing model configuration
                                       or other runtime parameters in a real implementation.

        Returns:
            Union[List[float], List[List[float]]]:
                - If `data` was a single `str`, returns a `List[float]` (the embedding vector).
                - If `data` was a `List[str]`, returns a `List[List[float]]` (a list of embedding vectors).

        Raises:
            TypeError: If the input `data` is not a `str` or a `List[str]`.
            Exception: Catches and re-raises any other unexpected errors during processing.
        """
        logger.debug(f"[{self.node_name}] Processing initiated for data of type: {type(data)}")

        try:
            if isinstance(data, str):
                logger.info(f"[{self.node_name}] Generating embedding for a single text input.")
                embedding = self._generate_dummy_embedding(data)
                logger.debug(f"[{self.node_name}] Successfully generated single embedding of dimension {len(embedding)}.")
                return embedding
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                logger.info(f"[{self.node_name}] Generating embeddings for a list of {len(data)} text inputs.")
                embeddings = [self._generate_dummy_embedding(text) for text in data]
                if embeddings:
                    logger.debug(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings, each of dimension {len(embeddings[0])}.")
                else:
                    logger.debug(f"[{self.node_name}] Input list was empty; returned an empty list of embeddings.")
                return embeddings
            else:
                error_message = (
                    f"[{self.node_name}] Invalid input data type. "
                    f"Expected 'str' or 'List[str]', but received '{type(data)}'."
                )
                logger.error(error_message)
                raise TypeError(error_message)
        except Exception as e:
            # Catching a broad exception to ensure all processing errors are logged
            # and re-raised for upstream handling.
            error_message = f"[{self.node_name}] An unexpected error occurred during embedding generation: {e}"
            logger.exception(error_message) # Logs the full traceback
            raise # Re-raise the exception after logging
