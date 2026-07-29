import logging
import random
from typing import Any, Dict, List, Union

# Assume vishustra_core.nodes.base_node exists at the specified path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates generating embeddings for input text(s).

    This node takes text (or a list of texts) and returns a simulated embedding
    vector (or a list of vectors). In a real-world scenario, this would
    interface with an actual embeddings model (e.g., OpenAI, HuggingFace, local model).
    """

    def __init__(self, model_name: str = "simulated-embedding-model", embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            model_name (str): The name of the (simulated) embeddings model to use.
                              This could be used for dynamic model loading in a real scenario.
            embedding_dimension (int): The desired dimension of the output embedding vectors.
                                       Must be a positive integer.
        Raises:
            ValueError: If `embedding_dimension` is not a positive integer.
        """
        self._model_name = model_name
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"Invalid embedding_dimension: {embedding_dimension}. Must be a positive integer.")
            raise ValueError("embedding_dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.info(f"EmbeddingsGeneratorNode initialized with model: '{self._model_name}', "
                    f"dimension: {self._embedding_dimension}.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates generating a single embedding vector for a given text.
        In a real scenario, this would call an external embeddings API or model.
        """
        # Simulate a fixed-size float vector for the given dimension
        # Use a consistent seed for reproducibility in testing, or truly random for production simulation
        # For simplicity, using random.uniform for values between -1.0 and 1.0
        return [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for the input text or list of texts.

        Args:
            data (Union[str, List[str]]): The text or list of texts to embed.
                                          Each text should be a string.
            context (Dict[str, Any]): A dictionary containing additional runtime context.
                                      This could potentially override model parameters
                                      or provide API keys in a real implementation.

        Returns:
            Union[List[float], List[List[float]]]: A single embedding vector (List[float])
                                                    if `data` was a string, or a list of
                                                    embedding vectors (List[List[float]])
                                                    if `data` was a list of strings.
                                                    Each embedding vector will have the
                                                    dimension specified at initialization.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If an unexpected error occurs during embedding generation (simulated).
        """
        logger.debug(f"[{self.node_name}] Starting processing. Input data type: {type(data)}.")

        if not isinstance(data, (str, list)):
            error_msg = f"Invalid input data type. Expected str or List[str], but got {type(data)}."
            logger.error(f"[{self.node_name}] {error_msg}")
            raise TypeError(error_msg)

        try:
            if isinstance(data, str):
                logger.debug(f"[{self.node_name}] Generating embedding for a single text input.")
                embedding = self._generate_single_embedding(data)
                logger.info(f"[{self.node_name}] Successfully generated single embedding (dimension: {len(embedding)}).")
                return embedding
            elif isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    error_msg = "All elements in the input list 'data' must be strings."
                    logger.error(f"[{self.node_name}] {error_msg}")
                    raise TypeError(error_msg)

                logger.debug(f"[{self.node_name}] Generating embeddings for a list of {len(data)} texts.")
                embeddings = [self._generate_single_embedding(text) for text in data]
                logger.info(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings "
                            f"(each with dimension: {self._embedding_dimension}).")
                return embeddings
        except TypeError as e:
            # Re-raise TypeErrors that are specific to input data validation
            raise e
        except Exception as e:
            # Catch any other unexpected errors during the simulated generation process
            logger.exception(f"[{self.node_name}] An unexpected error occurred during embedding generation.")
            raise ValueError(f"Failed to generate embeddings: {e}") from e