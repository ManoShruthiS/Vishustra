import logging
import hashlib
from typing import Any, Dict, List, Union

# Assume BaseNode is located in vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of text embeddings.

    This node takes a string or a list of strings as input and returns
    a simulated embedding vector or a list of embedding vectors.
    In a real-world scenario, this would interface with an external
    embedding model (e.g., OpenAI, HuggingFace, a local model server).

    Context Parameters:
    - 'embedding_dimension' (int, optional): The desired dimension of the
      generated embeddings. Defaults to 768. Must be a positive integer.
    - 'model_name' (str, optional): A placeholder for the specific embedding
      model to be used. Currently only influences logging output.
    """

    DEFAULT_EMBEDDING_DIMENSION = 768
    SIMULATED_MODEL_NAME = "simulated-text-embedding-v1"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGeneratorNode"

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of an embedding for a single piece of text.

        For demonstration purposes, this creates a simple, deterministic "embedding"
        based on the text's SHA256 hash and length. In a production system,
        this would involve calling an actual embedding model.
        """
        if not isinstance(text, str):
            # This should ideally be caught by the process method, but as a safeguard
            logger.error(f"Internal error: _generate_single_embedding received non-string input: {type(text).__name__}")
            raise TypeError(f"Expected text to be a string, but got {type(text).__name__}")

        if not text:
            logger.debug("Generating embedding for an empty string, returning zeros.")
            return [0.0] * dimension

        # Create a stable but simple "embedding" based on the text's hash
        text_hash_int = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        embedding = [
            (text_hash_int % (i + 1)) / (float(dimension) * 1000.0)
            for i in range(dimension)
        ]
        return embedding


    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (text or list of texts) to generate embeddings.

        Args:
            data (Union[str, List[str]]): The text or list of texts to embed.
                                          Each item in the list must be a string.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                      which may include 'embedding_dimension' or 'model_name'.

        Returns:
            Union[List[float], List[List[float]]]: The generated embedding vector
                                                    (for single text input) or a list of
                                                    embedding vectors (for batch input).

        Raises:
            ValueError: If `data` is None, or if 'embedding_dimension' in context is invalid.
            TypeError: If `data` is not a string or a list of strings, or if a list
                       contains non-string elements.
        """
        if data is None:
            logger.error("EmbeddingsGeneratorNode received None as data input.")
            raise ValueError("Data cannot be None for embeddings generation.")

        embedding_dimension = context.get('embedding_dimension', self.DEFAULT_EMBEDDING_DIMENSION)
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"Invalid 'embedding_dimension' in context: '{embedding_dimension}'. "
                f"Must be a positive integer. Using default: {self.DEFAULT_EMBEDDING_DIMENSION}."
            )
            # Optionally, could raise an error here to enforce strict context config
            embedding_dimension = self.DEFAULT_EMBEDDING_DIMENSION
            # Raise an error to strictly enforce context parameter validity
            # raise ValueError(f"Invalid 'embedding_dimension' in context: {embedding_dimension}. Must be a positive integer.")


        model_name = context.get('model_name', self.SIMULATED_MODEL_NAME)
        logger.info(
            f"EmbeddingsGeneratorNode processing data using model '{model_name}' "
            f"with simulated dimension {embedding_dimension}."
        )

        if isinstance(data, str):
            logger.debug(f"Generating embedding for single text input (length: {len(data)}).")
            return self._generate_single_embedding(data, embedding_dimension)
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                invalid_types = {type(item).__name__ for item in data if not isinstance(item, str)}
                logger.error(
                    f"List input for EmbeddingsGeneratorNode contains non-string elements. "
                    f"Found types: {invalid_types}. All list items must be strings."
                )
                raise TypeError(
                    f"All items in the input list must be strings. Found non-string types: {invalid_types}."
                )

            logger.debug(f"Generating embeddings for batch of {len(data)} texts.")
            embeddings = [self._generate_single_embedding(text, embedding_dimension) for text in data]
            return embeddings
        else:
            logger.error(
                f"Invalid data type for EmbeddingsGeneratorNode. Expected str or list[str], "
                f"but received {type(data).__name__}. Data: {data!r}"
            )
            raise TypeError(
                f"Invalid data type for EmbeddingsGeneratorNode. Expected 'str' or 'List[str]', "
                f"but received '{type(data).__name__}'."
            )