import logging
import random
import hashlib
from typing import Any, Dict, List, Union

# Assume vishustra_core exists and BaseNode is available at this path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate vector embeddings for input text.
    This node simulates the interaction with an external embedding model,
    converting textual data into fixed-size numerical vectors suitable for
    downstream machine learning tasks or similarity computations.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "Embeddings Generator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embeddings.

        This method expects `data` to be either a single string or a list of strings.
        It uses parameters from the `context` dictionary, such as `embedding_dimension`,
        to control the output vector size.

        Args:
            data: The input text (str) or a list of texts (List[str]) to be embedded.
            context: A dictionary containing runtime context and configuration parameters.
                     Expected keys:
                     - `embedding_dimension` (int, optional): The desired dimension
                       of the output embedding vector. Defaults to 768 if not provided
                       or invalid.

        Returns:
            - If `data` is a `str`: A `List[float]` representing the embedding vector.
            - If `data` is a `List[str]`: A `List[List[float]]` where each inner list
              is the embedding vector for the corresponding input text.

        Raises:
            ValueError: If the input `data` type is unsupported (not str or List[str]).
            RuntimeError: If an unexpected error occurs during the embedding generation process.
        """
        embedding_dimension = context.get("embedding_dimension", 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"Invalid 'embedding_dimension' found in context: '{embedding_dimension}'. "
                "It must be a positive integer. Defaulting to 768."
            )
            embedding_dimension = 768

        try:
            if isinstance(data, str):
                logger.info(f"Generating embedding for a single text input (length: {len(data)} chars).")
                return self._generate_single_embedding(data, embedding_dimension)
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                logger.info(f"Generating embeddings for a list of {len(data)} text inputs.")
                results = [
                    self._generate_single_embedding(text, embedding_dimension)
                    for text in data
                ]
                return results
            else:
                error_msg = (
                    f"Unsupported data type for EmbeddingsGeneratorNode. "
                    f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            # Catching a broad exception to ensure robustness against various internal failures
            error_msg = f"An unexpected error occurred during embedding generation: {e}"
            logger.exception(error_msg)  # Log the full traceback for debugging
            raise RuntimeError(error_msg) from e

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a production environment, this method would integrate with an actual
        embedding model API (e.g., OpenAI, HuggingFace, local model).

        For this simulation, it generates a pseudo-random vector based on the
        input text's hash to provide a deterministic yet varying output.
        """
        if not text:
            logger.warning("Received empty text for embedding. Returning a zero vector of specified dimension.")
            return [0.0] * dimension

        # Generate a seed from the text's SHA256 hash for deterministic pseudo-random numbers
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        rng = random.Random(seed)

        # Generate a list of floats within a typical embedding range (-1.0 to 1.0)
        # for a more realistic simulation than just zeros.
        embedding = [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dimension)]
        logger.debug(f"Generated simulated embedding for text (first 20 chars): '{text[:20]}...'")
        return embedding
