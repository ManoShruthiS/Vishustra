import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is correctly located in the project structure as specified
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating numerical embeddings
    for input text or a list of texts.

    This node simulates the transformation of human-readable text into dense
    vector representations, which are fundamental for various NLP tasks such as
    semantic search, text similarity, clustering, and feature extraction for ML models.
    For demonstration purposes, this implementation generates mock embeddings.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGenerator"

    def _generate_mock_embedding(self, text: str, dimension: int, mock_seed: Any = None) -> List[float]:
        """
        Generates a simulated embedding vector for a given text.
        This function is for mocking purposes and does not invoke an actual
        embedding model. It uses pseudo-random numbers.

        Args:
            text (str): The input text to generate an embedding for.
            dimension (int): The desired dimensionality of the embedding vector.
            mock_seed (Any, optional): An optional seed to ensure reproducible
                                       mock embeddings. If None, a seed derived
                                       from the text's hash is used.

        Returns:
            List[float]: A list of floats representing the mock embedding vector.
        """
        # Store the current state of the random number generator
        current_random_state = random.getstate()

        try:
            if mock_seed is not None:
                # Use the explicit mock_seed for global reproducibility if provided
                random.seed(mock_seed)
            else:
                # Use a seed derived from the text's hash for consistent embeddings per text
                # Modulo operation ensures the seed fits into common 32-bit integer ranges,
                # which can be relevant for cross-platform consistency in some contexts.
                random.seed(hash(text) % (2**32 - 1))

            # Generate random floats for the embedding vector within a typical range
            embedding = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
            return embedding
        finally:
            # Always restore the previous state of the random number generator
            random.setstate(current_random_state)

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        The `context` dictionary can optionally specify parameters for
        embedding generation:
        - `embedding_model_name` (str): A string identifying the simulated model.
                                        (Default: "simulated-bge-large-en-v1.5").
        - `embedding_dimension` (int): The target dimensionality for the
                                       embedding vectors. (Default: 768).
        - `mock_embedding_seed` (Any): An optional seed for the mock embedding
                                       generation, allowing for deterministic
                                       outputs for testing.

        Args:
            data (Any): The input data, expected to be a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing runtime configuration
                                      and metadata relevant for the node's operation.

        Returns:
            Union[List[float], List[List[float]]]: If `data` was a single string,
                                                  returns a `List[float]` (a single embedding).
                                                  If `data` was a list of strings,
                                                  returns a `List[List[float]]` (a list of embeddings).

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
            ValueError: If `embedding_dimension` specified in the context is invalid
                        (e.g., not a positive integer).
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}.")

        # Retrieve configuration from context with sensible defaults
        model_name = context.get("embedding_model_name", "simulated-bge-large-en-v1.5")
        dimension = context.get("embedding_dimension", 768)
        mock_seed = context.get("mock_embedding_seed")

        # Validate embedding_dimension
        if not isinstance(dimension, int) or dimension <= 0:
            logger.error(
                f"[{self.node_name}] Invalid 'embedding_dimension' in context: '{dimension}'. "
                "Must be a positive integer."
            )
            raise ValueError(f"Invalid 'embedding_dimension': '{dimension}'. Must be a positive integer.")

        if isinstance(data, str):
            logger.info(
                f"[{self.node_name}] Generating embedding for a single text "
                f"using simulated model: '{model_name}' (dimension: {dimension})."
            )
            return self._generate_mock_embedding(data, dimension, mock_seed)
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            logger.info(
                f"[{self.node_name}] Generating embeddings for {len(data)} texts "
                f"using simulated model: '{model_name}' (dimension: {dimension})."
            )
            embeddings = [self._generate_mock_embedding(text, dimension, mock_seed) for text in data]
            return embeddings
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str' or 'list[str]', "
                f"but received type: {type(data)}."
            )
            # Provide a truncated representation of the data for logging, if it's too large
            data_repr = str(data)[:200] + ("..." if len(str(data)) > 200 else "")
            raise TypeError(
                f"Invalid input data for '{self.node_name}'. Expected a string or a list of strings, "
                f"but received type: {type(data)}. Data sample: {data_repr}"
            )