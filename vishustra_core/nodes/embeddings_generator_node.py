import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is located in the specified path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that generates numerical embeddings for text data.

    This node simulates the process of converting input text (or a list of texts)
    into fixed-size numerical vectors (embeddings), which can be used for
    semantic search, clustering, or as input to other machine learning models.

    It supports configurable embedding dimensions and model names, with the
    ability to override these settings via the processing context.
    """

    _DEFAULT_EMBEDDING_DIMENSION: int = 768
    _DEFAULT_MODEL_NAME: str = "simulated-text-embedding-v1"

    def __init__(self, model_name: str = None, embedding_dimension: int = None):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            model_name (str, optional): The name of the embedding model to simulate.
                                        Defaults to "simulated-text-embedding-v1".
            embedding_dimension (int, optional): The dimension of the generated embeddings.
                                                 Defaults to 768.
        """
        self._model_name = model_name if model_name is not None else self._DEFAULT_MODEL_NAME
        self._embedding_dimension = embedding_dimension if embedding_dimension is not None else self._DEFAULT_EMBEDDING_DIMENSION

        if not isinstance(self._embedding_dimension, int) or self._embedding_dimension <= 0:
            logger.error(
                f"Initialization failed: Invalid default embedding_dimension provided: "
                f"{self._embedding_dimension}. Must be a positive integer."
            )
            raise ValueError(
                f"Invalid initial embedding_dimension: {self._embedding_dimension}. "
                "Must be a positive integer."
            )

        logger.info(
            f"EmbeddingsGeneratorNode initialized with model: '{self._model_name}' "
            f"and default dimension: {self._embedding_dimension}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates generating a single embedding for a given text.
        In a real-world scenario, this method would interface with an actual
        embedding service or model (e.g., OpenAI, HuggingFace, local model).
        """
        # For simulation, we return a list of random floats.
        # The specific values or distribution might be refined for different model simulations.
        return [random.uniform(-1.0, 1.0) for _ in range(dimension)]

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        This method expects `data` to be either a single string or a list of strings.
        It generates a numerical vector (embedding) for each valid string.

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing context-specific information.
                                      Can optionally override 'embedding_model_name'
                                      and 'embedding_dimension' for this specific processing call.

        Returns:
            Union[List[float], List[List[float]]]:
                A list of floats for a single string input, or a list of lists
                of floats for a list of strings input. Each inner list represents
                an embedding vector.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        if any string in the list is not valid, or if the
                        embedding dimension specified is invalid.
        """
        logger.debug(f"Entering EmbeddingsGeneratorNode.process with data type: {type(data)}")

        # Allow context to dynamically override configured parameters for this call
        current_model_name = context.get('embedding_model_name', self._model_name)
        current_dimension = context.get('embedding_dimension', self._embedding_dimension)

        if not isinstance(current_dimension, int) or current_dimension <= 0:
            logger.error(
                f"Processing failed: Invalid 'embedding_dimension' received from context "
                f"or node configuration: {current_dimension}. Must be a positive integer."
            )
            raise ValueError(
                f"Invalid embedding_dimension: {current_dimension}. "
                "Must be a positive integer."
            )

        if isinstance(data, str):
            if not data.strip():
                logger.warning(
                    f"Received an empty or whitespace-only string for embedding "
                    f"using model '{current_model_name}'. Returning a zero vector."
                )
                return [0.0] * current_dimension
            logger.info(
                f"Generating embedding for a single text using model: '{current_model_name}' "
                f"with dimension: {current_dimension}"
            )
            return self._generate_single_embedding(data, current_dimension)

        elif isinstance(data, list):
            if not data:
                logger.warning(
                    f"Received an empty list of texts for embedding using model "
                    f"'{current_model_name}'. Returning an empty list of embeddings."
                )
                return []

            # Validate all items in the list are strings
            if not all(isinstance(item, str) for item in data):
                invalid_items = [item for item in data if not isinstance(item, str)]
                error_msg = (
                    f"Input list contains non-string elements. First 5 invalid types: "
                    f"{[type(item).__name__ for item in invalid_items[:5]]}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(
                f"Generating embeddings for {len(data)} texts using model: '{current_model_name}' "
                f"with dimension: {current_dimension}"
            )
            embeddings = []
            for i, text in enumerate(data):
                if not text.strip():
                    logger.warning(
                        f"Skipping empty or whitespace-only string at index {i} "
                        f"in the input list. Appending a zero vector."
                    )
                    embeddings.append([0.0] * current_dimension)
                else:
                    embeddings.append(self._generate_single_embedding(text, current_dimension))
            return embeddings

        else:
            error_msg = (
                f"Invalid input data type: {type(data).__name__}. "
                "Expected 'str' or 'List[str]' for embedding generation."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)