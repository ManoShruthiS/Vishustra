import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate vector embeddings for textual data.
    This node simulates the behavior of an embedding model, producing
    dense numerical representations of input text.
    """

    def __init__(self, model_name: str = "vishustra-simulated-embedder-v1", embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode with a specified (simulated) model
        and embedding dimension.

        Args:
            model_name (str): A descriptive identifier for the embedding model being
                              simulated. This helps in tracking which model configuration
                              is used.
            embedding_dimension (int): The desired dimension of the output embedding vectors.
                                       Common values are 384, 768, 1024, etc.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            logger.error("Attempted to initialize EmbeddingsGeneratorNode with an invalid model_name.")
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(f"Attempted to initialize EmbeddingsGeneratorNode with an invalid embedding_dimension: {embedding_dimension}")
            raise ValueError("embedding_dimension must be a positive integer.")

        self._model_name = model_name
        self._embedding_dimension = embedding_dimension
        logger.info(
            f"EmbeddingsGeneratorNode initialized. "
            f"Simulating model: '{self._model_name}', "
            f"output dimension: {self._embedding_dimension}."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self) -> List[float]:
        """
        Simulates the generation of a single embedding vector.
        In a real-world scenario, this would involve calling an external
        embedding service or a local model.
        """
        # Generate a list of random floats within a typical embedding range (-1.0 to 1.0)
        return [random.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input text or list of texts to generate corresponding embedding vectors.

        Args:
            data (Union[str, List[str]]): The input text (a single string) or a
                                          list of texts (list of strings) for which
                                          embeddings need to be generated.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. While not
                                       directly utilized for core embedding logic here,
                                       it's available for potential future extensions
                                       (e.g., batching hints, specific model overrides).

        Returns:
            Union[List[float], List[List[float]]]: If `data` was a single string, returns
                                                    a single list of floats representing its
                                                    embedding. If `data` was a list of strings,
                                                    returns a list of such embedding lists.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if any text item within a list is empty or invalid.
        """
        logger.debug(
            f"[{self.node_name}] Processing initiated for data type: {type(data)}. "
            f"Using simulated model '{self._model_name}'."
        )

        if isinstance(data, str):
            if not data.strip():
                logger.warning(f"[{self.node_name}] Received an empty string input. Cannot generate embedding.")
                raise ValueError("Input text for embedding generation cannot be empty.")
            logger.debug(f"[{self.node_name}] Generating embedding for single text input.")
            return self._generate_single_embedding()
        elif isinstance(data, list):
            if not data:
                logger.warning(f"[{self.node_name}] Received an empty list of texts. Returning an empty list of embeddings.")
                return []

            embeddings: List[List[float]] = []
            for i, text_item in enumerate(data):
                if not isinstance(text_item, str):
                    logger.error(
                        f"[{self.node_name}] Invalid element type at index {i} in batch. "
                        f"Expected string, got {type(text_item)}."
                    )
                    raise ValueError(f"All elements in the input list must be strings. Found non-string at index {i}.")
                if not text_item.strip():
                    logger.warning(
                        f"[{self.node_name}] Text item at index {i} is empty or whitespace-only. "
                        "Skipping this item as it cannot be embedded meaningfully."
                    )
                    # Depending on framework's error policy, one might:
                    # 1. Raise an error (current strict approach for a demo)
                    # 2. Skip and log, continuing with other items
                    # 3. Insert a 'null' or zero vector
                    raise ValueError(f"Text item at index {i} in the list cannot be empty.")

                embeddings.append(self._generate_single_embedding())

            logger.debug(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings for the batch input.")
            return embeddings
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data)}. "
                "Expected 'str' or 'List[str]'."
            )
            raise ValueError(
                f"Invalid input data type for EmbeddingsGeneratorNode. "
                f"Expected 'str' or 'List[str]', got '{type(data)}'."
            )