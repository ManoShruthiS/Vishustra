import logging
import random
from typing import Any, Dict, List, Union

# Assuming the vishustra_core.nodes.base_node module exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node designed to generate vector embeddings for text inputs.

    This node accepts either a single string or a list of strings as input `data`.
    It simulates the process of converting these texts into fixed-size numerical vectors
    (embeddings), which are crucial for many LLM-related tasks like semantic search,
    clustering, or retrieval-augmented generation.

    The dimensions of the generated embeddings can be specified via the `context` dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "EmbeddingsGenerator"

    def _generate_dummy_embedding(self, text: str, dimensions: int) -> List[float]:
        """
        Simulates the generation of an embedding for a given text.
        In a real-world production scenario, this method would interact with an
        actual embedding model API (e.g., OpenAI, Hugging Face, custom model).

        For this simulation, it generates a list of random floats representing the embedding.
        If the text is empty or only whitespace, it returns a zero vector of the specified dimensions.

        Args:
            text (str): The text content for which to generate an embedding.
            dimensions (int): The desired dimensionality of the output embedding vector.

        Returns:
            List[float]: A list of floats representing the generated embedding.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning(
                f"[{self.node_name}] Attempted to generate dummy embedding for an empty or "
                f"whitespace-only input string. Returning a zero vector of {dimensions} dimensions."
            )
            return [0.0] * dimensions

        # Simulate embedding generation by returning a list of random floats.
        # Values typically range from -1.0 to 1.0 or similar.
        embedding = [random.uniform(-1.0, 1.0) for _ in range(dimensions)]
        logger.debug(
            f"[{self.node_name}] Generated dummy embedding of {dimensions} dimensions "
            f"for text snippet: '{text[:50]}{'...' if len(text) > 50 else ''}'"
        )
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[List[float]], List[float]]:
        """
        Processes the input data to generate text embeddings.

        This method expects the `data` to be either a single string or a list of strings.
        It retrieves embedding generation parameters from the `context`, such as
        `embedding_dimensions`.

        Args:
            data (Any): The input data, expected to be a `str` (single text)
                        or `List[str]` (multiple texts).
            context (Dict[str, Any]): A dictionary containing runtime parameters for the node.
                                       Expected keys:
                                       - 'embedding_dimensions' (int, optional): The desired
                                         dimensionality for the output embeddings. Defaults to 768.

        Returns:
            Union[List[List[float]], List[float]]:
                - If `data` was a `str`, returns a single `List[float]` (the embedding).
                - If `data` was a `List[str]`, returns a `List[List[float]]` (list of embeddings).

        Raises:
            TypeError: If the input `data` is not a `str` or `List[str]`.
            ValueError: If an invalid `embedding_dimensions` is provided in the context.
            Exception: Propagates any unexpected errors during the embedding generation process.
        """
        logger.info(f"[{self.node_name}] Starting embedding generation process.")

        # Determine embedding dimensions from context, with a robust default
        dimensions = context.get('embedding_dimensions', 768)
        if not isinstance(dimensions, int) or dimensions <= 0:
            error_msg = (
                f"[{self.node_name}] Invalid or non-positive 'embedding_dimensions' "
                f"specified in context ({dimensions}). Must be a positive integer."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            if isinstance(data, str):
                logger.debug(f"[{self.node_name}] Processing a single text input.")
                return self._generate_dummy_embedding(data, dimensions)
            elif isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    error_msg = (
                        f"[{self.node_name}] Input data is a list, but not all elements are strings. "
                        f"Found types: {[type(item).__name__ for item in data if not isinstance(item, str)]}"
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)

                logger.debug(f"[{self.node_name}] Processing a list of {len(data)} text inputs.")
                embeddings: List[List[float]] = []
                for i, text_item in enumerate(data):
                    try:
                        embedding = self._generate_dummy_embedding(text_item, dimensions)
                        embeddings.append(embedding)
                    except Exception as item_e:
                        logger.error(
                            f"[{self.node_name}] Failed to generate embedding for item at index {i}. "
                            f"Error: {item_e}", exc_info=True
                        )
                        # Depending on desired resilience, either re-raise, append empty/zero vector, or skip.
                        # For critical orchestration, re-raising is often preferred.
                        raise # Re-raise to halt pipeline on individual item failure

                return embeddings
            else:
                error_msg = (
                    f"[{self.node_name}] Invalid input data type. "
                    f"Expected 'str' or 'List[str]', but received '{type(data).__name__}'. "
                    f"Data preview: {str(data)[:100]}..."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

        except Exception as e:
            logger.critical(
                f"[{self.node_name}] A critical, unexpected error occurred during "
                f"embedding generation: {e}", exc_info=True
            )
            # Re-raise the exception to signal upstream orchestrators of the failure.
            raise
