import logging
import random
from typing import Any, Dict, List, Union

# Assuming this path exists in the project structure for Vishustra
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating text embeddings.

    This node simulates the interaction with an embedding model to transform
    input text (either a single string or a list of strings) into numerical
    embedding vectors. The dimensions and model used can be configured via context.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates embedding vectors for the provided text data.

        The method expects 'data' to be either a single string or a list of strings.
        It uses 'context' to determine the embedding model name and the desired
        dimension for the generated embeddings.

        Args:
            data (Union[str, List[str]]): The text input for which to generate embeddings.
                                          Can be a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing operational parameters.
                                      Expected keys include:
                                      - 'embedding_model_name' (str, optional): The identifier for the
                                        embedding model to simulate. Defaults to 'default_embedding_model'.
                                      - 'embedding_dimension' (int, optional): The dimensionality of the
                                        output embedding vectors. Defaults to 768.

        Returns:
            Union[List[float], List[List[float]]]: If 'data' was a single string, a list of floats
                                                    representing one embedding vector. If 'data' was a
                                                    list of strings, a list of lists of floats, where
                                                    each inner list is an embedding vector for the
                                                    corresponding input string.

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings,
                       or if a list contains non-string items.
            ValueError: If an unexpected error occurs during the embedding generation process.
        """
        logger.debug(f"[{self.node_name}] Initiating processing for input data of type: {type(data)}")
        logger.debug(f"[{self.node_name}] Received context: {context}")

        embedding_model_name: str = context.get('embedding_model_name', 'default_embedding_model')
        embedding_dimension: int = context.get('embedding_dimension', 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'embedding_dimension' '{embedding_dimension}' specified in context. "
                f"It must be a positive integer. Defaulting to 768."
            )
            embedding_dimension = 768

        logger.info(f"[{self.node_name}] Simulating embedding generation using model "
                    f"'{embedding_model_name}' with dimension {embedding_dimension}.")

        def _generate_single_embedding(text: str) -> List[float]:
            """
            Internal helper to simulate the generation of a single embedding vector.
            In a real-world scenario, this would involve calling an actual ML model
            or an external API.
            """
            # For demonstration, we produce a vector of random floats.
            return [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]

        try:
            if isinstance(data, str):
                logger.debug(f"[{self.node_name}] Processing a single string input.")
                embedding = _generate_single_embedding(data)
                logger.info(f"[{self.node_name}] Successfully generated embedding for a single text input.")
                return embedding
            elif isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    problematic_types = {type(item).__name__ for item in data if not isinstance(item, str)}
                    raise TypeError(
                        f"[{self.node_name}] All items within the input list must be strings. "
                        f"Detected non-string types: {', '.join(problematic_types)}."
                    )
                logger.debug(f"[{self.node_name}] Processing a list of {len(data)} strings.")
                embeddings = [_generate_single_embedding(item) for item in data]
                logger.info(f"[{self.node_name}] Successfully generated embeddings for {len(data)} text items.")
                return embeddings
            else:
                raise TypeError(
                    f"[{self.node_name}] Input 'data' must be a string or a list of strings, "
                    f"but received type: {type(data).__name__}."
                )
        except TypeError as e:
            logger.error(f"[{self.node_name}] Data type validation failed: {e}", exc_info=False)
            raise  # Re-raise the TypeError
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unhandled exception occurred during embedding generation: {e}", exc_info=True
            )
            # Re-raise as a generic ValueError for upstream handling, providing context
            raise ValueError(f"[{self.node_name}] Failed to generate embeddings due to an unexpected error.") from e