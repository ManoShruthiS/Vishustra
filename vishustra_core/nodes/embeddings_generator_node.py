import logging
import random
from typing import Any, Dict, List, Union

# Vishustra project core import for BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node designed to generate numerical embeddings
    for textual data.

    This node simulates the interaction with an embedding model, converting
    input text (or a list of texts) into high-dimensional vectors that
    capture semantic meaning. Configuration for the embedding model, such
    as embedding dimension, can be passed via the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input text data to generate simulated embeddings.

        This method expects 'data' to be either a single string or a list of strings.
        It simulates the output of an embedding model by generating lists of random
        floating-point numbers.

        Configuration parameters can be provided in the 'context' dictionary:
        - 'embedding_dim' (int): The desired dimension of the output embeddings.
                                 Defaults to 768 if not specified or invalid.
        - 'model_name' (str): A descriptive name for the simulated embedding model.
                              Defaults to 'simulated-embedding-model-v1'.

        Args:
            data (Union[str, List[str]]): The input text(s) for which embeddings
                                          are to be generated.
            context (Dict[str, Any]): A dictionary containing runtime context and
                                      configuration for the embedding process.

        Returns:
            Union[List[float], List[List[float]]]:
                - A list of floats if 'data' was a single string.
                - A list of lists of floats if 'data' was a list of strings.
                Returns an empty list for an empty input list, or a zero-vector
                for an empty/invalid single string input.

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings.
            Exception: Propagates any unexpected errors encountered during processing.
        """
        # Retrieve and validate embedding dimension from context
        embedding_dim = context.get('embedding_dim', 768)
        model_name = context.get('model_name', 'simulated-embedding-model-v1')

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid 'embedding_dim' in context. "
                f"Expected a positive integer, received '{embedding_dim}'. "
                f"Defaulting to 768 for resilience."
            )
            embedding_dim = 768 # Fallback to default if invalid

        logger.debug(
            f"[{self.node_name}] Initializing embedding generation with "
            f"model: '{model_name}', dimension: {embedding_dim}."
        )

        def _generate_single_embedding(text_segment: str) -> List[float]:
            """
            Internal helper to simulate generating an embedding for a single text segment.
            In a production environment, this would involve calling a specific
            embedding service or an NLP library.
            """
            # Simulate embedding with random floats, typically between -1 and 1 or 0 and 1.
            # Using -1 to 1 for a general neural network output simulation.
            return [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]

        try:
            if isinstance(data, str):
                if not data.strip():
                    logger.warning(
                        f"[{self.node_name}] Input 'data' is an empty or whitespace-only string. "
                        f"Returning a zero-vector embedding of dimension {embedding_dim}."
                    )
                    return [0.0] * embedding_dim
                return _generate_single_embedding(data)

            elif isinstance(data, list):
                if not data:
                    logger.warning(
                        f"[{self.node_name}] Input 'data' is an empty list. "
                        f"Returning an empty list of embeddings."
                    )
                    return []

                embeddings: List[List[float]] = []
                for idx, item in enumerate(data):
                    if not isinstance(item, str):
                        logger.error(
                            f"[{self.node_name}] List item at index {idx} is not a string "
                            f"(type: {type(item)}). Skipping this item as it cannot be embedded."
                        )
                        continue # Skip non-string items to avoid downstream errors
                    
                    if not item.strip():
                        logger.warning(
                            f"[{self.node_name}] List item at index {idx} is an empty "
                            f"or whitespace-only string. Generating a zero-vector."
                        )
                        embeddings.append([0.0] * embedding_dim)
                    else:
                        embeddings.append(_generate_single_embedding(item))
                
                if not embeddings and data:
                    logger.warning(
                        f"[{self.node_name}] No valid embeddings could be generated from the "
                        f"provided list. All items were either invalid types or empty strings."
                    )
                return embeddings

            else:
                error_msg = (
                    f"[{self.node_name}] Invalid input 'data' type. "
                    f"Expected 'str' or 'List[str]', but received '{type(data)}'."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

        except Exception as e:
            # Catch any unexpected errors during the embedding simulation process
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during embedding generation."
            )
            # Re-raise the exception for upstream orchestration to handle consistently.
            raise