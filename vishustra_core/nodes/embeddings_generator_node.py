import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that generates simulated text embeddings.

    This node takes text (or a list of texts) as input and returns a
    simulated fixed-size vector representation for each text. It provides
    a placeholder for integrating actual embedding models, with configuration
    options passed via the context.
    """

    # Define a default embedding dimension for simulation purposes
    _EMBEDDING_DIMENSION = 128

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates generating a single embedding vector for a given text.

        In a production environment, this method would interface with a
        real embedding model (e.g., OpenAI, HuggingFace, local ONNX model)
        to produce a meaningful vector. For this simulation, it returns
        a list of random floats.
        """
        # Ensure determinism for testing or make it truly random
        # For simplicity, using random.uniform for simulation
        return [random.uniform(-1.0, 1.0) for _ in range(self._EMBEDDING_DIMENSION)]

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates simulated embeddings for the input text or a list of texts.

        The `context` dictionary can be used to pass parameters such as
        the desired embedding model's name or specific API keys, although
        these are currently only logged for the simulation.

        Args:
            data: The input text (str) or a list of texts (List[str]) to embed.
                  Empty strings or lists are handled gracefully.
            context: A dictionary containing contextual information, e.g.,
                     `model_name` for the embedding model to be used.

        Returns:
            A list of floats representing the embedding for a single text,
            or a list of lists of floats for multiple texts. The dimension
            of each embedding vector is fixed by `_EMBEDDING_DIMENSION`.

        Raises:
            ValueError: If the input 'data' is not a string or a list of strings,
                        or if any item in a list is not a string.
            Exception: For any unexpected operational errors during the process.
        """
        model_name = context.get("model_name", "default-simulated-embedding-model")
        logger.debug(f"[{self.node_name}] Starting embedding process with model: '{model_name}'.")

        try:
            if isinstance(data, str):
                if not data.strip():
                    logger.warning(
                        f"[{self.node_name}] Received an empty or whitespace-only string for embedding. "
                        "Returning a dummy embedding."
                    )
                embedding = self._generate_single_embedding(data)
                logger.debug(f"[{self.node_name}] Generated single embedding of dimension {len(embedding)}.")
                return embedding
            elif isinstance(data, list):
                if not data:
                    logger.debug(f"[{self.node_name}] Received an empty list of strings. Returning an empty list of embeddings.")
                    return []
                if not all(isinstance(item, str) for item in data):
                    # Log specific problematic items if needed for debugging
                    problematic_items = [item for item in data if not isinstance(item, str)]
                    logger.error(
                        f"[{self.node_name}] Input list contains non-string elements. "
                        f"First non-string type: {type(problematic_items[0]).__name__}."
                    )
                    raise ValueError("All items in the input list must be strings for batch embedding.")

                embeddings = [self._generate_single_embedding(text) for text in data]
                logger.debug(
                    f"[{self.node_name}] Generated {len(embeddings)} embeddings, "
                    f"each of dimension {self._EMBEDDING_DIMENSION}."
                )
                return embeddings
            else:
                logger.error(
                    f"[{self.node_name}] Invalid input data type. Expected 'str' or 'List[str]', "
                    f"but received '{type(data).__name__}'."
                )
                raise ValueError(
                    f"Invalid input data type for EmbeddingsGeneratorNode. "
                    f"Expected 'str' or 'List[str]', got '{type(data).__name__}'."
                )
        except ValueError as ve:
            logger.error(f"[{self.node_name}] Data validation failed: {ve}")
            raise
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during embedding generation for data type "
                f"'{type(data).__name__}'."
            )
            raise # Re-raise the exception after logging for upstream handling