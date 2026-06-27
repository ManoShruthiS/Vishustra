import logging
import random
from typing import Any, Dict, List, Union

# Assuming this path is correct based on project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of text embeddings.

    This node takes textual data (a single string or a list of strings)
    and, for demonstration purposes, returns a simulated list of float
    embeddings. In a real-world scenario, this would interface with
    an actual embedding model service (e.g., OpenAI, HuggingFace, local model).

    The embedding dimension can be configured via the 'embedding_dimension'
    key in the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "embeddings_generator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates simulated embeddings for the input data.

        Expected data types:
        - str: A single text string to embed.
        - List[str]: A list of text strings to embed.

        Context keys:
        - 'embedding_dimension' (int, optional): The desired dimension for the
          generated embeddings. Defaults to 768 if not provided, aligning
          with common model output sizes like `text-embedding-ada-002`.

        Args:
            data (Any): The input data, expected to be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       and configuration for the node.

        Returns:
            Union[List[float], List[List[float]]]: A single embedding (List[float])
            if data was a string, or a list of embeddings (List[List[float]])
            if data was a list of strings.

        Raises:
            ValueError: If the input data type is not supported.
            TypeError: If 'embedding_dimension' in context is not a positive integer.
        """
        logger.info(f"[{self.node_name}] Starting embedding generation process.")

        embedding_dimension = context.get("embedding_dimension", 768)
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"[{self.node_name}] Invalid 'embedding_dimension' in context. "
                f"Expected a positive integer, got {embedding_dimension!r} "
                f"of type {type(embedding_dimension).__name__}."
            )
            raise TypeError(
                f"Context 'embedding_dimension' must be a positive integer, "
                f"but got {type(embedding_dimension).__name__} with value {embedding_dimension}."
            )

        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing a single text string.")
            result = self._generate_single_embedding(data, embedding_dimension)
            logger.info(f"[{self.node_name}] Successfully generated single embedding.")
            return result
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            logger.debug(f"[{self.node_name}] Processing a list of text strings ({len(data)} items).")
            embeddings = [self._generate_single_embedding(item, embedding_dimension) for item in data]
            logger.info(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        else:
            logger.error(
                f"[{self.node_name}] Invalid data type for embedding generation. "
                f"Expected str or List[str], got {type(data).__name__}."
            )
            raise ValueError(
                f"Unsupported data type for EmbeddingsGeneratorNode. "
                f"Expected str or List[str], got {type(data).__name__}."
            )

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Helper method to simulate generating a single embedding vector.

        Args:
            text (str): The input text to be embedded.
            dimension (int): The desired dimension of the embedding vector.

        Returns:
            List[float]: A list of floats representing the simulated embedding.
        """
        # In a real-world scenario, this method would interface with an actual
        # embedding model API or a locally loaded model. For this simulation,
        # we generate a list of random floats within a common embedding range.
        _ = text  # Acknowledge the 'text' parameter is received but unused in simulation
        embedding = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
        logger.debug(f"[{self.node_name}] Generated dummy embedding of dimension {dimension}.")
        return embedding