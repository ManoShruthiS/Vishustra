import logging
import hashlib
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node responsible for generating numerical embeddings for text data.

    This node simulates the process of converting input text (either a single string
    or a list of strings) into fixed-size numerical vector embeddings. In a production
    environment, this would integrate with an actual embedding model service (e.g.,
    via an API call to a cloud provider or a local model server).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str, embedding_dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        This method provides a deterministic, dummy embedding for demonstration purposes.
        It converts the input text into a fixed-size list of floats, mimicking
        the structure of a real embedding vector.

        Args:
            text (str): The input text to generate an embedding for.
            embedding_dimension (int): The desired dimension of the output vector.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        # Generate a stable "seed" from the text content.
        # In a real system, this would involve a complex neural network inference.
        hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)

        # Create a list of floats based on the hash, scaled and wrapped.
        # This provides a unique (for given text) and somewhat diverse vector.
        embedding = [
            (hash_val / (10**9) * (i + 1) % 1.0) * 2 - 1
            for i in range(embedding_dimension)
        ]

        # Simple L2 normalization simulation, common for embeddings.
        norm = sum(x * x for x in embedding)**0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (text or a list of texts) to generate embeddings.

        The `context` dictionary can be used to pass configuration such as the
        desired embedding dimension or the name of the underlying model.

        Args:
            data (Union[str, List[str]]): The text data to embed. Can be a single
                                          string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for embedding generation. Expected keys:
                                      - 'embedding_dimension' (int, optional): The desired
                                        dimension of the output embedding vectors. Defaults to 768.
                                      - 'model_name' (str, optional): The name of the embedding model
                                        to use (e.g., "text-embedding-ada-002"). Used for logging.

        Returns:
            Union[List[float], List[List[float]]]: The generated embedding(s).
                                                    A list of floats if `data` was a single string,
                                                    or a list of lists of floats if `data` was a list
                                                    of strings.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if elements within the list are not strings.
            Exception: For any unexpected errors encountered during the embedding
                       generation process.
        """
        embedding_dimension = context.get("embedding_dimension", 768)
        model_name = context.get("model_name", "simulated-embedding-model")

        logger.info(
            f"[{self.node_name}] Starting embedding generation using model "
            f"'{model_name}' with dimension {embedding_dimension}."
        )

        if not isinstance(data, (str, list)):
            logger.error(
                f"[{self.node_name}] Invalid input data type: Expected 'str' or 'list[str]', "
                f"but received '{type(data).__name__}'."
            )
            raise ValueError(
                f"Invalid input data type for {self.node_name}. Expected 'str' or 'list[str]', "
                f"got '{type(data).__name__}'."
            )

        try:
            if isinstance(data, str):
                logger.debug(f"[{self.node_name}] Generating embedding for a single text item.")
                embedding = self._generate_single_embedding(data, embedding_dimension)
                logger.info(
                    f"[{self.node_name}] Successfully generated single embedding of "
                    f"dimension {len(embedding)}."
                )
                return embedding
            elif isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    logger.error(
                        f"[{self.node_name}] Invalid list elements: All elements in the input list "
                        f"must be strings. Detected non-string elements."
                    )
                    raise ValueError(
                        f"All elements in the input list for {self.node_name} must be strings."
                    )

                logger.debug(
                    f"[{self.node_name}] Generating embeddings for a list of {len(data)} text items."
                )
                embeddings = [
                    self._generate_single_embedding(item, embedding_dimension) for item in data
                ]
                logger.info(
                    f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings, "
                    f"each of dimension {embedding_dimension}."
                )
                return embeddings
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during embedding generation."
            )
            # Re-raise with additional context for upstream error handling
            raise Exception(f"Failed to generate embeddings in {self.node_name}: {e}") from e