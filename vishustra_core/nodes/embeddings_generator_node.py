import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node designed to generate or simulate the generation
    of text embeddings.

    This node accepts either a single string or a list of strings as input.
    It returns a corresponding numerical vector (embedding) or a list of vectors.
    The dimension of the generated embeddings can be configured via the 'embedding_dim'
    key in the context dictionary. If 'embedding_dim' is not provided or is invalid,
    a sensible default dimension is used.

    In a production environment, this node would typically interface with an
    external embedding model service or a local embedding model. For demonstration
    purposes, it simulates embedding generation.
    """

    DEFAULT_EMBEDDING_DIMENSION = 1536  # A common embedding dimension (e.g., OpenAI ada-002)

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "EmbeddingsGeneratorNode"

    def _generate_single_embedding(self, text: str, embedding_dim: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.

        In a real-world scenario, this method would encapsulate the logic for
        calling an external embedding API (e.g., via HTTP, gRPC) or invoking
        a local machine learning model.

        Args:
            text (str): The input text for which to generate an embedding.
            embedding_dim (int): The desired dimension of the output embedding vector.

        Returns:
            List[float]: A list of floats representing the generated embedding.

        Raises:
            TypeError: If the input `text` is not a string.
        """
        if not isinstance(text, str):
            logger.error(f"Attempted to generate embedding for non-string input type: {type(text)}")
            raise TypeError(f"Input for embedding generation must be a string, received {type(text)}.")

        # Simulate a vector of 'embedding_dim' random floats between -1.0 and 1.0.
        # This stands in for the actual numerical output from an embedding model.
        embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]
        logger.debug(f"Generated simulated embedding of dimension {embedding_dim} for text (first 20 chars): '{text[:20]}...'")
        return embedding

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        This method orchestrates the embedding generation, handling both single
        text inputs and batches of text inputs. It retrieves configuration like
        embedding dimension from the `context`.

        Args:
            data (Union[str, List[str]]): The text or list of texts for which
                                          embeddings are to be generated.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       and configuration parameters for the node.
                                       Expected keys:
                                       - 'embedding_dim' (int, optional): The target
                                         dimension for the embedding vectors. If not
                                         specified or invalid, `DEFAULT_EMBEDDING_DIMENSION`
                                         is used.

        Returns:
            Union[List[float], List[List[float]]]:
                - If `data` was a single string, returns a `List[float]` representing
                  its embedding vector.
                - If `data` was a `List[str]`, returns a `List[List[float]]` where each
                  inner list is the embedding vector for the corresponding input string.

        Raises:
            TypeError: If `data` is not a string or a list of strings.
            ValueError: If `data` is a list containing non-string elements.
            RuntimeError: If an error occurs during the simulated embedding generation
                          for any input.
        """
        embedding_dim = context.get('embedding_dim', self.DEFAULT_EMBEDDING_DIMENSION)

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.warning(
                f"Invalid 'embedding_dim' in context: {embedding_dim}. "
                f"Expected a positive integer. Falling back to default: {self.DEFAULT_EMBEDDING_DIMENSION}."
            )
            embedding_dim = self.DEFAULT_EMBEDDING_DIMENSION

        if isinstance(data, str):
            try:
                logger.info(f"Generating embedding for a single text input (first 50 chars): '{data[:50]}...'")
                return self._generate_single_embedding(data, embedding_dim)
            except Exception as e:
                logger.error(f"Failed to generate embedding for single text input: {e}", exc_info=True)
                raise RuntimeError(f"Embedding generation failed for single text: {e}") from e
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                non_str_items = [item for item in data if not isinstance(item, str)]
                error_msg = (
                    f"Input list for EmbeddingsGeneratorNode contains non-string elements. "
                    f"Found types: {[type(item).__name__ for item in non_str_items[:5]]} (showing first 5)."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Generating embeddings for a batch of {len(data)} text inputs.")
            results = []
            for i, text_item in enumerate(data):
                try:
                    results.append(self._generate_single_embedding(text_item, embedding_dim))
                except Exception as e:
                    logger.error(
                        f"Failed to generate embedding for item {i} in batch "
                        f"(first 50 chars): '{text_item[:50]}...': {e}", exc_info=True
                    )
                    # For a batch, a failure in one item might be critical enough to halt.
                    # Depending on requirements, one might log and skip, or raise. Here, we raise.
                    raise RuntimeError(f"Embedding generation failed for batch item {i}: {e}") from e
            return results
        else:
            error_msg = (
                f"Invalid input type for EmbeddingsGeneratorNode. Expected str or List[str], "
                f"but received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)