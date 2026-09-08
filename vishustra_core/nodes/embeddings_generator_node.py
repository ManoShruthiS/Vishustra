import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node that simulates the generation of text embeddings.
    
    This node expects a string or a list of strings as input data. It simulates
    the transformation of this text into numerical vector representations (embeddings).
    The embedding dimension can be specified in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        This method generates a simulated embedding vector (or a list of vectors)
        for the provided text input.

        Expected `data` types:
        - `str`: A single text string to embed.
        - `List[str]`: A list of text strings to embed.

        Expected `context` parameters:
        - `embedding_dim` (int, optional): The desired dimension of the output embeddings.
          Defaults to 768 if not provided.

        Args:
            data: The input text(s) to be embedded.
            context: A dictionary containing operational context,
                     including optional `embedding_dim`.

        Returns:
            A list of floats (for single string input) or a list of lists of floats
            (for list of strings input), representing the generated embeddings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if list elements are not strings.
            ValueError: If `embedding_dim` in `context` is not a positive integer.
            Exception: For any other unexpected errors during embedding generation.
        """
        logger.debug(f"[{self.node_name}] Starting processing for data type: {type(data)}")

        embedding_dim = context.get("embedding_dim", 768)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.error(f"[{self.node_name}] Invalid 'embedding_dim' in context: {embedding_dim}. Must be a positive integer.")
            raise ValueError(f"Invalid 'embedding_dim' in context. Expected a positive integer, got {embedding_dim}.")

        def _generate_single_embedding(text_input: str, dim: int) -> List[float]:
            """Helper to simulate generating a single embedding vector."""
            # Simulate a vector of 'dim' random floats between -1.0 and 1.0.
            # The input text is not actually used in this simulation, only its presence matters.
            return [random.uniform(-1.0, 1.0) for _ in range(dim)]

        try:
            if isinstance(data, str):
                if not data.strip():
                    logger.warning(f"[{self.node_name}] Received an empty string for embedding generation. Generating a placeholder embedding.")
                embedding = _generate_single_embedding(data, embedding_dim)
                logger.info(f"[{self.node_name}] Generated single embedding of dimension {embedding_dim} for input string.")
                return embedding
            elif isinstance(data, list):
                if not data:
                    logger.warning(f"[{self.node_name}] Received an empty list for embedding generation. Returning an empty list of embeddings.")
                    return []
                
                if not all(isinstance(item, str) for item in data):
                    logger.error(f"[{self.node_name}] List input contains non-string elements.")
                    raise TypeError("All elements in the input list must be strings for embedding generation.")
                
                embeddings = []
                for item in data:
                    embeddings.append(_generate_single_embedding(item, embedding_dim))
                
                logger.info(f"[{self.node_name}] Generated {len(embeddings)} embeddings of dimension {embedding_dim} for list of strings.")
                return embeddings
            else:
                logger.error(f"[{self.node_name}] Invalid input data type: {type(data)}. Expected str or List[str].")
                raise TypeError(f"Invalid input data type for EmbeddingsGeneratorNode. Expected str or List[str], got {type(data)}.")
        except (TypeError, ValueError) as e:
            logger.error(f"[{self.node_name}] Data validation or processing error: {e}")
            raise
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during embedding generation.")
            raise # Re-raise the exception after logging for upstream handling
