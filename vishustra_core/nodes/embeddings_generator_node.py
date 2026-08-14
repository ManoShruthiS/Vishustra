import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is available from the vishustra_core package as specified
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate (or simulate the generation of)
    text embeddings.

    This node accepts a single string or a list of strings as input data.
    It returns a list of embedding vectors, where each vector is a list of floats.
    The dimension of the embeddings can be specified via the 'embedding_dim'
    key in the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str, dim: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        In a production environment, this method would interface with an
        actual embedding model (e.g., from Hugging Face, OpenAI, Cohere, etc.).

        Args:
            text: The input string for which to generate an embedding.
            dim: The desired dimension for the embedding vector.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text:
            # Handle empty strings by returning a zero vector or specific placeholder
            logger.debug(f"[{self.node_name}] Generating zero vector for empty text.")
            return [0.0] * dim
        # Simulate an embedding with random floats between -1.0 and 1.0
        return [random.uniform(-1.0, 1.0) for _ in range(dim)]

    def process(self, data: Any, context: Dict[str, Any]) -> List[List[float]]:
        """
        Processes input data to generate embedding vectors.

        Args:
            data: The input to be embedded. This can be:
                  - A `str`: A single text string.
                  - A `List[str]`: A list of text strings.
            context: A dictionary containing operational context.
                     Expected keys:
                     - 'embedding_dim' (int): The desired dimension of the
                       embedding vectors. Defaults to 128 if not provided
                       or invalid.

        Returns:
            A `List[List[float]]`, where each inner list is an embedding vector.
            Each inner list corresponds to an input text in the order they were
            processed. Returns an empty list if no valid texts were processed
            or all embedding generations failed.

        Raises:
            TypeError: If the input 'data' is not a string or a list of strings.
            ValueError: If 'data' is a list containing non-string elements.
        """
        logger.debug(f"[{self.node_name}] Process initiated for data of type: {type(data).__name__}")

        embedding_dim = context.get('embedding_dim', 128)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid or non-positive 'embedding_dim' in context "
                f"('{embedding_dim}'). Using default dimension: 128."
            )
            embedding_dim = 128

        texts_to_embed: List[str] = []

        if isinstance(data, str):
            texts_to_embed.append(data)
        elif isinstance(data, list):
            # Validate that all elements in the list are strings
            if not all(isinstance(item, str) for item in data):
                non_str_items = [item for item in data if not isinstance(item, str)]
                error_msg = (
                    f"[{self.node_name}] Input list contains non-string elements. "
                    f"First non-string type encountered: {type(non_str_items[0]).__name__}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            texts_to_embed.extend(data)
        else:
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' or 'List[str]', got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not texts_to_embed:
            logger.info(f"[{self.node_name}] No valid texts provided for embedding. Returning an empty list.")
            return []

        embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_embed):
            try:
                embedding = self._generate_single_embedding(text, embedding_dim)
                embeddings.append(embedding)
                logger.debug(
                    f"[{self.node_name}] Generated embedding for text index {i} "
                    f"(first 20 chars: '{text[:20]}...') with dimension {embedding_dim}."
                )
            except Exception as e:
                # Log the error and continue processing other texts.
                # The caller will receive a list with fewer embeddings than input texts
                # for such failures.
                logger.error(
                    f"[{self.node_name}] Error generating embedding for text at index {i} "
                    f"('{text[:50]}...'): {e}",
                    exc_info=True  # Include stack trace for detailed debugging
                )
                # Decide whether to append a placeholder or simply skip. Skipping here.
                pass

        if not embeddings and texts_to_embed:
            logger.warning(f"[{self.node_name}] All embedding generations failed for the given inputs. Returning an empty list.")

        logger.debug(
            f"[{self.node_name}] Process completed. Generated {len(embeddings)} embedding vectors."
            f"{f' First vector length: {len(embeddings[0])}' if embeddings else ''}"
        )
        return embeddings
