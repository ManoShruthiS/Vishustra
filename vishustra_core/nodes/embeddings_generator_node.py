import logging
from typing import Any, Dict, List, Union
import numpy as np

# Assuming BaseNode is located in vishustra_core.nodes.base_node as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating embedding vectors
    from input text data.

    This node is designed to simulate or integrate with an external embedding model
    to transform textual input into high-dimensional numerical vectors.
    It supports processing both single strings and lists of strings.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "EmbeddingsGenerator"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[List[float]]:
        """
        Generates simulated embedding vectors for the provided text data.

        The `data` input can be a single string or a list of strings. Each string
        will be processed to generate a corresponding embedding vector. In a
        production environment, this method would interface with an actual
        embedding service or model.

        Configuration for the embedding dimension can be provided through the
        `context` dictionary.

        Args:
            data: The input text (str or List[str]) to be embedded.
                  If a string, it's treated as a single item. If a list,
                  each element is expected to be a string.
            context: A dictionary containing operational context, which may include
                     configuration parameters for the embedding generation:
                     - `embedding_config`: (Optional) A dictionary that can contain:
                         - `dimension`: An integer specifying the desired dimension
                           for the output embedding vectors. If not provided or invalid,
                           a default of 768 is used.
                     - `embedding_dimension`: (Optional) An integer for the embedding
                       dimension, as an alternative top-level key to `embedding_config`.

        Returns:
            A list of lists of floats, where each inner list represents
            an embedding vector for an input text item. The order of embeddings
            corresponds to the order of input text items.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if any item within a list is not a string.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"Received invalid input data type: {type(data)}. "
                f"Expected `str` or `List[str]` for EmbeddingsGeneratorNode."
            )
            raise ValueError(
                f"EmbeddingsGeneratorNode requires input 'data' to be a string "
                f"or a list of strings, but received {type(data)}."
            )

        texts_to_embed: List[str]
        if isinstance(data, str):
            texts_to_embed = [data]
        else:  # data is List
            if not all(isinstance(item, str) for item in data):
                logger.error("List input data contains non-string elements.")
                raise ValueError(
                    "EmbeddingsGeneratorNode received a list containing non-string "
                    "elements. All list items must be strings."
                )
            texts_to_embed = data

        embedding_dimension: int = 768  # Default embedding dimension
        configured_dimension: Union[int, None] = None

        # Prioritize 'embedding_config' dictionary
        if 'embedding_config' in context and isinstance(context['embedding_config'], dict):
            config = context['embedding_config']
            if 'dimension' in config:
                try:
                    dim_from_context = int(config['dimension'])
                    if dim_from_context <= 0:
                        raise ValueError("Embedding dimension must be a positive integer.")
                    configured_dimension = dim_from_context
                    logger.debug(f"Embedding dimension '{configured_dimension}' from 'embedding_config'.")
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid 'dimension' in 'embedding_config' context: '{config['dimension']}'. "
                        f"Reason: {e}. Using default dimension {embedding_dimension}."
                    )
        
        # Fallback to top-level 'embedding_dimension' if not configured via 'embedding_config'
        if configured_dimension is None and 'embedding_dimension' in context:
            try:
                dim_from_context = int(context['embedding_dimension'])
                if dim_from_context <= 0:
                    raise ValueError("Embedding dimension must be a positive integer.")
                configured_dimension = dim_from_context
                logger.debug(f"Embedding dimension '{configured_dimension}' from top-level context key.")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid 'embedding_dimension' in context: '{context['embedding_dimension']}'. "
                    f"Reason: {e}. Using default dimension {embedding_dimension}."
                )
        
        if configured_dimension is not None:
            embedding_dimension = configured_dimension

        logger.info(
            f"Generating {len(texts_to_embed)} simulated embeddings "
            f"with dimension {embedding_dimension}."
        )

        embeddings: List[List[float]] = []
        if not texts_to_embed:
            logger.info("No text items to embed. Returning an empty list of embeddings.")
            return []

        for i, text_item in enumerate(texts_to_embed):
            # In a real-world scenario, this is where the actual call to an
            # embedding model (e.g., self._embedding_model.encode(text_item))
            # would occur. For this simulation, we generate random vectors.
            simulated_embedding = np.random.rand(embedding_dimension).tolist()
            embeddings.append(simulated_embedding)
            logger.debug(
                f"Generated embedding for item {i+1}/{len(texts_to_embed)} "
                f"(first 20 chars): '{text_item[:20]}...'"
            )

        logger.info(f"Successfully generated {len(embeddings)} embedding vectors.")
        return embeddings