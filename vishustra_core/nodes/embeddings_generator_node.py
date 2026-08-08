import logging
import math
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate or simulate the generation of embeddings
    for text data.

    This node accepts input data as either a single string or a list of strings.
    It processes these inputs to produce corresponding embedding vectors. For
    demonstration and testing purposes, this implementation generates
    deterministic mock embedding vectors based on the input text's properties
    (e.g., length, hash).

    Configuration parameters can be passed via the `context` dictionary:
    - 'embedding_dimension': int (default: 768) - Specifies the desired
      dimension of the output embedding vectors. This must be a positive integer.
    """

    # Default embedding dimension used if not specified in the context.
    _DEFAULT_EMBEDDING_DIMENSION: int = 768

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node.
        """
        return "EmbeddingsGeneratorNode"

    def _generate_mock_embedding(self, text: str, dim: int) -> List[float]:
        """
        Generates a deterministic mock embedding vector for a given text.
        This internal helper ensures that the same input text always yields
        the identical mock embedding, which is useful for predictable testing.

        Args:
            text (str): The input text for which to generate an embedding.
            dim (int): The desired dimension of the embedding vector.

        Returns:
            List[float]: A list of floats representing the mock embedding.
                         Returns a vector of zeros if the input text is empty.
        """
        if not text:
            # For an empty string, return a vector of zeros.
            return [0.0] * dim
        
        # Derive deterministic properties from the text.
        text_length = len(text)
        # Use absolute hash to avoid issues with negative hash values in calculations
        # and modulo to keep it within a manageable range for the sin function.
        text_hash_component = abs(hash(text)) % 1_000_000 
        
        embedding = []
        for i in range(dim):
            # Generate values between 0 and 1 using a sinusoidal function,
            # incorporating text properties to ensure determinism and variation.
            # (sin(x) + 1) / 2 maps values from [-1, 1] to [0, 1].
            value = (math.sin(i + text_length + text_hash_component) + 1) / 2.0
            embedding.append(value)
        
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        The `data` input is expected to be either a single string or a list of strings.
        The `context` dictionary can specify the 'embedding_dimension'.

        Args:
            data (Any): The input data. Must be a `str` or `List[str]`.
            context (Dict[str, Any]): A dictionary containing configuration for the node.
                                       Expected to optionally contain:
                                       - 'embedding_dimension': int, a positive integer
                                         specifying the dimension of the generated embeddings.

        Returns:
            Union[List[float], List[List[float]]]: If `data` is a `str`, returns a single
            `List[float]` (the embedding vector). If `data` is a `List[str]`, returns
            a `List[List[float]]` (a list of embedding vectors).

        Raises:
            TypeError: If the input data is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If an invalid or non-positive integer `embedding_dimension`
                        is provided in the context.
        """
        logger.debug(f"[{self.node_name}] Initiating process for input data of type: {type(data)}.")

        # Determine the embedding dimension from context or use the default.
        embedding_dimension = context.get('embedding_dimension')
        if embedding_dimension is None:
            embedding_dimension = self._DEFAULT_EMBEDDING_DIMENSION
            logger.debug(f"[{self.node_name}] Using default embedding dimension: {embedding_dimension}.")
        elif not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                f"[{self.node_name}] Invalid 'embedding_dimension' in context: {embedding_dimension}. "
                "Must be a positive integer."
            )
            raise ValueError(
                f"Invalid 'embedding_dimension' in context: {embedding_dimension}. "
                "Expected a positive integer."
            )
        else:
            logger.debug(f"[{self.node_name}] Using configured embedding dimension: {embedding_dimension}.")

        if isinstance(data, str):
            logger.info(
                f"[{self.node_name}] Generating embedding for a single text input (length: {len(data)}) "
                f"with dimension {embedding_dimension}."
            )
            return self._generate_mock_embedding(data, embedding_dimension)
        
        elif isinstance(data, list):
            # Validate that all elements in the list are strings.
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. "
                    "All elements in the list must be strings for embedding generation."
                )
                raise TypeError(
                    "Input data is a list, but contains non-string elements. "
                    "Expected List[str] for embedding generation."
                )
            
            logger.info(
                f"[{self.node_name}] Generating embeddings for a list of {len(data)} text inputs "
                f"with dimension {embedding_dimension} per embedding."
            )
            results = [self._generate_mock_embedding(text, embedding_dimension) for text in data]
            return results
            
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data)}. "
                "Expected 'str' or 'List[str]' for embedding generation."
            )
            raise TypeError(
                f"Invalid input data type: {type(data)}. "
                "Expected 'str' or 'List[str]' for embedding generation."
            )