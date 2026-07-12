import logging
import random
import hashlib
from typing import Any, Dict, List, Union

# Assuming this import path based on project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of vector embeddings from text data.

    This node takes a string or a list of strings as input and produces
    a corresponding embedding vector or a list of embedding vectors.
    The embeddings are simulated for demonstration purposes, producing
    deterministic but distinct float vectors based on the input text.

    Configuration can be provided via the context dictionary, e.g.,
    'embedding_dimension' to specify the desired output vector size.
    """

    _DEFAULT_EMBEDDING_DIMENSION = 768 # Common dimension for many embedding models

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Simulates the generation of a single embedding vector for a given text.
        Uses a deterministic approach based on text hash to ensure consistent
        output for the same input text during simulation.

        Args:
            text: The input string for which to generate an embedding.
            dimension: The desired dimensionality of the output embedding vector.

        Returns:
            A list of floats representing the simulated embedding vector.
        """
        if not text:
            # For empty text, return a vector of zeros to signify no content
            return [0.0] * dimension

        # Use a consistent hash for seeding the random generator to ensure
        # deterministic "embeddings" for the same input text across runs.
        text_hash = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        
        # Create a local random generator instance to prevent interference with
        # the global random state and ensure reproducibility specific to this text.
        rng = random.Random(text_hash)
        
        # Generate 'dimension' number of floats between -1.0 and 1.0,
        # which is a common range for normalized embeddings.
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embeddings.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing operational context, which may include
                     'embedding_dimension' (int) to specify the size of the output vectors.

        Returns:
            A list of floats (for single string input) or a list of lists of floats
            (for list of strings input), representing the generated embeddings.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if list input contains non-string elements.
        """
        embedding_dimension = context.get('embedding_dimension', self._DEFAULT_EMBEDDING_DIMENSION)
        
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                "Invalid 'embedding_dimension' in context (%s). "
                "Must be a positive integer. Falling back to default: %s.",
                embedding_dimension, self._DEFAULT_EMBEDDING_DIMENSION
            )
            embedding_dimension = self._DEFAULT_EMBEDDING_DIMENSION

        if isinstance(data, str):
            logger.debug("Generating embedding for single text input.")
            return self._generate_single_embedding(data, embedding_dimension)
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                non_string_types = {type(item).__name__ for item in data if not isinstance(item, str)}
                logger.error(
                    "Input data is a list, but contains non-string elements. "
                    "Expected list of strings. Found types: %s", 
                    ", ".join(non_string_types)
                )
                raise ValueError("List input must contain only strings for embedding generation.")
            
            logger.debug(
                "Generating embeddings for a list of text inputs (batch size: %d).", 
                len(data)
            )
            return [self._generate_single_embedding(text, embedding_dimension) for text in data]
        else:
            logger.error(
                "Unsupported input data type: %s. Expected str or list of str.", 
                type(data).__name__
            )
            raise ValueError(
                f"Unsupported input data type: {type(data).__name__}. "
                "EmbeddingsGenerator expects a string or a list of strings."
            )