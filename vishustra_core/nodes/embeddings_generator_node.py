import logging
import hashlib
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A processing node that simulates the generation of embeddings for text data.
    
    This node expects input `data` to be either a single string or a list of strings.
    It generates a deterministic, fixed-dimension embedding vector for each text
    input, simulating a call to an embedding model.
    
    Configuration for the embedding dimension can be provided via the `context`
    dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_dummy_embedding(self, text: str, dim: int) -> List[float]:
        """
        Generates a deterministic, normalized dummy embedding vector for a given text.
        This simulates the output of an embedding model without an actual external call.
        """
        if not isinstance(text, str):
            logger.error(f"Expected text to be a string, but received {type(text)}")
            raise TypeError("Input for dummy embedding generation must be a string.")
            
        if not text:
            # Return a zero vector for empty strings
            return [0.0] * dim

        # Use SHA256 hash of the text to seed the random number generator
        # This ensures deterministic embeddings for identical text inputs.
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        seed = int(text_hash, 16) % (2**32 - 1) # Ensure seed fits within standard integer limits
        rng = random.Random(seed)

        # Generate a vector of random floats
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(dim)]

        # L2 Normalize the vector
        magnitude = sum(x**2 for x in embedding)**0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        else:
            # If magnitude is 0 (e.g., all zeros vector), return it as is.
            # This handles cases where rng might generate all zeros, though unlikely.
            pass
            
        logger.debug(f"Generated dummy embedding (dim={dim}) for text snippet: '{text[:50]}...'")
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        Expected `data` types:
        - `str`: A single text string to embed.
        - `List[str]`: A list of text strings to embed.

        Optional `context` parameters:
        - `embedding_dim` (int): The desired dimension of the embedding vectors.
                                 Defaults to 1536 if not provided.

        Args:
            data (Any): The input data (string or list of strings).
            context (Dict[str, Any]): A dictionary containing contextual information
                                      and configuration parameters.

        Returns:
            Union[List[float], List[List[float]]]:
                - A single list of floats if `data` was a string.
                - A list of lists of floats if `data` was a list of strings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
            ValueError: If `embedding_dim` in context is not a positive integer.
        """
        embedding_dim = context.get('embedding_dim', 1536)

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.error(f"Invalid 'embedding_dim' in context: {embedding_dim}. Must be a positive integer.")
            raise ValueError("Configuration error: 'embedding_dim' must be a positive integer.")

        if isinstance(data, str):
            logger.info(f"Generating embedding for a single text input (dim={embedding_dim}).")
            try:
                return self._generate_dummy_embedding(data, embedding_dim)
            except Exception as e:
                logger.error(f"Failed to generate embedding for single string: {e}", exc_info=True)
                raise
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            logger.info(f"Generating embeddings for a list of {len(data)} text inputs (dim={embedding_dim}).")
            results = []
            for i, item in enumerate(data):
                try:
                    results.append(self._generate_dummy_embedding(item, embedding_dim))
                except Exception as e:
                    logger.error(f"Failed to generate embedding for item {i}: '{item[:50]}...'. Error: {e}", exc_info=True)
                    # Depending on desired error handling, either re-raise, return partial, or continue
                    raise # Re-raise immediately if any item fails
            return results
        else:
            logger.error(f"Invalid input data type: {type(data)}. Expected str or List[str].")
            raise TypeError("Input data must be a string or a list of strings.")

if __name__ == '__main__':
    # Basic usage example for local testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    embeddings_node = EmbeddingsGeneratorNode()

    # Test with a single string
    text_data = "Vishustra is a highly modular LLM orchestration framework."
    single_embedding = embeddings_node.process(text_data, {"embedding_dim": 768})
    logger.info(f"Single text embedding (dim={len(single_embedding)}): {single_embedding[:5]}...")

    # Test with a list of strings
    list_data = [
        "Hello world, this is a test sentence.",
        "Another sentence to be embedded by the Vishustra framework.",
        "Short text."
    ]
    list_embeddings = embeddings_node.process(list_data, {"embedding_dim": 1024})
    logger.info(f"List of embeddings (count={len(list_embeddings)}, dim={len(list_embeddings[0])}): {list_embeddings[0][:5]}...")

    # Test with different dimensions
    text_data_dim_512 = "This text will have 512 dimensions."
    embedding_512 = embeddings_node.process(text_data_dim_512, {"embedding_dim": 512})
    logger.info(f"512-dim embedding: {embedding_512[:5]}...")
    
    # Test with an empty string
    empty_string_embedding = embeddings_node.process("", {})
    logger.info(f"Empty string embedding (dim={len(empty_string_embedding)}): {empty_string_embedding[:5]}...")

    # Test error handling: invalid data type
    try:
        embeddings_node.process(123, {})
    except TypeError as e:
        logger.warning(f"Caught expected error for invalid data type: {e}")

    # Test error handling: invalid embedding_dim
    try:
        embeddings_node.process("some text", {"embedding_dim": "invalid"})
    except ValueError as e:
        logger.warning(f"Caught expected error for invalid embedding_dim: {e}")
        
    try:
        embeddings_node.process("some text", {"embedding_dim": 0})
    except ValueError as e:
        logger.warning(f"Caught expected error for non-positive embedding_dim: {e}")
