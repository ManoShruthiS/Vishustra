import logging
import random
from typing import Any, Dict, List, Union

# Assuming the BaseNode is structured under vishustra_core.nodes
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node designed to generate simulated vector embeddings for text data.

    This node accepts either a single string or a list of strings as input
    and produces corresponding simulated fixed-dimension embedding vectors.
    The simulation provides deterministic, pseudo-random vectors based on the
    content of the input text, making it highly suitable for development,
    testing, and prototyping scenarios where an actual embedding model
    is not required or desired.
    """

    DEFAULT_EMBEDDING_DIMENSION = 768
    # Define a range for dummy embedding values to simulate typical float outputs
    MIN_DUMMY_VALUE = -0.5
    MAX_DUMMY_VALUE = 0.5

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGeneratorNode"

    def _generate_dummy_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Generates a deterministic dummy embedding vector for a given text.
        
        The vector is pseudo-randomly generated using the hash of the text
        as a seed, ensuring that the same text consistently produces the
        same embedding vector across executions.

        Args:
            text (str): The input text for which to generate an embedding.
            dimension (int): The desired dimensionality of the output vector.

        Returns:
            List[float]: A list of floats representing the generated embedding.

        Raises:
            TypeError: If the input `text` is not a string.
        """
        if not isinstance(text, str):
            logger.error(
                f"[{self.node_name}] Attempted to generate dummy embedding for "
                f"non-string type: {type(text)}. Input must be a string."
            )
            raise TypeError("Input for dummy embedding generation must be a string.")

        # Use the hash of the text to seed the random number generator.
        # This ensures reproducibility: the same text always yields the same embedding.
        text_hash = hash(text)
        rng = random.Random(text_hash)

        embedding = []
        for _ in range(dimension):
            # Generate a float within the defined dummy value range
            value = rng.uniform(self.MIN_DUMMY_VALUE, self.MAX_DUMMY_VALUE)
            embedding.append(value)
        
        return embedding

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate simulated embedding vectors.

        The `context` dictionary can be used to specify configuration parameters:
        - 'embedding_dimension' (int, optional): The desired dimension for the
          output embeddings. If not provided or invalid, `DEFAULT_EMBEDDING_DIMENSION`
          will be used.

        Args:
            data (Union[str, List[str]]): The input text (single string) or
                                          a list of texts (batch) to embed.
            context (Dict[str, Any]): A dictionary containing runtime configuration
                                       and state relevant to the node's operation.

        Returns:
            Union[List[float], List[List[float]]]: If `data` is a single string,
            returns a `List[float]` (a single embedding vector). If `data` is
            a list of strings, returns a `List[List[float]]` (a list of embedding vectors).

        Raises:
            ValueError: If the input `data` is not a string or a list of strings.
            TypeError: If an element within an input list of strings is not a string.
        """
        logger.info(f"[{self.node_name}] Initiating data processing for embedding generation.")

        embedding_dimension = context.get('embedding_dimension', self.DEFAULT_EMBEDDING_DIMENSION)
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid or missing 'embedding_dimension' in context. "
                f"Expected a positive integer, received: {embedding_dimension}. "
                f"Falling back to default dimension: {self.DEFAULT_EMBEDDING_DIMENSION}."
            )
            embedding_dimension = self.DEFAULT_EMBEDDING_DIMENSION
        
        output_embeddings: Union[List[float], List[List[float]]]

        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing a single text input.")
            output_embeddings = self._generate_dummy_embedding(data, embedding_dimension)
            log_message = f"Generated single embedding of dimension {embedding_dimension}."
        elif isinstance(data, list):
            logger.debug(f"[{self.node_name}] Processing a list of text inputs (batch mode).")
            batch_embeddings = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(
                        f"[{self.node_name}] Batch input contains an invalid element type "
                        f"at index {i}. Expected 'str', found '{type(item)}'."
                    )
                    raise TypeError(
                        f"All elements within the input list must be strings. "
                        f"Encountered type '{type(item)}' at index {i}."
                    )
                batch_embeddings.append(self._generate_dummy_embedding(item, embedding_dimension))
            output_embeddings = batch_embeddings
            log_message = (
                f"Generated {len(output_embeddings)} embeddings, each of dimension "
                f"{embedding_dimension} for batch input."
            )
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str' or 'List[str]', "
                f"but received type '{type(data)}'."
            )
            raise ValueError(
                f"Invalid input data type for EmbeddingsGeneratorNode. Expected 'str' or 'List[str]', "
                f"but received '{type(data)}'."
            )

        logger.info(f"[{self.node_name}] Processing complete. {log_message}")
        return output_embeddings