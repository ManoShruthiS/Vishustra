import logging
import random
from typing import Any, Dict, List, Union

# Assuming the vishustra_core package structure is in place
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of embeddings for text data.

    This node takes a string or a list of strings as input and returns
    a fixed-size vector (or a list of vectors) representing the embeddings.
    The embedding generation is simulated for demonstration purposes and
    is deterministic for a given input text.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information.
                     Expected keys:
                       - 'embedding_dimension' (int, optional): The desired
                         dimension of the output embedding vectors. Defaults to 768.

        Returns:
            A list of floats (if input data was a single string) or a list of lists of floats
            (if input data was a list of strings), representing the generated embeddings.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if elements within a list are not strings.
            ValueError: If 'embedding_dimension' in context is not a positive integer.
        """
        embedding_dimension = context.get('embedding_dimension', 768)

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.error(
                "Configuration error: 'embedding_dimension' in context must be a positive integer. Received: %s",
                embedding_dimension
            )
            raise ValueError(
                f"Invalid 'embedding_dimension' in context: {embedding_dimension}. "
                "Must be a positive integer."
            )

        is_single_string = isinstance(data, str)
        texts_to_process: List[str]

        if is_single_string:
            texts_to_process = [data]
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    "Data integrity error: Input list contains non-string elements. All elements must be strings."
                )
                raise TypeError(
                    "Input data must be a string or a list of strings. "
                    "List contains non-string elements."
                )
            texts_to_process = data
        else:
            logger.error(
                "Invalid input type for EmbeddingsGeneratorNode. Expected 'str' or 'List[str]', received '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"Input data must be a string or a list of strings, but received {type(data).__name__}."
            )

        logger.debug(
            "Initiating embedding generation for %d text item(s) with dimension %d.",
            len(texts_to_process), embedding_dimension
        )

        generated_embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_process):
            # Simulate embedding generation: generate a list of random floats.
            # In a production scenario, this would involve calling a specific
            # embedding model (e.g., from an NLP library or an external API).
            # We use a pseudo-random generator seeded by the text hash to ensure
            # reproducible "embeddings" for the same input text in tests.
            text_hash_seed = hash(text)
            rng = random.Random(text_hash_seed)  # Seed for deterministic simulation
            
            # Generate floats typically found in embedding spaces (e.g., between -1 and 1)
            embedding = [rng.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]
            generated_embeddings.append(embedding)
            
            logger.debug(
                "Generated embedding for text item %d (first 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]...)",
                i, *embedding[:5]
            )

        logger.info(
            "Successfully generated embeddings for %d item(s).",
            len(generated_embeddings)
        )

        # Return a single embedding if the input was a single string,
        # otherwise return the list of embeddings.
        return generated_embeddings[0] if is_single_string else generated_embeddings