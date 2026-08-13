import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of embedding vectors for text inputs.

    This node takes text (a string or a list of strings) and, based on configuration
    in the context, generates a list of floating-point numbers representing the
    embedding for each text input. This implementation uses random numbers to
    simulate embedding generation, making it suitable for development and testing
    without external model dependencies.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Generates embedding vectors for the input text(s).

        The 'context' dictionary can be used to specify:
        - 'embedding_dim': The desired dimension of the embedding vector (int).
                         Defaults to 768 if not provided or invalid.
        - 'model_name': A descriptive name for the simulated embedding model (str).
                      Defaults to "simulated-bge-small" if not provided.

        Args:
            data (Union[str, List[str]]): The text or list of texts to embed.
                                          Expects valid string input(s).
            context (Dict[str, Any]): A dictionary containing runtime parameters
                                      for the node.

        Returns:
            Union[List[float], List[List[float]]]: A single embedding vector (List[float])
                                                   if the input 'data' was a string,
                                                   or a list of embedding vectors (List[List[float]])
                                                   if the input 'data' was a list of strings.

        Raises:
            ValueError: If the input data is None, not a string or a list of strings,
                        or if the input list contains no valid strings.
            RuntimeError: If an unexpected error occurs during the embedding generation process.
        """
        # Retrieve configuration from context with sensible defaults
        embedding_dim = context.get("embedding_dim", 768)
        model_name = context.get("model_name", "simulated-bge-small")

        # Validate embedding_dim
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            logger.warning(
                "Invalid 'embedding_dim' in context: %s. Expected a positive integer. "
                "Defaulting to 768.",
                embedding_dim
            )
            embedding_dim = 768  # Fallback to default if context value is bad

        logger.info(
            "EmbeddingsGeneratorNode '%s' initiated processing. Model: '%s', Dimension: %d.",
            self.node_name, model_name, embedding_dim
        )

        texts_to_embed: List[str] = []
        is_single_input = False

        # Validate and prepare input data
        if data is None:
            logger.error("Input data is None. EmbeddingsGeneratorNode requires string or list of strings.")
            raise ValueError("Input data cannot be None for EmbeddingsGeneratorNode.")
        elif isinstance(data, str):
            texts_to_embed = [data]
            is_single_input = True
            logger.debug("Received a single string input for embedding.")
        elif isinstance(data, list):
            # Filter the list, keeping only strings and warning about invalid items
            filtered_data = []
            for item in data:
                if isinstance(item, str):
                    filtered_data.append(item)
                else:
                    logger.warning(
                        "Skipping non-string item in input list for embedding: %s (type: %s).",
                        item, type(item)
                    )
            if not filtered_data:
                logger.error("Input list contains no valid strings after filtering for EmbeddingsGeneratorNode.")
                raise ValueError("Input list must contain at least one valid string for embedding.")
            texts_to_embed = filtered_data
            logger.debug("Received a list of %d string inputs for embedding.", len(texts_to_embed))
        else:
            logger.error(
                "Invalid input data type for EmbeddingsGeneratorNode. Expected str or List[str], got %s.",
                type(data)
            )
            raise ValueError(
                f"Unsupported data type for embedding. Expected str or List[str], got {type(data)}."
            )

        generated_embeddings: List[List[float]] = []
        try:
            # Simulate embedding generation
            for i, text in enumerate(texts_to_embed):
                # Generate a list of random floats between -1.0 and 1.0
                embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]
                generated_embeddings.append(embedding)
                logger.debug(
                    "Generated embedding for text item %d (first 20 chars: '%s...') of dimension %d.",
                    i + 1, text[:20].replace('\n', ' '), embedding_dim
                )

            logger.info(
                "Successfully generated %d embedding(s) using model '%s' with dimension %d.",
                len(generated_embeddings), model_name, embedding_dim
            )

            # Return a single embedding if the input was a single string, otherwise return the list
            return generated_embeddings[0] if is_single_input else generated_embeddings

        except Exception as e:
            logger.exception(
                "An unexpected error occurred during embedding generation within node '%s'.",
                self.node_name
            )
            raise RuntimeError(
                f"Failed to generate embeddings for node '{self.node_name}': {e}"
            ) from e
