import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that simulates the generation of embeddings for input text data.
    It takes a string or a list of strings and returns a corresponding embedding
    vector or a list of embedding vectors.

    This node uses a fixed set of parameters from the context to simulate
    embedding dimensions and provides basic error handling for input types.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Embeddings Generator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates simulated embeddings for the input text data.

        The `data` input can be a single string or a list of strings.
        The `context` dictionary can optionally contain 'embedding_config' with
        'dimensions' (int) and 'model_name' (str) to configure the simulated
        embedding generation.

        Args:
            data: The input text data, either a single string or a list of strings.
            context: A dictionary containing contextual information, potentially
                     including 'embedding_config' -> {'dimensions': int, 'model_name': str}.

        Returns:
            A list of floats (for a single string input) or a list of lists of floats
            (for a list of strings input), representing the simulated embeddings.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if a non-string item is found within an input list.
            RuntimeError: If embedding generation parameters (e.g., 'dimensions')
                          are invalid or missing in a way that prevents simulation.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                f"Invalid input data type for EmbeddingsGeneratorNode. "
                f"Expected `str` or `list[str]`, but received `{type(data).__name__}`."
            )
            raise ValueError(
                f"EmbeddingsGeneratorNode requires input data to be a string or a list of strings, "
                f"but received type: {type(data).__name__}"
            )

        embedding_config = context.get("embedding_config", {})
        dimensions = embedding_config.get("dimensions", 768)  # Default BERT-like dimension
        model_name = embedding_config.get("model_name", "simulated-embedding-model")

        if not isinstance(dimensions, int) or dimensions <= 0:
            logger.error(
                f"Invalid 'dimensions' specified in `context['embedding_config']`. "
                f"Expected a positive integer, but got `{dimensions}`."
            )
            raise RuntimeError(
                f"Invalid 'dimensions' for embedding generation: {dimensions}. "
                "Expected a positive integer."
            )

        num_items = len(data) if isinstance(data, list) else 1
        logger.info(
            f"EmbeddingsGeneratorNode initiated processing for {num_items} item(s) "
            f"using simulated model '{model_name}' with {dimensions} dimensions."
        )

        def _generate_single_embedding(text: str) -> List[float]:
            """Simulates generating an embedding for a single text string."""
            # In a real scenario, this would call an external embedding model API or local model.
            # Here, we generate random floats for demonstration purposes.
            # Using a simple deterministic seed based on text content for consistency in simulation.
            seed = sum(ord(c) for c in text[:100]) % 1000 # Limit text for seed to avoid very long sums
            current_rng_state = random.getstate() # Save current state
            random.seed(seed)
            embedding = [random.uniform(-1.0, 1.0) for _ in range(dimensions)]
            random.setstate(current_rng_state) # Restore RNG state
            logger.debug(f"Generated simulated embedding for text (first 20 chars): '{text[:20]}...' of dimension {dimensions}.")
            return embedding

        if isinstance(data, str):
            try:
                logger.debug(f"Processing single string input for embedding: '{data[:50]}...'")
                return _generate_single_embedding(data)
            except Exception as e:
                logger.error(f"Failed to generate embedding for single string input. Error: {e}", exc_info=True)
                raise RuntimeError(f"Failed to generate embedding for input string due to: {e}") from e
        else: # data is List[Any]
            embeddings: List[List[float]] = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.warning(
                        f"Skipping non-string item at index {i} in input list for EmbeddingsGeneratorNode. "
                        f"Expected `str`, but found `{type(item).__name__}`. This item will not be embedded."
                    )
                    # Depending on requirements, could raise ValueError here or append a placeholder.
                    # For robustness, we skip and log a warning. If strictness is required, uncomment next line.
                    # raise ValueError(f"List item at index {i} is not a string (type: {type(item).__name__}).")
                    continue
                try:
                    embeddings.append(_generate_single_embedding(item))
                except Exception as e:
                    logger.error(f"Failed to generate embedding for item at index {i}: '{item[:50]}...'. Error: {e}", exc_info=True)
                    # For critical failures on individual items, we raise to signal an issue.
                    raise RuntimeError(f"Failed to generate embedding for item at index {i} due to: {e}") from e
            logger.info(f"Successfully generated {len(embeddings)} simulated embeddings out of {num_items} processed items.")
            return embeddings