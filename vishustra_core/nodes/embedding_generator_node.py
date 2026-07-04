import logging
from typing import Any, Dict, List, Union

# --- Vishustra Core Imports ---
# Assuming BaseNode is available at this path as per project context.
# A fallback definition is included for isolated file testing/linting
# where the full vishustra_core package might not be installed.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # This block is for local development/testing of this single file
    # outside the full Vishustra project environment.
    from abc import ABC, abstractmethod

    class BaseNode(ABC):
        """
        Base class for all Vishustra processing nodes.
        Each node must implement the process method.
        (Fallback definition for local testing)
        """

        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """
            Processes the input data and returns the result.
            """
            pass

        @property
        @abstractmethod
        def node_name(self) -> str:
            """Returns the name of the node."""
            pass

# --- Node Specific Code ---
logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node that generates numerical embeddings for text data.

    This node is designed to process either a single string or a list of strings,
    returning corresponding embedding vectors. For demonstration, the embedding
    generation is simulated to produce a deterministic, fixed-size vector.

    In a production environment, this node would interface with a dedicated
    embedding model service (e.g., a local model, an external API like OpenAI,
    or a HuggingFace model via an inference endpoint) to perform the actual
    embedding computation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Generates simulated embedding vectors for the input text data.

        Args:
            data (Union[str, List[str]]): The text input(s) for which to generate embeddings.
                                         Can be a single string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the processing pipeline. This might include
                                     model configuration, API keys, or other runtime
                                     parameters in a more advanced implementation.
                                     Currently not utilized by this simulated node.

        Returns:
            Union[List[float], List[List[float]]]: A list of floats representing the
                                                  embedding vector for a single input string,
                                                  or a list of such lists if the input was
                                                  a list of strings.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if a list contains non-string elements.
            RuntimeError: If an unexpected issue occurs during the simulated
                          embedding generation process.
        """
        if not isinstance(data, (str, list)):
            logger.error(
                "EmbeddingsGeneratorNode received invalid data type. Expected 'str' or 'list[str]', got '%s'.",
                type(data).__name__
            )
            raise ValueError(
                f"Invalid input data type for EmbeddingsGeneratorNode. Expected 'str' or 'list[str]', "
                f"got '{type(data).__name__}'."
            )

        if isinstance(data, str):
            texts_to_embed = [data]
            return_single_embedding = True
        else:  # data is a list
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    "EmbeddingsGeneratorNode received a list containing non-string items. "
                    "All elements in the input list must be strings."
                )
                raise ValueError(
                    "List input for EmbeddingsGeneratorNode must contain only strings."
                )
            texts_to_embed = data
            return_single_embedding = False

        try:
            generated_embeddings = []
            # Simulate a common embedding dimension, e.g., 768 for many transformer models
            embedding_dimension = 768

            for text_segment in texts_to_embed:
                # In a real-world scenario, this is where the call to an actual
                # embedding model's API or inference method would occur.
                # For this simulation, we generate a deterministic pseudo-embedding
                # based on the input text's hash to ensure consistency for identical inputs.
                
                # A simple, deterministic way to create a seed from text
                text_hash = sum(ord(c) for c in text_segment) % 1_000_000
                
                # Generate a fixed-size vector of floats based on the hash
                # Values are normalized to be between 0.0 and 1.0 (exclusive of 1.0),
                # mimicking common float ranges for embeddings.
                simulated_embedding = [
                    (text_hash + i * 13) % 99999 / 100000.0
                    for i in range(embedding_dimension)
                ]
                generated_embeddings.append(simulated_embedding)

            logger.debug("Successfully generated embeddings for %d text segments.", len(texts_to_embed))

            if return_single_embedding:
                return generated_embeddings[0]
            else:
                return generated_embeddings

        except Exception as e:
            logger.exception("An unexpected error occurred during simulated embedding generation.")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e