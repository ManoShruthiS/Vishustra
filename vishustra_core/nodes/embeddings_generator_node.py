import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node designed to generate vector embeddings for text.

    This node takes text (either a single string or a list of strings) and
    produces a fixed-size numerical vector representation (embedding) for each
    text input. In a production environment, this would typically interface
    with a pre-trained embedding model (e.g., from Hugging Face, OpenAI, etc.).

    For demonstration, this implementation simulates embedding generation
    using a deterministic pseudo-random process based on the input text.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            embedding_dimension (int): The desired dimension for the output embedding vectors.
                                       Defaults to 768, a common dimension for many contemporary
                                       embedding models.

        Raises:
            ValueError: If `embedding_dimension` is not a positive integer.
        """
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.critical(
                "Initialization failed: embedding_dimension must be a positive integer. Got: %s",
                embedding_dimension,
            )
            raise ValueError("Embedding dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.debug(
            "EmbeddingsGeneratorNode initialized with dimension: %d",
            self._embedding_dimension,
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of an embedding vector for a single string.

        This method generates a fixed-size list of floats acting as a placeholder
        embedding. It uses the hash of the input text as a seed to ensure
        determinism for identical inputs. The generated vector is then L2-normalized.

        Args:
            text (str): The input string to generate an embedding for.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        # Use a deterministic pseudo-random number generator for consistent outputs
        # for the same input text.
        seed = hash(text) % (2**31 - 1)  # Ensure seed is within a reasonable int range
        rng = random.Random(seed)

        # Generate a vector of random floats within a typical range (-1.0 to 1.0)
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(self._embedding_dimension)]

        # L2-normalize the vector, a common practice for embeddings
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 1e-6:  # Avoid division by zero for extremely small magnitudes
            embedding = [x / magnitude for x in embedding]
        else:
            # If magnitude is zero, return a zero vector (or handle as appropriate)
            embedding = [0.0] * self._embedding_dimension
            logger.warning("Generated zero-magnitude embedding for text: '%s'", text)

        return embedding

    def process(
        self, data: Union[str, List[str]], context: Dict[str, Any]
    ) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embedding vectors.

        Args:
            data (Union[str, List[str]]): The text or list of texts for which
                                          embeddings need to be generated.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. This simulation does
                                       not extensively use the context beyond logging.

        Returns:
            Union[List[float], List[List[float]]]:
                - If `data` was a single string, returns a single embedding vector (List[float]).
                - If `data` was a list of strings, returns a list of embedding vectors (List[List[float]]).

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if a list contains non-string elements.
            RuntimeError: For any unexpected errors during the embedding generation process.
        """
        logger.info(
            "EmbeddingsGeneratorNode '%s' starting process. Input data type: %s.",
            self.node_name,
            type(data).__name__,
        )
        logger.debug(
            "Processing data with embedding dimension: %d", self._embedding_dimension
        )

        try:
            if isinstance(data, str):
                logger.debug("Generating embedding for a single string input.")
                embedding = self._generate_single_embedding(data)
                logger.debug("Successfully generated embedding for single input.")
                return embedding
            elif isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    invalid_types = [
                        type(item).__name__ for item in data if not isinstance(item, str)
                    ]
                    logger.error(
                        "Input list contains non-string elements. Expected List[str], found types: %s.",
                        invalid_types,
                    )
                    raise TypeError(
                        f"All elements in the input list must be strings. Found non-string types: {invalid_types}."
                    )
                logger.debug("Generating embeddings for a list of %d strings.", len(data))
                embeddings = [self._generate_single_embedding(item) for item in data]
                logger.debug(
                    "Successfully generated %d embeddings for the list input.",
                    len(embeddings),
                )
                return embeddings
            else:
                logger.error(
                    "Invalid input data type for EmbeddingsGenerator. Expected str or List[str], got: %s.",
                    type(data).__name__,
                )
                raise TypeError(
                    f"Invalid input data type. Expected str or List[str], but received {type(data).__name__}."
                )
        except TypeError:
            # Re-raise TypeErrors as they indicate invalid input, allowing upstream to handle
            logger.exception("Type error encountered during embedding generation.")
            raise
        except Exception as e:
            # Catch any other unforeseen exceptions during the process
            logger.exception(
                "An unexpected error occurred during embedding generation: %s", e
            )
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e