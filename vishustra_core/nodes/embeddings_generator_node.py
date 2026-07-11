import logging
import hashlib
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGenerator(BaseNode):
    """
    A Vishustra node designed to generate vector embeddings for text data.

    This node accepts either a single string or a list of strings as input
    and simulates the process of transforming them into numerical vector
    representations (embeddings). For demonstration and testing purposes,
    it uses a deterministic, hash-based approach to generate these embeddings
    without relying on external ML models or APIs.
    """

    _EMBEDDING_DIMENSION: int = 8  # Fixed dimension for simulated embeddings

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "EmbeddingsGenerator"

    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Simulates the generation of an embedding vector for a single input string.

        This method produces a deterministic vector based on the input string's hash
        and length. In a production system, this would typically involve an
        inference call to an actual embedding model (e.g., from OpenAI, Hugging Face, etc.).

        Args:
            text: The string for which to generate an embedding.

        Returns:
            A list of floats representing the generated embedding vector.
        """
        # Using SHA-256 hash for a more robust and deterministic seed
        # In a real scenario, this would be model-dependent logic.
        text_bytes = text.encode('utf-8')
        hash_digest = hashlib.sha256(text_bytes).hexdigest()
        seed = int(hash_digest, 16) % (10**9) # Ensure seed is within a manageable integer range

        embedding: List[float] = []
        for i in range(self._EMBEDDING_DIMENSION):
            # Generate a value between 0.0 and 1.0, ensuring determinism
            # based on the seed, index, and text length.
            # This is a simple placeholder for complex model logic.
            value = (seed + i * len(text) * 7) % 1000 / 999.0
            embedding.append(value)
        return embedding

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        This method expects `data` to be either a single string or a list of strings.
        It then generates a corresponding embedding vector or a list of embedding vectors.
        The `context` dictionary can be leveraged for configuration in a real scenario
        (e.g., specifying an embedding model, API keys), though for this simulated node,
        its primary role is to satisfy the interface.

        Args:
            data: The input text (`str`) or a list of texts (`List[str]`) for which
                  embeddings are to be generated.
            context: A dictionary holding node-specific or global contextual information.
                     For this simulated node, it's primarily a placeholder.

        Returns:
            If `data` is a `str`, returns `List[float]`.
            If `data` is `List[str]`, returns `List[List[float]]`.
            Each inner list represents an embedding vector.

        Raises:
            ValueError: If the input `data` is not a string or a list of strings,
                        or if a list contains non-string elements.
        """
        logger.debug(f"[{self.node_name}] Starting processing. Input data type: {type(data)}")
        if context:
            logger.debug(f"[{self.node_name}] Context keys received: {list(context.keys())}")

        if isinstance(data, str):
            logger.info(f"[{self.node_name}] Generating embedding for a single text input.")
            return self._generate_single_embedding(data)
        elif isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. "
                    f"Expected list of strings, but received types: {[type(item) for item in data if not isinstance(item, str)]}"
                )
                raise ValueError(
                    f"Invalid input data: Expected a list of strings, but found non-string elements "
                    f"in the input list for node '{self.node_name}'."
                )
            logger.info(f"[{self.node_name}] Generating embeddings for a list of {len(data)} text inputs.")
            return [self._generate_single_embedding(item) for item in data]
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected `str` or `List[str]`, "
                f"but received `{type(data).__name__}`."
            )
            raise ValueError(
                f"Invalid input data type for '{self.node_name}' node. "
                f"Expected `str` or `List[str]`, got `{type(data).__name__}`."
            )