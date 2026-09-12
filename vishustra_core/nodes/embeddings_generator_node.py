import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is located at vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra node designed to generate vector embeddings for textual input.

    This node accepts either a single string or a list of strings and produces
    corresponding simulated embedding vectors. The dimensionality of these
    embeddings can be configured via the node's context.

    In a production environment, this node would interface with a real
    embedding model (e.g., from OpenAI, HuggingFace, local ONNX model)
    to transform text into dense vector representations suitable for
    downstream tasks like similarity search, clustering, or input to other LLMs.
    For demonstration purposes, it generates random float vectors.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]], None]:
        """
        Generates simulated embeddings for the provided text data.

        This method processes the input `data`, which is expected to be either
        a single string or a list of strings, and returns corresponding
        embedding vector(s). Configuration such as the embedding dimension
        and a descriptive model name can be provided in the `context`.

        Args:
            data: The input text data. Expected types are `str` for a single
                  document or `List[str]` for batch processing.
            context: A dictionary containing operational parameters for the node.
                     - 'embedding_dimension' (int, optional): The target length
                       of the embedding vectors. Defaults to 768 if not provided
                       or invalid. Must be a positive integer.
                     - 'model_name' (str, optional): A label for the embedding
                       model being used/simulated. Defaults to
                       "simulated-embedding-model".

        Returns:
            - `List[float]` if `data` was a single string.
            - `List[List[float]]` if `data` was a list of strings.
            - `None` if the input `data` was `None`.
            - An empty list `[]` if the input `data` was an empty list of strings.
            - For empty or whitespace-only string inputs, a vector of zeros
              of the specified dimension will be returned as its embedding.

        Raises:
            ValueError: If the input `data` is not of an expected type (`str` or `List[str]`).
        """
        # --- Context Parameter Extraction and Validation ---
        embedding_dimension = context.get('embedding_dimension')
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            logger.warning(
                f"[{self.node_name}] Invalid or missing 'embedding_dimension' in context. "
                f"Received '{embedding_dimension}'. Defaulting to 768."
            )
            embedding_dimension = 768

        model_name = context.get('model_name', "simulated-embedding-model")
        if not isinstance(model_name, str):
            logger.warning(
                f"[{self.node_name}] Invalid 'model_name' in context. "
                f"Received '{model_name}'. Defaulting to 'simulated-embedding-model'."
            )
            model_name = "simulated-embedding-model"

        # --- Input Data Validation and Normalization ---
        if data is None:
            logger.info(f"[{self.node_name}] Received None as input data. Returning None.")
            return None

        is_single_text = False
        texts_to_process: List[str] = []

        if isinstance(data, str):
            texts_to_process = [data]
            is_single_text = True
        elif isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                texts_to_process = data
            else:
                logger.error(
                    f"[{self.node_name}] Input list contains non-string elements. "
                    f"Expected List[str], but received List[{[type(item).__name__ for item in data[:3]]}...] "
                    f"for first few elements."
                )
                raise ValueError(
                    f"[{self.node_name}] Invalid input data: list must contain only strings."
                )
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected str or List[str], but received {type(data).__name__}."
            )
            raise ValueError(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected str or List[str], but received {type(data).__name__}."
            )

        if not texts_to_process:
            logger.info(f"[{self.node_name}] Received an empty list of texts. Returning an empty list.")
            return []

        # --- Embedding Generation Simulation ---
        embeddings: List[List[float]] = []
        for i, text_item in enumerate(texts_to_process):
            if not text_item.strip():
                logger.warning(
                    f"[{self.node_name}] Input text at index {i} is empty or whitespace-only. "
                    f"Generating a zero vector of dimension {embedding_dimension}."
                )
                # For empty texts, a common practice is to return a zero vector or handle it specially.
                embeddings.append([0.0] * embedding_dimension)
                continue

            # Simulate embedding generation: In a real implementation, this would
            # involve an API call or model inference.
            simulated_embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dimension)]
            embeddings.append(simulated_embedding)
            logger.debug(
                f"[{self.node_name}] Generated embedding for text (first 30 chars): "
                f"'{text_item[:30].replace('\\n', ' ')}...' "
                f"using simulated model '{model_name}' (dim: {embedding_dimension})."
            )

        logger.info(
            f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings "
            f"using simulated model '{model_name}' with dimension {embedding_dimension}."
        )

        # --- Return Formatted Output ---
        return embeddings[0] if is_single_text else embeddings
