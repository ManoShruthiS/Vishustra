import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A processing node that simulates the generation of vector embeddings for text data.
    
    This node expects input `data` to be either a single string or a list of strings.
    It returns a single embedding (list of floats) if the input was a single string,
    or a list of embeddings if the input was a list of strings.

    In a real-world scenario, this node would interface with an actual embedding model
    (e.g., OpenAI, Hugging Face, Cohere) to generate high-dimensional vectors.
    For this simulation, it generates deterministic placeholder vectors.
    """

    def __init__(self, model_name: str = "simulated-embedding-model", embedding_dimension: int = 128):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            model_name (str): The name of the embedding model to simulate.
                              Primarily for logging and clarity.
            embedding_dimension (int): The desired dimension of the simulated embeddings.
        """
        self._model_name = model_name
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")
        self._embedding_dimension = embedding_dimension
        logger.debug(f"EmbeddingsGeneratorNode initialized with model '{self._model_name}' and dimension {self._embedding_dimension}.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Processes the input data to generate embeddings.

        Expects `data` to be a string or a list of strings.
        Returns a single list of floats (embedding) or a list of lists of floats (embeddings).

        Args:
            data (Any): The text data to embed. Can be a string or a list of strings.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow (e.g., global configs,
                                       shared resources). Not directly used for
                                       simulation logic here, but available.

        Returns:
            Union[List[float], List[List[float]]]: The generated embedding(s).

        Raises:
            ValueError: If the input `data` is not a string or a list of strings.
            RuntimeError: If an unexpected error occurs during embedding simulation.
        """
        logger.info(f"EmbeddingsGeneratorNode '{self.node_name}' starting process for data.")
        is_single_string_input = False
        texts_to_embed: List[str]

        if isinstance(data, str):
            is_single_string_input = True
            texts_to_embed = [data]
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            texts_to_embed = data
        else:
            logger.error(
                "Invalid data type provided to EmbeddingsGeneratorNode. "
                "Expected str or List[str], but received %s.", type(data)
            )
            raise ValueError(
                "EmbeddingsGeneratorNode expects input 'data' to be a string or a list of strings."
            )

        generated_embeddings: List[List[float]] = []
        for i, text in enumerate(texts_to_embed):
            try:
                # Simulate embedding generation
                # This is a deterministic placeholder for a real embedding model.
                # A real model would involve API calls or local model inference.
                seed = len(text) + sum(ord(char) for char in text) % 1000
                
                # Create a deterministic, but unique-ish vector based on the text properties
                simulated_vector = [
                    float((i * seed + j * 17) % 1000) / 1000.0
                    for j in range(self._embedding_dimension)
                ]
                generated_embeddings.append(simulated_vector)
                logger.debug(f"Generated simulated embedding for text item {i+1}/{len(texts_to_embed)} (first 5 dims: {simulated_vector[:5]}).")
            except Exception as e:
                logger.error(
                    f"Failed to simulate embedding for text item {i+1}: '{text[:50]}...' "
                    f"Error: {e}", exc_info=True
                )
                raise RuntimeError(f"Embedding simulation failed for item {i+1}.") from e

        logger.info(f"EmbeddingsGeneratorNode '{self.node_name}' finished processing. Generated {len(generated_embeddings)} embeddings.")

        if is_single_string_input:
            return generated_embeddings[0]
        else:
            return generated_embeddings
