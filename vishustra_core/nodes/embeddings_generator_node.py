import logging
import random
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A Vishustra processing node responsible for generating vector embeddings
    for given text data.

    This node simulates the interaction with an external embedding service
    or an internal embedding model to transform textual input into numerical
    vector representations.
    """
    _DEFAULT_EMBEDDING_DIMENSION = 768
    _DEFAULT_MODEL_NAME = "simulated-text-embedding-model-v1"

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION):
        """
        Initializes the EmbeddingsGeneratorNode.

        Args:
            model_name (str): The name of the embedding model to simulate.
            embedding_dimension (int): The dimension of the simulated embedding vectors.
        """
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        logger.info(f"EmbeddingsGeneratorNode initialized with model: '{self.model_name}' "
                    f"and dimension: {self.embedding_dimension}.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "EmbeddingsGenerator"

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generates a mock embedding vector for a given text.
        In a real scenario, this would call an actual embedding model.
        """
        # Simple hash-based seeding for deterministic-ish mock embeddings for same text
        seed = sum(ord(char) for char in text) % 1000000
        random.seed(seed)
        return [random.uniform(-1.0, 1.0) for _ in range(self.embedding_dimension)]

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[List[float]]:
        """
        Processes the input text data to generate embedding vectors.

        Args:
            data (Union[str, List[str]]): The text or list of texts to embed.
                                          Each string should be a distinct piece of text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing, such as run metadata or
                                      configuration overrides.

        Returns:
            List[List[float]]: A list of embedding vectors. Each inner list is a
                               vector representation for a corresponding input text.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
            ValueError: If the input list contains non-string elements.
            Exception: For unexpected errors during embedding generation.
        """
        logger.debug(f"[{self.node_name}] Starting process with data type: {type(data).__name__}.")
        
        input_texts: List[str] = []
        if isinstance(data, str):
            input_texts = [data]
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, str):
                    logger.error(f"[{self.node_name}] Invalid item type in list: expected str, got {type(item).__name__}.")
                    raise ValueError(f"Input list must only contain strings, but found type {type(item).__name__}.")
                input_texts.append(item)
        else:
            logger.error(f"[{self.node_name}] Invalid input data type: expected str or List[str], got {type(data).__name__}.")
            raise TypeError(f"Input data must be a string or a list of strings, got {type(data).__name__}.")

        embeddings: List[List[float]] = []
        try:
            for i, text in enumerate(input_texts):
                # Simulate embedding generation
                embedding = self._generate_mock_embedding(text)
                embeddings.append(embedding)
                logger.debug(f"[{self.node_name}] Generated embedding for text piece {i+1}/{len(input_texts)}. "
                             f"Vector starts with: {embedding[:5]}...") # Log first few elements
            
            logger.info(f"[{self.node_name}] Successfully generated {len(embeddings)} embeddings "
                        f"using model '{self.model_name}'.")
            
            # Example of using context (not strictly necessary for this simulation but shows capability)
            if 'trace_id' in context:
                logger.debug(f"[{self.node_name}] Context trace_id: {context['trace_id']}")

            return embeddings
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during embedding generation.")
            raise # Re-raise the exception after logging