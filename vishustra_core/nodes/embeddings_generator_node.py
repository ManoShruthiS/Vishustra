import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A node designed to transform textual input into high-dimensional vector embeddings.
    
    This node expects a string or a list of strings and returns their corresponding 
    embeddings. It can be configured via the context to use different model providers 
    or specific embedding dimensions.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for the node type.
        """
        return "EmbeddingsGeneratorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        """
        Converts input text into numerical embeddings.
        
        Args:
            data (Any): The input data, expected to be a str or List[str].
            context (Dict[str, Any]): Execution context containing configuration 
                                      such as 'model_provider' or 'api_key'.

        Returns:
            Union[List[float], List[List[float]]]: A single embedding vector or a list of vectors.

        Raises:
            TypeError: If the input data is not a string or list of strings.
            ValueError: If the embedding generation fails due to configuration issues.
        """
        logger.info(f"Node '{self.node_name}' started processing.")

        if not isinstance(data, (str, list)):
            error_msg = f"Invalid data type received: {type(data)}. Expected str or List[str]."
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Extract configuration from context with sensible defaults
            model_provider = context.get("embedding_provider", "mock_provider")
            model_name = context.get("embedding_model", "text-embedding-ada-002")
            
            logger.debug(f"Generating embeddings using provider: {model_provider}, model: {model_name}")

            # Simulated embedding generation logic
            # In a production environment, this would interface with an external API (OpenAI/Cohere)
            # or a local model (Sentence-Transformers/PyTorch).
            result = self._generate_vectors(data)

            logger.info(f"Successfully generated embeddings for input of size: {len(data) if isinstance(data, list) else 1}")
            return result

        except Exception as e:
            logger.exception(f"An error occurred during embedding generation: {str(e)}")
            raise

    def _generate_vectors(self, data: Union[str, List[str]]) -> Any:
        """
        Internal method to simulate the mathematical transformation of text to vectors.
        """
        # Note: In a real-world scenario, replace this mock with actual model inference.
        mock_dimension = 1536
        
        if isinstance(data, str):
            # Simulate a single vector
            return [0.01] * mock_dimension
        
        # Simulate a list of vectors for batch processing
        return [[0.01] * mock_dimension for _ in range(len(data))]

