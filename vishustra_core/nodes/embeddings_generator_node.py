import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    EmbeddingsGeneratorNode handles the transformation of textual data into high-dimensional 
    vector representations. It supports both single string inputs and batches of text.
    
    This node is designed to interface with various embedding providers, utilizing the 
    provided context for configuration and model selection.
    """

    def __init__(self, provider: str = "openai", default_model: str = "text-embedding-3-small"):
        """
        Initializes the node with a specific provider and default model.
        """
        self.provider = provider
        self.default_model = default_model

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "EmbeddingsGeneratorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to generate embeddings.
        
        Args:
            data: The text data to be vectorized. Can be a string or a list of strings.
            context: Orchestration context containing operational metadata, 
                     API keys, and model overrides.
            
        Returns:
            A dictionary containing the generated embeddings and execution metadata.
            
        Raises:
            ValueError: If the input data is null or empty.
            TypeError: If the input data format is not supported.
        """
        try:
            if data is None or (isinstance(data, (str, list)) and len(data) == 0):
                raise ValueError(f"[{self.node_name}] Input data cannot be empty.")

            # Extract configuration from context or use defaults
            model = context.get("embedding_model", self.default_model)
            api_key = context.get("api_key")

            logger.info(f"Generating embeddings using provider: {self.provider}, model: {model}")

            # Normalize input to a list for consistent processing
            input_texts = [data] if isinstance(data, str) else data
            
            if not isinstance(input_texts, list):
                raise TypeError(f"[{self.node_name}] Unsupported data type: {type(data)}. Expected str or list.")

            # Simulate the transformation logic
            # In a production scenario, this would invoke the specific provider client (e.g., OpenAI, HuggingFace)
            # Example: response = self.client.embeddings.create(input=input_texts, model=model)
            
            embedding_results = self._generate_simulated_embeddings(input_texts)

            result = {
                "embeddings": embedding_results,
                "metadata": {
                    "node": self.node_name,
                    "provider": self.provider,
                    "model": model,
                    "batch_size": len(input_texts),
                    "dimensions": 1536  # Standard for the default model
                }
            }

            logger.debug(f"Successfully processed {len(input_texts)} text sequences.")
            return result

        except Exception as e:
            logger.error(f"Critical failure in {self.node_name}: {str(e)}", exc_info=True)
            raise

    def _generate_simulated_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Internal utility to simulate vector generation.
        In integration, this will be replaced by the actual provider SDK call.
        """
        # Simulated 1536-dimension vector for demonstration purposes
        return [[0.0123] * 1536 for _ in texts]

if __name__ == "__main__":
    # Internal component testing block
    node = EmbeddingsGeneratorNode()
    sample_context = {"embedding_model": "text-embedding-3-large"}
    try:
        output = node.process("Sample text for vectorization.", sample_context)
    except Exception:
        pass