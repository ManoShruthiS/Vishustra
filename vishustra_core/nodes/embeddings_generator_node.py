import logging
from typing import Any, Dict, List, Union, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A node responsible for transforming textual data into numerical vector embeddings.
    
    This node integrates with configurable embedding models (e.g., OpenAI, HuggingFace)
    to facilitate downstream vector search or semantic analysis.
    """

    @property
    def node_name(self) -> str:
        """Returns the canonical name for this node type."""
        return "EmbeddingsGeneratorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates embeddings for the provided input data.
        
        Args:
            data: The input text or list of texts to be vectorized.
            context: Orchestration context containing 'embedding_config' or 'provider' details.
            
        Returns:
            A dictionary containing the generated 'embeddings', 'model_name', and 'usage' statistics.
            
        Raises:
            ValueError: If the input data format is invalid.
            RuntimeError: If the embedding generation fails due to provider issues.
        """
        logger.info(f"[{self.node_name}] Starting embedding generation process.")

        if not data:
            logger.warning(f"[{self.node_name}] Received empty input data.")
            return {"embeddings": [], "model_name": None}

        # Normalize data to a list for consistent processing
        inputs = [data] if isinstance(data, str) else data
        
        if not isinstance(inputs, list):
            raise ValueError(f"Invalid input type: {type(data)}. Expected str or List[str].")

        try:
            # Extract configuration from context
            config = context.get("embedding_config", {})
            provider = config.get("provider", "mock-provider")
            model = config.get("model", "text-embedding-3-small")
            
            logger.debug(f"[{self.node_name}] Using provider: {provider}, model: {model}")

            # Note: In a production environment, this is where the call to a specific 
            # client library (OpenAI, LangChain, or custom REST client) would occur.
            # We simulate the transformation logic here.
            embeddings = self._generate_vectors(inputs, model)

            result = {
                "embeddings": embeddings,
                "model_name": model,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings)
            }
            
            logger.info(f"[{self.node_name}] Successfully generated {len(embeddings)} vectors.")
            return result

        except Exception as e:
            logger.error(f"[{self.node_name}] Failed to generate embeddings: {str(e)}", exc_info=True)
            raise RuntimeError(f"Embedding generation error: {e}") from e

    def _generate_vectors(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Internal utility to interface with the embedding provider.
        Placeholder implementation for vector generation logic.
        """
        # This is a simulation of the vectorization logic.
        # Logic would typically involve: response = self.client.embeddings.create(input=texts, model=model)
        mock_dimension = 1536
        return [[0.0123 * (i + 1) for i in range(mock_dimension)] for _ in texts]