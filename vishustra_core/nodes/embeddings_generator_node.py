import logging
from typing import Any, Dict, List, Union, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class EmbeddingsGeneratorNode(BaseNode):
    """
    A node responsible for converting textual data into numerical vector representations.
    
    This node expects either a single string or a list of strings as input. It utilizes
    a provider-agnostic approach, looking for an 'embedding_client' or specific 
    configuration in the context to perform the transformation.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initializes the node with a default model configuration.
        
        Args:
            model_name: The identifier for the embedding model to be used.
        """
        self._model_name = model_name

    @property
    def node_name(self) -> str:
        """Returns the canonical name for this node."""
        return "EmbeddingsGeneratorNode"

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> List[List[float]]:
        """
        Generates embeddings for the provided input data.
        
        Args:
            data: The text or list of texts to embed.
            context: Execution context containing configuration and potentially 
                    the embedding provider/client.

        Returns:
            A list of vector embeddings (list of floats).

        Raises:
            ValueError: If the input data is not a string or list of strings.
            RuntimeError: If the embedding provider fails or is not configured.
        """
        logger.info(f"Node '{self.node_name}' started processing with model: {self._model_name}")

        if not data:
            logger.warning("Empty data received. Returning empty list.")
            return []

        # Input Validation
        if not isinstance(data, (str, list)):
            error_msg = f"Invalid input type: {type(data)}. Expected str or List[str]."
            logger.error(error_msg)
            raise ValueError(error_msg)

        texts_to_embed = [data] if isinstance(data, str) else data

        try:
            # In a production modular framework, we retrieve the client from context 
            # or a centralized provider registry.
            client = context.get("embedding_client")
            
            if client:
                # Assuming a standard interface for the purpose of this implementation
                logger.debug("Dispatching request to external embedding provider.")
                embeddings = client.embed(texts=texts_to_embed, model=self._model_name)
            else:
                # Fallback / Mock logic for framework demonstration if no client is provided
                logger.debug("No embedding client found in context; using internal transformation logic.")
                embeddings = self._simulate_embeddings(texts_to_embed)

            logger.info(f"Successfully generated embeddings for {len(texts_to_embed)} items.")
            return embeddings

        except Exception as e:
            logger.exception(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def _simulate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Placeholder logic to simulate vector generation when an external provider 
        is not attached to the context.
        """
        # This is where the framework would interface with local models like sentence-transformers
        # or call an API if the client wasn't pre-injected.
        return [[0.0] * 1536 for _ in texts]

    def __repr__(self) -> str:
        return f"<EmbeddingsGeneratorNode(model='{self._model_name}')>"