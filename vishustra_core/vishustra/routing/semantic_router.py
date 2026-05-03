import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

from pydantic import BaseModel, Field

# Setup logging for the module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Abstract Base Classes (ABCs) for Vishustra Components ---

class AbstractEmbeddingModel(ABC):
    """
    Abstract Base Class for Vishustra's embedding models.
    Concrete implementations will provide specific embedding logic (e.g., OpenAI, HuggingFace, local models).
    """
    @abstractmethod
    async def aembed(self, text: str) -> List[float]:
        """
        Asynchronously embeds a single text string into a vector.

        Args:
            text: The text string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        raise NotImplementedError

    @abstractmethod
    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a batch of text strings into vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, where each vector is a list of floats.
        """
        raise NotImplementedError

class AbstractVectorStore(ABC):
    """
    Abstract Base Class for Vishustra's vector store connectors.
    Concrete implementations will interface with specific vector databases (e.g., Qdrant, Pinecone, Chroma, FAISS).
    """
    @abstractmethod
    async def aadd_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """
        Asynchronously adds vectors and their associated metadata to the vector store.

        Args:
            vectors: A list of embedding vectors.
            metadatas: A list of dictionaries, where each dictionary contains metadata
                       corresponding to a vector at the same index.
            ids: Optional list of unique identifiers for the vectors. If not provided,
                 the store might generate them.
        """
        raise NotImplementedError

    @abstractmethod
    async def asearch(self, query_vector: List[float], k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Asynchronously performs a similarity search in the vector store.

        Args:
            query_vector: The embedding vector of the query.
            k: The number of top similar results to return.

        Returns:
            A list of tuples, where each tuple contains:
            - A dictionary of metadata for the retrieved vector.
            - The similarity score (e.g., cosine similarity) of the retrieved vector
              to the query vector.
            Results are typically sorted by score in descending order.
        """
        raise NotImplementedError

# --- Pydantic Data Models ---

class RoutingTarget(BaseModel):
    """
    Defines a target for the semantic router. Each target represents a specific
    downstream action, chain, or tool within Vishustra.
    """
    name: str = Field(..., description="A unique, human-readable name for the routing target.")
    description: str = Field(..., description="A detailed description of what this target handles or does.")
    example_queries: List[str] = Field(default_factory=list,
                                       description="A list of example user queries that should route to this target.")
    handler_identifier: str = Field(..., description="A unique identifier (e.g., function name, chain ID, tool key) "
                                                     "that Vishustra uses to invoke the actual handler for this target.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional arbitrary metadata for the target.")

    @property
    def training_texts(self) -> List[str]:
        """Combines description and example queries for embedding."""
        texts = [self.description]
        if self.example_queries:
            texts.extend(self.example_queries)
        return texts

class RouteMatch(BaseModel):
    """
    Represents a successful match from the semantic router, indicating the best
    target and its similarity score.
    """
    target: RoutingTarget = Field(..., description="The matched routing target.")
    score: float = Field(..., description="The similarity score (e.g., cosine similarity) of the match.")

# --- Semantic Router Core Logic ---

class SemanticRouter:
    """
    The SemanticRouter intelligently dispatches incoming user queries to the most
    relevant pre-defined 'routing targets' within Vishustra. It uses an embedding model
    and a vector store to find the semantic closest target based on its description
    and example queries.

    This enables dynamic and context-aware routing, allowing Vishustra to adapt
    to diverse user intents without explicit keyword matching.

    The router's index, mapping target information to embeddings in the vector store,
    must be asynchronously built via `ainitialize()` before `aroute()` can be used.
    """
    _is_initialized: bool = False

    def __init__(self,
                 embedding_model: AbstractEmbeddingModel,
                 vector_store: AbstractVectorStore,
                 targets: List[RoutingTarget],
                 confidence_threshold: float = 0.7,
                 max_target_examples_per_entry: int = 5):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of a concrete AbstractEmbeddingModel.
            vector_store: An instance of a concrete AbstractVectorStore.
            targets: A list of RoutingTarget objects that this router can dispatch to.
            confidence_threshold: The minimum similarity score required for a match
                                  to be considered valid. Scores below this will result
                                  in no route being returned.
            max_target_examples_per_entry: Maximum number of example queries from a
                                           single target to bundle into one entry in
                                           the vector store. Helps manage vector store
                                           granularity and embedding costs.
        """
        if not isinstance(embedding_model, AbstractEmbeddingModel):
            raise TypeError("embedding_model must be an instance of AbstractEmbeddingModel")
        if not isinstance(vector_store, AbstractVectorStore):
            raise TypeError("vector_store must be an instance of AbstractVectorStore")
        if not all(isinstance(t, RoutingTarget) for t in targets):
            raise TypeError("All items in 'targets' must be instances of RoutingTarget")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        # Map targets by handler_identifier for quick lookup after a vector store match
        self._targets_map = {t.handler_identifier: t for t in targets}
        self._confidence_threshold = confidence_threshold
        self._max_target_examples_per_entry = max_target_examples_per_entry
        self._is_initialized = False
        logger.info(f"SemanticRouter initialized with {len(targets)} targets. "
                    "Call ainitialize() to build the routing index.")

    async def ainitialize(self):
        """
        Asynchronously initializes the semantic router by embedding all target descriptions
        and example queries, then adding them to the vector store.
        This method must be called once before `aroute` can be used.
        """
        if self._is_initialized:
            logger.warning("SemanticRouter is already initialized. Skipping re-initialization.")
            return

        logger.info("Initializing SemanticRouter index by embedding target descriptions and examples...")
        vectors_to_add: List[List[float]] = []
        metadatas_to_add: List[Dict[str, Any]] = []

        # Prepare texts for batch embedding
        target_texts_to_embed: List[str] = []
        # Store corresponding handler_identifier for each text to link back after embedding
        # and vector store lookup.
        text_handler_identifiers: List[str] = []

        for target in self._targets_map.values():
            # Add description as a primary embedding point
            target_texts_to_embed.append(target.description)
            text_handler_identifiers.append(target.handler_identifier)

            # Process example queries, potentially in chunks
            if target.example_queries:
                for i in range(0, len(target.example_queries), self._max_target_examples_per_entry):
                    chunk = target.example_queries[i : i + self._max_target_examples_per_entry]
                    # Combine example queries into a single string for embedding
                    # This can be adjusted to embed each example separately if finer granularity is desired
                    target_texts_to_embed.append("\n".join(chunk))
                    text_handler_identifiers.append(target.handler_identifier)

        if not target_texts_to_embed:
            logger.warning("No training texts found for targets. Router index will be empty and ineffective.")
            self._is_initialized = True
            return

        logger.debug(f"Embedding {len(target_texts_to_embed)} training texts...")
        embedded_texts = await self._embedding_model.aembed_batch(target_texts_to_embed)

        for i, vector in enumerate(embedded_texts):
            vectors_to_add.append(vector)
            # Link back to the original RoutingTarget using handler_identifier
            metadata = {
                "handler_identifier": text_handler_identifiers[i],
                "original_text": target_texts_to_embed[i],
                # Include additional metadata for debugging/context
                "target_name": next((t.name for t in self._targets_map.values() if t.handler_identifier == text_handler_identifiers[i]), "unknown"),
            }
            # Add any custom metadata from the original RoutingTarget
            original_target = self._targets_map.get(text_handler_identifiers[i])
            if original_target and original_target.metadata:
                metadata.update(original_target.metadata)
            metadatas_to_add.append(metadata)


        logger.info(f"Adding {len(vectors_to_add)} vectors to the vector store for routing index...")
        await self._vector_store.aadd_vectors(vectors_to_add, metadatas_to_add)
        self._is_initialized = True
        logger.info("SemanticRouter index built successfully.")

    async def aroute(self, query: str) -> Optional[RouteMatch]:
        """
        Asynchronously routes an incoming user query to the most semantically
        similar target.

        Args:
            query: The user's input query string.

        Returns:
            An optional RouteMatch object if a suitable target is found above
            the confidence threshold, otherwise None.
        """
        if not self._is_initialized:
            raise RuntimeError("SemanticRouter must be initialized by calling ainitialize() before aroute().")

        if not query:
            logger.warning("Received empty query, returning no route.")
            return None

        logger.debug(f"Attempting to route query: '{query}'")

        query_vector = await self._embedding_model.aembed(query)

        # Search for the top K results. K=1 is usually sufficient for simple routing,
        # but could be expanded for more complex fallbacks or multi-route scenarios.
        search_results = await self._vector_store.asearch(query_vector, k=1)

        if not search_results:
            logger.info(f"No semantic routes found in vector store for query: '{query}'")
            return None

        # The search result is a tuple: (metadata_dict, similarity_score)
        best_match_metadata, best_match_score = search_results[0]

        if best_match_score < self._confidence_threshold:
            logger.info(f"Best route for query '{query}' (score: {best_match_score:.2f}) "
                        f"is below the confidence threshold ({self._confidence_threshold:.2f}). No route returned.")
            return None

        handler_identifier = best_match_metadata.get("handler_identifier")
        if not handler_identifier:
            logger.error(f"Vector store returned a match without a 'handler_identifier' in its metadata: "
                         f"{best_match_metadata}. Cannot map to a RoutingTarget.")
            return None

        matched_target = self._targets_map.get(handler_identifier)

        if not matched_target:
            logger.error(f"Handler identifier '{handler_identifier}' found in vector store metadata but "
                         "corresponding RoutingTarget not found in the router's internal targets map. "
                         "This indicates a data inconsistency or misconfiguration.")
            return None

        logger.info(f"Query '{query}' successfully routed to '{matched_target.name}' "
                    f"with score {best_match_score:.2f} (Handler: {matched_target.handler_identifier})")
        return RouteMatch(target=matched_target, score=best_match_score)