import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable
import uuid

# Configure basic logging for the module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Type variable for embedding vectors
Embedding = List[float]
T = TypeVar('T')

@runtime_checkable
class EmbeddingModel(Protocol):
    """
    Protocol for an embedding model.

    Any class implementing this protocol must provide an `aembed` method
    that takes a list of texts and returns a list of corresponding embeddings.
    """
    async def aembed(self, texts: List[str]) -> List[Embedding]:
        """
        Asynchronously embeds a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            The order of embeddings matches the order of input texts.
        """
        ...

@dataclass(frozen=True)
class Route:
    """
    Represents a specific routing target within the framework.

    Attributes:
        name: A unique identifier for the route (e.g., "summarize_document").
        description: A human-readable description of what this route does.
                     This description is used for embedding and semantic matching.
        metadata: Optional dictionary for additional information relevant
                  to the route (e.g., required parameters, associated chain ID).
        route_id: A unique UUID for this route, auto-generated if not provided.
    """
    name: str
    description: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Route 'name' must be a non-empty string.")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Route 'description' must be a non-empty string.")
        if not isinstance(self.metadata, (dict, type(None))):
            raise ValueError("Route 'metadata' must be a dictionary or None.")

@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol for a vector store used to store and retrieve route embeddings.

    Implementations should handle persistence and efficient similarity search.
    """
    async def aupsert_vectors(self, ids: List[str], vectors: List[Embedding], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Asynchronously inserts or updates vectors in the store.

        Args:
            ids: A list of unique identifiers for the vectors.
            vectors: A list of embedding vectors to store.
            metadata: An optional list of dictionaries, where each dictionary
                      corresponds to the metadata for a vector at the same index.
        """
        ...

    async def aquery(self, query_vector: Embedding, top_k: int = 1, min_score: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Asynchronously queries the vector store for similar vectors.

        Args:
            query_vector: The embedding vector to query with.
            top_k: The maximum number of most similar results to return.
            min_score: The minimum similarity score for a result to be included.

        Returns:
            A list of tuples, where each tuple contains:
            (vector_id: str, similarity_score: float, metadata: Dict[str, Any]).
        """
        ...

    async def adelete_vectors(self, ids: List[str]) -> None:
        """
        Asynchronously deletes vectors from the store by their IDs.

        Args:
            ids: A list of vector IDs to delete.
        """
        ...

    async def aget_vector_count(self) -> int:
        """
        Asynchronously returns the total number of vectors currently in the store.
        """
        ...


class SemanticRouter:
    """
    The SemanticRouter orchestrates routing incoming queries to the most
    semantically relevant predefined 'Route' based on embedding similarity.

    It leverages an `EmbeddingModel` to convert text into vector embeddings
    and a `VectorStore` to efficiently store and query these embeddings.

    This enables dynamic, intelligent routing within Vishustra, allowing the
    framework to autonomously direct a user's request to the appropriate
    backend process, tool, or sub-chain.
    """
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 similarity_threshold: float = 0.75):
        """
        Initializes the SemanticRouter with an embedding model and a vector store.

        Args:
            embedding_model: An instance of an object implementing the EmbeddingModel protocol.
                             Used to generate embeddings for routes and queries.
            vector_store: An instance of an object implementing the VectorStore protocol.
                          Used to store and search for route embeddings.
            similarity_threshold: The minimum similarity score required for a query
                                  to be considered a match for a route. Default is 0.75.
                                  Scores below this will not be returned.
        Raises:
            TypeError: If embedding_model or vector_store do not conform to their protocols.
            ValueError: If similarity_threshold is not between 0 and 1.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must implement the EmbeddingModel protocol.")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must implement the VectorStore protocol.")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._similarity_threshold = similarity_threshold
        logger.info(f"SemanticRouter initialized with threshold: {self._similarity_threshold}")

    async def add_route(self, route: Route) -> None:
        """
        Adds a new route to the semantic router.

        The route's description is embedded and stored in the vector store,
        allowing it to be semantically matched later. If a route with the
        same `route_id` already exists, it will be updated.

        Args:
            route: The Route object to add.
        """
        try:
            route_embedding_list = await self._embedding_model.aembed([route.description])
            if not route_embedding_list:
                logger.error(f"Failed to generate embedding for route '{route.name}'. Description: '{route.description}'")
                raise ValueError("Embedding generation failed for route description.")

            route_embedding = route_embedding_list[0]
            await self._vector_store.aupsert_vectors(
                ids=[route.route_id],
                vectors=[route_embedding],
                metadata=[{'name': route.name, 'description': route.description, 'full_metadata': route.metadata}]
            )
            logger.info(f"Route '{route.name}' (ID: {route.route_id}) added/updated successfully.")
        except Exception as e:
            logger.error(f"Error adding route '{route.name}' (ID: {route.route_id}): {e}", exc_info=True)
            raise

    async def add_routes(self, routes: List[Route]) -> None:
        """
        Adds multiple routes to the semantic router in a batch.

        This method optimizes for batch embedding and upsert operations.

        Args:
            routes: A list of Route objects to add.
        """
        if not routes:
            logger.warning("Attempted to add an empty list of routes.")
            return

        descriptions = [route.description for route in routes]
        route_ids = [route.route_id for route in routes]
        metadatas = [{'name': r.name, 'description': r.description, 'full_metadata': r.metadata} for r in routes]

        try:
            embeddings = await self._embedding_model.aembed(descriptions)
            if len(embeddings) != len(routes):
                logger.error(f"Mismatch in number of embeddings generated ({len(embeddings)}) "
                             f"vs. number of routes provided ({len(routes)}).")
                raise RuntimeError("Embedding generation for batch routes failed partially.")

            await self._vector_store.aupsert_vectors(
                ids=route_ids,
                vectors=embeddings,
                metadata=metadatas
            )
            logger.info(f"Successfully added/updated {len(routes)} routes.")
        except Exception as e:
            logger.error(f"Error adding batch of routes: {e}", exc_info=True)
            raise

    async def remove_route(self, route_id: str) -> None:
        """
        Removes a route from the semantic router by its unique ID.

        Args:
            route_id: The unique identifier of the route to remove.
        """
        try:
            await self._vector_store.adelete_vectors(ids=[route_id])
            logger.info(f"Route with ID '{route_id}' removed successfully.")
        except Exception as e:
            logger.error(f"Error removing route with ID '{route_id}': {e}", exc_info=True)
            raise

    async def route_query(self, query_text: str, top_k: int = 1) -> List[Tuple[Route, float]]:
        """
        Semantically routes an incoming query to the most relevant predefined Route.

        The query text is embedded, and a similarity search is performed against
        all stored route embeddings. Only routes with a similarity score
        above the `similarity_threshold` are considered.

        Args:
            query_text: The input query string to be routed.
            top_k: The maximum number of best matching routes to return.

        Returns:
            A list of tuples, each containing a matched `Route` object and its
            similarity score, sorted by score in descending order. An empty
            list is returned if no route meets the similarity threshold.
        """
        if not query_text:
            logger.warning("Empty query_text provided for routing.")
            return []

        try:
            query_embedding_list = await self._embedding_model.aembed([query_text])
            if not query_embedding_list:
                logger.error(f"Failed to generate embedding for query: '{query_text}'")
                return []

            query_embedding = query_embedding_list[0]
            results = await self._vector_store.aquery(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=self._similarity_threshold
            )

            routed_results: List[Tuple[Route, float]] = []
            for route_id, score, metadata in results:
                try:
                    # Reconstruct the Route object from stored metadata
                    full_metadata = metadata.get('full_metadata', {})
                    reconstructed_route = Route(
                        name=metadata['name'],
                        description=metadata['description'],
                        metadata=full_metadata,
                        route_id=route_id
                    )
                    routed_results.append((reconstructed_route, score))
                    logger.debug(f"Query matched route '{reconstructed_route.name}' (ID: {route_id}) "
                                 f"with score: {score:.4f}")
                except KeyError as e:
                    logger.error(f"Missing key in metadata for route ID '{route_id}' during reconstruction: {e}")
                    continue

            # Sort results by score in descending order (vector stores usually return sorted, but good to ensure)
            routed_results.sort(key=lambda x: x[1], reverse=True)
            return routed_results
        except Exception as e:
            logger.error(f"Error routing query '{query_text}': {e}", exc_info=True)
            return []

    async def get_route_count(self) -> int:
        """
        Asynchronously retrieves the total number of routes currently stored
        in the underlying vector store.

        Returns:
            The number of routes.
        """
        try:
            count = await self._vector_store.aget_vector_count()
            logger.debug(f"Total routes in store: {count}")
            return count
        except Exception as e:
            logger.error(f"Error retrieving route count: {e}", exc_info=True)
            return 0

    @property
    def similarity_threshold(self) -> float:
        """Returns the current similarity threshold."""
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        """Sets a new similarity threshold."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
        self._similarity_threshold = value
        logger.info(f"Similarity threshold updated to {self._similarity_threshold}")